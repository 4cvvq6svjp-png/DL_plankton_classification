# coding: utf-8

import logging
import sys
import os
import pathlib
import signal

import yaml
import torch
import torchinfo.torchinfo as torchinfo
from torch.utils.tensorboard import SummaryWriter
import datetime

from . import data
from . import models
from . import optim
from . import utils


_SIGTERM_RECEIVED = False


def _sigterm_handler(signum, frame):
    global _SIGTERM_RECEIVED
    _SIGTERM_RECEIVED = True
    logging.warning("SIGTERM reçu — sauvegarde en cours avant arrêt...")


def _save_last_checkpoint(model, optimizer, scheduler_state, epoch, logdir):
    path = str(logdir / "last_checkpoint.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler_state,
    }, path)
    logging.info(f"Checkpoint sauvegardé : {path} (epoch {epoch + 1})")


def train(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # ── Data ──────────────────────────────────
    logging.info("= Building the dataloaders")
    data_config = config["data"]
    train_loader, valid_loader, input_size, num_classes, class_weights = \
        data.get_dataloaders(data_config, use_cuda)

    # ── Model ─────────────────────────────────
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)

    # ── Resume from checkpoint ────────────────
    resume_cfg = config.get("resume", {})
    resume_path = resume_cfg.get("checkpoint", None)
    resumed = False
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        resumed = True
        logging.info(f"  ✓ Poids chargés depuis {resume_path}")

    # ── Loss ──────────────────────────────────
    logging.info("= Loss")
    loss_cfg = config.get("loss", {"class": "CrossEntropyLoss"})
    loss_class_name = loss_cfg.get("class", "CrossEntropyLoss")

    if loss_class_name == "FocalLoss":
        use_alpha = loss_cfg.get("use_class_weights", True)
        alpha = class_weights.to(device) if use_alpha else None
        loss_fn = optim.FocalLoss(
            alpha=alpha,
            gamma=loss_cfg.get("gamma", 2.0),
            label_smoothing=loss_cfg.get("label_smoothing", 0.0),
        )
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    # ── Logging ───────────────────────────────
    logging_config = config["logging"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logname = f"{model_config['class']}_{timestamp}"

    logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
    os.makedirs(logdir, exist_ok=True)
    logging.info(f"Will be logging into {logdir}")

    tensorboard_writer = SummaryWriter(logdir)
    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    input_size_batch = next(iter(train_loader))[0].shape
    summary_text = (
        f"Logdir : {logdir}\n"
        f"## Command \n{' '.join(sys.argv)}\n\n"
        f" Config : {config} \n\n"
        f"## Summary of the model architecture\n"
        f"{torchinfo.summary(model, input_size=input_size_batch)}\n\n"
        f"## Loss\n\n{loss_fn}\n\n"
        f"## Datasets : \n"
        f"Train : {train_loader.dataset}\n"
        f"Validation : {valid_loader.dataset}"
    )
    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)

    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=False
    )

    signal.signal(signal.SIGTERM, _sigterm_handler)
    checkpoint_every = config.get("checkpoint_every", 5)

    # ── AMP ───────────────────────────────────
    use_amp = config.get("use_amp", True) and use_cuda
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ── Mixup / CutMix config ─────────────────
    mix_cfg = config.get("mix", {})
    mixup_alpha = mix_cfg.get("mixup_alpha", 0.0)
    cutmix_alpha = mix_cfg.get("cutmix_alpha", 0.0)
    mix_prob = mix_cfg.get("prob", 0.5)

    has_backbone = (
        (hasattr(model, "model") and hasattr(model.model, "base_model"))
        or (hasattr(model, "model") and hasattr(model.model, "fc"))
        or (hasattr(model, "model") and hasattr(model.model, "classifier"))
    )
    optim_config = config.get("optim", {})

    # ════════════════════
    #  PHASE 1 : WARM-UP
    # ════════════════════
    warmup_epochs = 0 if resumed else config.get("warmup_epochs", 0)
    global_epoch = 0

    if warmup_epochs > 0 and has_backbone:
        head_lr = float(optim_config.get("head_lr", 1e-3))
        logging.info(f"\n=== PHASE 1 : WARM-UP ({warmup_epochs} epochs, head LR={head_lr}) ===")

        if hasattr(model.model, "base_model"):
            for param in model.model.base_model.parameters():
                param.requires_grad = False
        else:
            for n, p in model.model.named_parameters():
                if "fc" not in n and "classifier" not in n:
                    p.requires_grad = False

        optimizer_warmup = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=head_lr, weight_decay=0.01,
        )

        for e in range(warmup_epochs):
            train_metrics = utils.train_one_epoch(
                model, train_loader, loss_fn, optimizer_warmup, device,
                scaler=scaler, mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha, mix_prob=mix_prob,
            )
            test_metrics = utils.test(model, valid_loader, loss_fn, device)

            updated = model_checkpoint.update(test_metrics['f1'])
            logging.info(
                f"[Warm-up {e + 1}/{warmup_epochs}] "
                f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.3f} | "
                f"Val Loss: {test_metrics['loss']:.4f} | Val Macro-F1: {test_metrics['f1']:.4f} "
                f"{'[BEST]' if updated else ''}"
            )

            tensorboard_writer.add_scalar('Loss/Train', train_metrics['loss'], e)
            tensorboard_writer.add_scalar('Loss/Validation', test_metrics['loss'], e)
            tensorboard_writer.add_scalar('MacroF1/Validation', test_metrics['f1'], e)

            if _SIGTERM_RECEIVED:
                _save_last_checkpoint(model, optimizer_warmup, None, e, logdir)
                tensorboard_writer.close()
                return

            global_epoch += 1

        for param in model.parameters():
            param.requires_grad = True

    # ═══════════════════════
    #  PHASE 2 : FINE-TUNING 
    # ════════════════════════
    finetune_epochs = config["nepochs"] - warmup_epochs

    # ── Optimizer ─────────────────────────────
    optimizer = optim.build_optimizer(config, model, has_backbone)

    if has_backbone:
        backbone_lr = float(optim_config.get("backbone_lr", 1e-5))
        finetune_head_lr = float(optim_config.get("finetune_head_lr", 1e-4))
        weight_decay = float(optim_config.get("weight_decay", 0.05))

        logging.info(
            f"\n=== PHASE 2 : FINE-TUNING ({finetune_epochs} epochs, "
            f"backbone LR={backbone_lr}, head LR={finetune_head_lr}, "
            f"weight_decay={weight_decay}) ==="
        )
    else:
        logging.info(f"\n=== TRAINING ({finetune_epochs} epochs) ===")


    # ── Scheduler ─────────────────────────────
    scheduler, plateau_scheduler, finetune_warmup, initial_lrs = optim.build_scheduler(
        config, optimizer, finetune_epochs
    )

    for e in range(finetune_epochs):
        epoch_num = warmup_epochs + e

        if finetune_warmup > 0 and e < finetune_warmup:
            wf = (e + 1) / finetune_warmup
            for i, pg in enumerate(optimizer.param_groups):
                pg['lr'] = initial_lrs[i] * wf

        train_metrics = utils.train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            scaler=scaler, mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha, mix_prob=mix_prob,
        )

        if has_backbone:
            current_lr_bb = optimizer.param_groups[0]["lr"]
            current_lr_head = optimizer.param_groups[1]["lr"]
        else:
            current_lr_bb = optimizer.param_groups[0]["lr"]
            current_lr_head = current_lr_bb

        test_metrics = utils.test(model, valid_loader, loss_fn, device)
        updated = model_checkpoint.update(test_metrics['f1'])

        if e >= finetune_warmup:
            if plateau_scheduler:
                scheduler.step(test_metrics['f1'])
            else:
                scheduler.step()

        if has_backbone:
            logging.info(
                f"[Fine-Tune {epoch_num + 1}/{config['nepochs']}] "
                f"BB-LR: {current_lr_bb:.2e} | Head-LR: {current_lr_head:.2e} | "
                f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.3f} | "
                f"Val Loss: {test_metrics['loss']:.4f} | Val Macro-F1: {test_metrics['f1']:.4f} "
                f"{'[BEST]' if updated else ''}"
            )
        else:
            logging.info(
                f"[{epoch_num + 1}/{config['nepochs']}] "
                f"Train loss={train_metrics['loss']:.4f} F1={train_metrics['f1']:.4f} | "
                f"Val loss={test_metrics['loss']:.4f} F1={test_metrics['f1']:.4f} | "
                f"LR={current_lr_bb:.2e} "
                f"{'★ BEST' if updated else ''}"
            )

        tensorboard_writer.add_scalar('Loss/Train', train_metrics['loss'], global_epoch)
        tensorboard_writer.add_scalar('Loss/Validation', test_metrics['loss'], global_epoch)
        tensorboard_writer.add_scalar('MacroF1/Validation', test_metrics['f1'], global_epoch)
        tensorboard_writer.add_scalar('LR/Backbone', current_lr_bb, global_epoch)
        tensorboard_writer.add_scalar('LR/Head', current_lr_head, global_epoch)

        sched_state = scheduler.state_dict()
        if (epoch_num + 1) % checkpoint_every == 0:
            _save_last_checkpoint(model, optimizer, sched_state, epoch_num, logdir)

        if _SIGTERM_RECEIVED:
            _save_last_checkpoint(model, optimizer, sched_state, epoch_num, logdir)
            tensorboard_writer.close()
            return

        global_epoch += 1

    tensorboard_writer.close()
    logging.info(f"Best val macro-F1: {model_checkpoint.best_score:.4f}")
    logging.info("=== Entraînement terminé ! ===")


def test(config):
    raise NotImplementedError


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage: {sys.argv[0]} path/to/config.yaml <train|test>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))

    command = sys.argv[2]
    eval(f"{command}(config)")
