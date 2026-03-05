# coding: utf-8

import os

import numpy as np
import torch
import torch.nn
import tqdm
from sklearn.metrics import accuracy_score, f1_score


def generate_unique_logpath(logdir, raw_run_name):
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


class ModelCheckpoint(object):

    def __init__(self, model, savepath, min_is_best=True):
        self.model = model
        self.savepath = savepath
        self.best_score = None
        if min_is_best:
            self.is_better = self.lower_is_better
        else:
            self.is_better = self.higher_is_better

    def lower_is_better(self, score):
        return self.best_score is None or score < self.best_score

    def higher_is_better(self, score):
        return self.best_score is None or score > self.best_score

    def update(self, score):
        if self.is_better(score):
            torch.save(self.model.state_dict(), self.savepath)
            self.best_score = score
            return True
        return False


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)

    _, _, H, W = x.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)

    cy, cx = np.random.randint(H), np.random.randint(W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    return mixed_x, y, y[index], lam


def train_one_epoch(model, loader, f_loss, optimizer, device,
                    scaler=None, mixup_alpha=0.0, cutmix_alpha=0.0, mix_prob=0.5):
    model.train()

    total_loss = 0
    num_samples = 0
    all_preds = []
    all_targets = []
    use_mix = (mixup_alpha > 0 or cutmix_alpha > 0)

    non_blocking = device.type == "cuda"
    for inputs, targets in (pbar := tqdm.tqdm(loader, desc="Train")):
        inputs = inputs.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)

        targets_a, targets_b, lam = targets, targets, 1.0
        if use_mix and np.random.rand() < mix_prob:
            use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or np.random.rand() < 0.5)
            if use_cutmix:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, cutmix_alpha)
            else:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            outputs = model(inputs)
            loss = lam * f_loss(outputs, targets_a) + (1 - lam) * f_loss(outputs, targets_b)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += inputs.shape[0] * loss.item()
        num_samples += inputs.shape[0]

        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets_a.cpu().numpy())

        pbar.set_description(f"Train loss : {total_loss / num_samples:.4f}")

    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    return {
        'loss': total_loss / num_samples,
        'accuracy': accuracy,
        'f1': f1,
    }


def test(model, loader, f_loss, device):
    model.eval()

    total_loss = 0
    num_samples = 0
    all_preds = []
    all_targets = []

    non_blocking = device.type == "cuda"
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            outputs = model(inputs)
            loss = f_loss(outputs, targets)

            total_loss += inputs.shape[0] * loss.item()
            num_samples += inputs.shape[0]

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    return {
        'loss': total_loss / num_samples,
        'accuracy': accuracy,
        'f1': f1,
    }
