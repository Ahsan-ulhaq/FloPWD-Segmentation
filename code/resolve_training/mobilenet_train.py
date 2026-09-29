import os
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from mobilenet_seg import MobileNetV2Seg
from mobilenet_dataset import (
    PlasticMaskBank,
    MobileNetDataset,
    MobileNetLoss,
    compute_metrics,
    mixup_batch,
)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay   = decay
        self.shadow  = {}
        self._backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone().float()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n] = (self.decay * self.shadow[n]
                                  + (1 - self.decay) * p.data.float())

    def store(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self._backup[n] = p.data.clone()

    def apply(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n].to(p.dtype))

    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self._backup:
                p.data.copy_(self._backup[n])
        self._backup.clear()


class WarmupCosine:
    def __init__(self, optimizer, warmup_ep: int, total_ep: int,
                 base_lr: float, eta_min: float = 1e-6):
        self.opt       = optimizer
        self.warmup_ep = warmup_ep
        self.total_ep  = total_ep
        self.base_lr   = base_lr
        self.eta_min   = eta_min
        self._epoch    = 0

    def step(self) -> float:
        self._epoch += 1
        e = self._epoch
        if e <= self.warmup_ep:
            lr = self.base_lr * e / self.warmup_ep
        else:
            prog = (e - self.warmup_ep) / (self.total_ep - self.warmup_ep)
            lr   = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
                1 + np.cos(np.pi * prog))
        for pg in self.opt.param_groups:
            scale = pg.get('lr_scale', 1.0)
            pg['lr'] = lr * scale
        return lr


TTA_SCALES = [0.75, 1.0, 1.25]

@torch.no_grad()
def tta_predict(model: nn.Module, img: torch.Tensor) -> torch.Tensor:
    H, W = img.shape[2:]
    acc  = torch.zeros(img.shape[0], 2, H, W, device=img.device)
    n    = 0

    was_training = model.training
    model.eval()

    for scale in TTA_SCALES:
        for flip in (False, True):
            x = torch.flip(img, [-1]) if flip else img
            if scale != 1.0:
                nh, nw = int(H * scale), int(W * scale)
                x = F.interpolate(x, (nh, nw), mode='bilinear', align_corners=False)

            logits = model(x)
            prob   = torch.softmax(logits, dim=1)
            prob   = F.interpolate(prob, (H, W), mode='bilinear', align_corners=False)
            if flip:
                prob = torch.flip(prob, [-1])
            acc += prob
            n   += 1

    if was_training:
        model.train()
    return acc / n


def plot_metrics(metrics: dict, epoch: int, plot_dir: Path):
    pairs = [
        ('train_loss',        'val_loss',        'Loss',           'Loss'),
        ('train_mPA',         'val_mPA',         'mPA',            'Mean Pixel Accuracy'),
        ('train_mIoU',        'val_mIoU',        'mIoU',           'Mean IoU'),
        ('train_f1',          'val_f1',          'F1 (plastic)',   'F1 Score'),
        ('train_plastic_iou', 'val_plastic_iou', 'Plastic IoU',    'Plastic Class IoU'),
        ('train_bg_iou',      'val_bg_iou',      'Background IoU', 'Background Class IoU'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'MobileNetV2Seg — Training Curves (Epoch {epoch})',
                 fontsize=13, fontweight='bold')
    for ax, (tk, vk, title, ylabel) in zip(axes.flat, pairs):
        if tk in metrics and metrics[tk]:
            ax.plot(metrics['epoch'], metrics[tk], label='Train', lw=1.5)
            ax.plot(metrics['epoch'], metrics[vk], label='Val',
                    lw=1.5, linestyle='--')
        ax.set(xlabel='Epoch', ylabel=ylabel, title=title)
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / f"training_plots_epoch_{epoch}.png", dpi=100)
    plt.close()


def save_individual_metrics(metrics: dict, csv_dir: Path):
    for key, vals in metrics.items():
        if key == 'epoch':
            continue
        pd.DataFrame({'epoch': metrics['epoch'], key: vals}).to_csv(
            csv_dir / f"{key}.csv", index=False)


def save_predictions(model: nn.Module, loader, save_dir: Path, device):
    model.eval()
    with torch.no_grad():
        for img, _, names in loader:
            img   = img.to(device)
            probs = tta_predict(model, img)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            for j, name in enumerate(names):
                out = np.zeros((*preds[j].shape, 3), dtype=np.uint8)
                out[preds[j] == 1] = [255, 0, 0]
                Image.fromarray(out).save(save_dir / f"{name}_pred.png")


def main():
    dal_lake_dir = "/home/2023bcse070/dallake"
    output_dir   = Path("/home/2023bcse070/dallake/output")

    batch_size    = 8
    epochs        = 2000
    base_lr       = 3e-4
    warmup_epochs = 15
    eta_min       = 1e-6

    PLASTIC_WEIGHT = 13.3
    MIXUP_ALPHA    = 0.4
    MIXUP_PROB     = 0.5
    CP_PROB        = 0.4
    EMA_DECAY      = 0.9995
    GRAD_CLIP      = 1.0

    BACKBONE_LR_SCALE = 0.1

    ckpt_dir  = output_dir / "checkpoints_mobilenet"
    logs_dir  = ckpt_dir   / "logs"
    plots_dir = ckpt_dir   / "plots"
    pred_dir  = output_dir / "predictions_mobilenet"

    for d in [ckpt_dir, logs_dir, plots_dir, pred_dir]:
        d.mkdir(parents=True, exist_ok=True)

    best_ckpt    = ckpt_dir / "best_model_mobilenet.pth"
    combined_csv = logs_dir / "training_metrics_mobilenet.csv"

    print("=" * 65)
    print("MobileNetV2Seg Training  (Fair Comparison vs ENet / BiSeNetV2)")
    print("  FocalTversky + WeightedCE  |  MixUp  |  EMA  |  TTA")
    print("  Per-class IoU tracked: background + plastic")
    print("=" * 65)

    train_dir = os.path.join(dal_lake_dir, 'train')
    mask_bank = PlasticMaskBank(train_dir, train_dir)

    print("\nLoading datasets…")
    train_ds = MobileNetDataset(dal_lake_dir, 'train',
                                mask_bank=mask_bank, cp_prob=CP_PROB)
    val_ds   = MobileNetDataset(dal_lake_dir, 'val', mask_bank=None)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MobileNetV2Seg(num_classes=2, pretrained=False).to(device)
    print(f"\nDevice: {device}")

    total_p = sum(p.numel() for p in model.parameters())
    enc_p   = sum(p.numel() for p in list(model.low_encoder.parameters())
                              + list(model.high_encoder.parameters()))
    dec_p   = total_p - enc_p
    print(f"MobileNetV2Seg  total: {total_p:,} ({total_p/1e6:.2f}M)")
    print(f"  Backbone: {enc_p:,}  |  Decoder: {dec_p:,}")

    backbone_params = (list(model.low_encoder.parameters())
                       + list(model.high_encoder.parameters()))
    decoder_params  = (list(model.aspp.parameters())
                       + list(model.low_proj.parameters())
                       + list(model.fuse.parameters())
                       + list(model.classifier.parameters()))

    optimizer = torch.optim.AdamW([
        {'params': backbone_params,
         'lr': base_lr * BACKBONE_LR_SCALE,
         'lr_scale': BACKBONE_LR_SCALE},
        {'params': decoder_params,
         'lr': base_lr,
         'lr_scale': 1.0},
    ], weight_decay=1e-4, betas=(0.9, 0.999))

    criterion = MobileNetLoss(plastic_weight=PLASTIC_WEIGHT)
    scheduler = WarmupCosine(optimizer, warmup_epochs, epochs, base_lr, eta_min)
    ema       = EMA(model, decay=EMA_DECAY)

    print(f"\n[LR]  WarmupCosine: {base_lr} → {eta_min}  "
          f"(warmup {warmup_epochs} ep)")
    print(f"      Backbone LR scale: {BACKBONE_LR_SCALE}×")
    print(f"[EMA] decay={EMA_DECAY}")
    print(f"[MixUp] α={MIXUP_ALPHA}, prob={MIXUP_PROB}")
    print(f"[CopyPaste] prob={CP_PROB}")

    metric_keys = [
        'epoch',
        'train_loss',        'val_loss',
        'train_mIoU',        'val_mIoU',
        'train_mPA',         'val_mPA',
        'train_bg_iou',      'val_bg_iou',
        'train_plastic_iou', 'val_plastic_iou',
        'train_bg_pa',       'val_bg_pa',
        'train_plastic_pa',  'val_plastic_pa',
        'train_f1',          'val_f1',
        'train_prec',        'val_prec',
        'train_rec',         'val_rec',
        'train_fps',         'val_fps',
    ]
    metrics       = {k: [] for k in metric_keys}
    best_val_miou = 0.0

    print("\nStarting training…\n")

    for epoch in range(epochs):

        model.train()

        t = {k: 0.0 for k in ['loss', 'mIoU', 'mPA',
                               'bg_iou', 'plastic_iou',
                               'bg_pa', 'plastic_pa',
                               'f1', 'prec', 'rec']}
        t0 = time.time()

        for img, lbl, _ in train_loader:
            img, lbl = img.to(device), lbl.to(device)

            if img.shape[0] > 1 and random.random() < MIXUP_PROB:
                img, lbl_f = mixup_batch(img, lbl.float(), alpha=MIXUP_ALPHA)
            else:
                lbl_f = lbl.float()

            optimizer.zero_grad()
            outputs = model(img)
            loss    = criterion(outputs, lbl_f)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            ema.update(model)

            with torch.no_grad():
                m = compute_metrics(outputs, lbl)

            t['loss']        += loss.item()
            t['mIoU']        += m['mIoU']
            t['mPA']         += m['mPA']
            t['bg_iou']      += m['bg_iou']
            t['plastic_iou'] += m['plastic_iou']
            t['bg_pa']       += m['bg_pa']
            t['plastic_pa']  += m['plastic_pa']
            t['f1']          += m['f1']
            t['prec']        += m['precision']
            t['rec']         += m['recall']

        n_tr      = len(train_loader)
        train_fps = n_tr * batch_size / max(time.time() - t0, 1e-6)
        current_lr = scheduler.step()

        ema.store(model)
        ema.apply(model)
        model.eval()

        v = {k: 0.0 for k in t.keys()}
        t0 = time.time()

        with torch.no_grad():
            for img, lbl, _ in val_loader:
                img, lbl = img.to(device), lbl.to(device)

                probs  = tta_predict(model, img)
                logits = model(img)
                loss   = criterion(logits, lbl)

                m = compute_metrics(probs, lbl)

                v['loss']        += loss.item()
                v['mIoU']        += m['mIoU']
                v['mPA']         += m['mPA']
                v['bg_iou']      += m['bg_iou']
                v['plastic_iou'] += m['plastic_iou']
                v['bg_pa']       += m['bg_pa']
                v['plastic_pa']  += m['plastic_pa']
                v['f1']          += m['f1']
                v['prec']        += m['precision']
                v['rec']         += m['recall']

        ema.restore(model)

        n_vl    = len(val_loader)
        val_fps = n_vl * batch_size / max(time.time() - t0, 1e-6)

        def a(s, n): return s / n

        avg = {
            'train_loss':        a(t['loss'],         n_tr),
            'val_loss':          a(v['loss'],          n_vl),
            'train_mIoU':        a(t['mIoU'],          n_tr),
            'val_mIoU':          a(v['mIoU'],           n_vl),
            'train_mPA':         a(t['mPA'],            n_tr),
            'val_mPA':           a(v['mPA'],             n_vl),
            'train_bg_iou':      a(t['bg_iou'],         n_tr),
            'val_bg_iou':        a(v['bg_iou'],          n_vl),
            'train_plastic_iou': a(t['plastic_iou'],    n_tr),
            'val_plastic_iou':   a(v['plastic_iou'],     n_vl),
            'train_bg_pa':       a(t['bg_pa'],           n_tr),
            'val_bg_pa':         a(v['bg_pa'],            n_vl),
            'train_plastic_pa':  a(t['plastic_pa'],      n_tr),
            'val_plastic_pa':    a(v['plastic_pa'],       n_vl),
            'train_f1':          a(t['f1'],               n_tr),
            'val_f1':            a(v['f1'],                n_vl),
            'train_prec':        a(t['prec'],              n_tr),
            'val_prec':          a(v['prec'],               n_vl),
            'train_rec':         a(t['rec'],                n_tr),
            'val_rec':           a(v['rec'],                 n_vl),
            'train_fps':         train_fps,
            'val_fps':           val_fps,
        }

        print(f"Epoch [{epoch+1:03d}/{epochs}]  lr={current_lr:.2e}")
        print(f"  Train → Loss:{avg['train_loss']:.4f}"
              f"  mIoU:{avg['train_mIoU']:.4f}"
              f"  [BG:{avg['train_bg_iou']:.4f} | Plastic:{avg['train_plastic_iou']:.4f}]"
              f"  mPA:{avg['train_mPA']:.4f}"
              f"  F1:{avg['train_f1']:.4f}"
              f"  Prec:{avg['train_prec']:.4f}"
              f"  Rec:{avg['train_rec']:.4f}"
              f"  FPS:{train_fps:.1f}")
        print(f"  Val   → Loss:{avg['val_loss']:.4f}"
              f"  mIoU:{avg['val_mIoU']:.4f}"
              f"  [BG:{avg['val_bg_iou']:.4f} | Plastic:{avg['val_plastic_iou']:.4f}]"
              f"  mPA:{avg['val_mPA']:.4f}"
              f"  F1:{avg['val_f1']:.4f}"
              f"  Prec:{avg['val_prec']:.4f}"
              f"  Rec:{avg['val_rec']:.4f}"
              f"  FPS:{val_fps:.1f}")
        print(f"  Gap   → mIoU:{avg['train_mIoU']-avg['val_mIoU']:.4f}"
              f"  PlasticIoU:{avg['train_plastic_iou']-avg['val_plastic_iou']:.4f}")

        metrics['epoch'].append(epoch + 1)
        for k in metric_keys:
            if k != 'epoch':
                metrics[k].append(avg[k])

        if avg['val_mIoU'] > best_val_miou:
            best_val_miou = avg['val_mIoU']
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✅ Best model saved  "
                  f"(val mIoU {best_val_miou:.4f}  "
                  f"| plastic IoU {avg['val_plastic_iou']:.4f})")

        plot_metrics(metrics, epoch + 1, plots_dir)
        save_individual_metrics(metrics, logs_dir)

    pd.DataFrame(metrics).to_csv(combined_csv, index=False)
    print(f"\n📊 Metrics CSV → {combined_csv}")

    print("\nGenerating final val predictions (TTA + EMA)…")
    missing, unexpected = model.load_state_dict(
        torch.load(best_ckpt, map_location=device), strict=False)
    real_issues = [k for k in missing + unexpected
                   if 'total_ops' not in k and 'total_params' not in k]
    if real_issues:
        print(f"  [WARNING] {real_issues[:3]}")
    ema.store(model)
    ema.apply(model)
    save_predictions(model, val_loader, pred_dir, device)
    print(f"✅ Predictions → {pred_dir}")


if __name__ == "__main__":
    main()
