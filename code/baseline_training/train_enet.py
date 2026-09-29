import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import time

from dal_base import DalLake
from enet import ENet


def count_flops_params(model, input_tensor):
    flops_counter = [0]

    def conv_hook(module, inp, out):
        in_c  = module.in_channels
        out_c = module.out_channels
        kH, kW = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        groups = module.groups
        _, _, oH, oW = out.shape
        flops_counter[0] += (in_c // groups) * kH * kW * out_c * oH * oW * 2

    def convt_hook(module, inp, out):
        in_c  = module.in_channels
        out_c = module.out_channels
        kH, kW = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        groups = module.groups
        _, _, oH, oW = out.shape
        flops_counter[0] += (in_c // groups) * kH * kW * out_c * oH * oW * 2

    def bn_hook(module, inp, out):
        flops_counter[0] += out.numel() * 2

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.ConvTranspose2d):
            hooks.append(m.register_forward_hook(convt_hook))
        elif isinstance(m, nn.BatchNorm2d):
            hooks.append(m.register_forward_hook(bn_hook))

    with torch.no_grad():
        model(input_tensor)

    for h in hooks:
        h.remove()

    params = sum(p.numel() for p in model.parameters())
    return flops_counter[0], params


def compute_metrics(preds, target, num_classes=2):
    preds = torch.argmax(preds, dim=1)

    pixel_acc    = torch.zeros(num_classes, device=preds.device)
    intersection = torch.zeros(num_classes, device=preds.device)
    union        = torch.zeros(num_classes, device=preds.device)

    for cls in range(num_classes):
        pred_cls   = (preds == cls)
        target_cls = (target == cls)
        total      = target_cls.sum().float()

        pixel_acc[cls]    = (pred_cls & target_cls).sum().float() / total if total > 0 else 0.0
        intersection[cls] = (pred_cls & target_cls).sum().float()
        union[cls]        = (pred_cls | target_cls).sum().float()

    mPA  = pixel_acc.mean().item()
    mIoU = (intersection / union.clamp(min=1e-6)).mean().item()

    tp = intersection[1].item()
    fp = ((preds == 1) & (target == 0)).sum().float().item()
    fn = ((preds == 0) & (target == 1)).sum().float().item()

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)

    return mPA, mIoU, f1, precision, recall


def save_predicted_images(model, loader, save_dir, device):
    model.eval()
    with torch.no_grad():
        for img, target, names in loader:
            img   = img.to(device)
            preds = torch.argmax(model(img), dim=1)
            for j in range(img.shape[0]):
                pred     = preds[j].cpu().numpy()
                pred_img = np.zeros((*pred.shape, 3), dtype=np.uint8)
                pred_img[pred == 1] = [255, 0, 0]
                Image.fromarray(pred_img).save(save_dir / f"{names[j]}_pred.png")


def plot_metrics(metrics, epoch, plot_dir, fold):
    pairs = [
        ('train_loss', 'val_loss', 'Loss',               'Loss'),
        ('train_mPA',  'val_mPA',  'mPA',                'Mean Pixel Accuracy'),
        ('train_mIoU', 'val_mIoU', 'mIoU',               'Mean IoU'),
        ('train_f1',   'val_f1',   'F1 (plastic)',        'F1 Score'),
        ('train_prec', 'val_prec', 'Precision (plastic)', 'Precision'),
        ('train_rec',  'val_rec',  'Recall (plastic)',    'Recall'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Fold {fold} - Training Curves (Epoch {epoch})', fontsize=14, fontweight='bold')

    for ax, (train_key, val_key, title, ylabel) in zip(axes.flat, pairs):
        if train_key in metrics and val_key in metrics:
            ax.plot(metrics['epoch'], metrics[train_key], label='Train', linewidth=1.5)
            ax.plot(metrics['epoch'], metrics[val_key],   label='Val',   linewidth=1.5, linestyle='--')
        ax.set(xlabel='Epoch', ylabel=ylabel, title=title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir / f"fold{fold}_training_plots_epoch_{epoch}.png", dpi=100)
    plt.close()


def save_individual_metrics(metrics, csv_dir, fold):
    for key, values in metrics.items():
        if key == 'epoch':
            continue
        pd.DataFrame({'epoch': metrics['epoch'], key: values}).to_csv(
            csv_dir / f"fold{fold}_{key}.csv", index=False
        )


def main():
    dal_lake_dir   = "/home/2023bcse070/dallake"
    batch_size     = 8
    epochs         = 2000
    lr             = 0.001
    crop_h, crop_w = 720, 1280
    fold           = 1

    output_dir      = Path("/home/2023bcse070/dallake/outputs")
    checkpoint_dir  = output_dir      / "checkpoints_enet"
    logs_dir        = checkpoint_dir  / "logs"
    plots_dir       = checkpoint_dir  / "plots"
    pred_save_dir   = output_dir      / "predictions_enet"

    for d in [checkpoint_dir, logs_dir, plots_dir, pred_save_dir]:
        d.mkdir(parents=True, exist_ok=True)

    model_save_path = checkpoint_dir / "best_model.pth"
    combined_csv    = logs_dir       / "training_metrics.csv"

    print(f"Data : {dal_lake_dir}")
    print(f"Size : {crop_h}x{crop_w}  |  Batch: {batch_size}  |  Epochs: {epochs}")
    print(f"Dirs :\n  checkpoints -> {checkpoint_dir}"
          f"\n  logs        -> {logs_dir}"
          f"\n  plots       -> {plots_dir}")

    print("\nLoading datasets...")
    train_data = DalLake(root=dal_lake_dir, split='train')
    val_data   = DalLake(root=dal_lake_dir, split='val')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Train: {len(train_data)} samples  |  Val: {len(val_data)} samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ENet(num_classes=2).to(device)
    print(f"\nModel on: {device}")

    dummy = torch.randn(1, 3, crop_h, crop_w).to(device)
    flops, params = count_flops_params(model, dummy)
    print(f"FLOPs : {flops/1e9:.2f} G  |  Params: {params/1e6:.2f} M")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    metric_keys = [
        'epoch',
        'train_loss', 'val_loss',
        'train_mPA',  'val_mPA',
        'train_mIoU', 'val_mIoU',
        'train_f1',   'val_f1',
        'train_prec', 'val_prec',
        'train_rec',  'val_rec',
        'train_fps',  'val_fps',
    ]
    metrics       = {k: [] for k in metric_keys}
    best_val_loss = float('inf')

    print("\nStarting training...\n")

    for epoch in range(epochs):

        model.train()
        t_loss = t_mPA = t_mIoU = t_f1 = t_prec = t_rec = 0.0
        t0 = time.time()

        for img, target, _ in train_loader:
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss   = criterion(output, target)
            loss.backward()
            optimizer.step()

            mPA, mIoU, f1, prec, rec = compute_metrics(output, target)
            t_loss += loss.item()
            t_mPA  += mPA
            t_mIoU += mIoU
            t_f1   += f1
            t_prec += prec
            t_rec  += rec

        n_train   = len(train_loader)
        train_fps = n_train * batch_size / max(time.time() - t0, 1e-6)

        model.eval()
        v_loss = v_mPA = v_mIoU = v_f1 = v_prec = v_rec = 0.0
        t0 = time.time()

        with torch.no_grad():
            for img, target, _ in val_loader:
                img, target = img.to(device), target.to(device)
                output = model(img)
                loss   = criterion(output, target)

                mPA, mIoU, f1, prec, rec = compute_metrics(output, target)
                v_loss += loss.item()
                v_mPA  += mPA
                v_mIoU += mIoU
                v_f1   += f1
                v_prec += prec
                v_rec  += rec

        n_val   = len(val_loader)
        val_fps = n_val * batch_size / max(time.time() - t0, 1e-6)

        def avg(s, n): return s / n

        avg_tl, avg_tmPA, avg_tmIoU = avg(t_loss, n_train), avg(t_mPA, n_train), avg(t_mIoU, n_train)
        avg_tf1, avg_tpr, avg_tre   = avg(t_f1, n_train),   avg(t_prec, n_train), avg(t_rec, n_train)

        avg_vl, avg_vmPA, avg_vmIoU = avg(v_loss, n_val), avg(v_mPA, n_val), avg(v_mIoU, n_val)
        avg_vf1, avg_vpr, avg_vre   = avg(v_f1, n_val),   avg(v_prec, n_val), avg(v_rec, n_val)

        print(f"Epoch [{epoch+1:03d}/{epochs}]")
        print(f"  Train -> Loss: {avg_tl:.4f}  mPA: {avg_tmPA:.4f}  mIoU: {avg_tmIoU:.4f}"
              f"  F1: {avg_tf1:.4f}  Prec: {avg_tpr:.4f}  Rec: {avg_tre:.4f}  FPS: {train_fps:.1f}")
        print(f"  Val   -> Loss: {avg_vl:.4f}  mPA: {avg_vmPA:.4f}  mIoU: {avg_vmIoU:.4f}"
              f"  F1: {avg_vf1:.4f}  Prec: {avg_vpr:.4f}  Rec: {avg_vre:.4f}  FPS: {val_fps:.1f}")

        for key, val in zip(metric_keys, [
            epoch + 1,
            avg_tl, avg_vl,
            avg_tmPA,  avg_vmPA,
            avg_tmIoU, avg_vmIoU,
            avg_tf1,   avg_vf1,
            avg_tpr,   avg_vpr,
            avg_tre,   avg_vre,
            train_fps, val_fps,
        ]):
            metrics[key].append(val)

        if avg_vl < best_val_loss:
            best_val_loss = avg_vl
            torch.save(model.state_dict(), model_save_path)
            print(f"  [BEST] Model saved -> {model_save_path}  (val loss {best_val_loss:.4f})")

        plot_metrics(metrics, epoch + 1, plots_dir, fold)
        save_individual_metrics(metrics, logs_dir, fold)

    pd.DataFrame(metrics).to_csv(combined_csv, index=False)
    print(f"\n[DONE] Combined metrics -> {combined_csv}")

    print("\nGenerating validation predictions from best model...")
    model.load_state_dict(torch.load(model_save_path))
    save_predicted_images(model, val_loader, pred_save_dir, device)
    print(f"[DONE] Predictions saved -> {pred_save_dir}")


if __name__ == "__main__":
    main()
