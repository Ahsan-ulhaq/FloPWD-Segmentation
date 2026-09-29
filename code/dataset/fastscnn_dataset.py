import os
import random
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A


class PlasticMaskBank:

    SMALL_THRESH = 500

    def __init__(self, train_img_dir: str, train_lbl_dir: str):
        self.small = []
        self.large = []
        img_dir = Path(train_img_dir)
        lbl_dir = Path(train_lbl_dir)
        print("[MaskBank] Building plastic crop bank…")

        for lf in sorted(lbl_dir.glob("*_trainLabelId.png")):
            base = lf.stem.replace("_trainLabelId", "")
            ip = img_dir / f"{base}.jpg"
            if not ip.exists():
                ip = img_dir / f"{base}.png"
            if not ip.exists():
                continue

            lbl = np.array(Image.open(lf).convert('L'))
            img = np.array(Image.open(ip).convert('RGB'))
            mask = (lbl == 1)
            if not mask.any():
                continue

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            r0, r1 = np.where(rows)[0][[0, -1]]
            c0, c1 = np.where(cols)[0][[0, -1]]

            rgb_c  = img[r0:r1+1, c0:c1+1]
            mask_c = mask[r0:r1+1, c0:c1+1]
            area   = mask_c.sum()

            bucket = self.small if area < self.SMALL_THRESH else self.large
            bucket.append((rgb_c, mask_c))

        total = len(self.small) + len(self.large)
        print(f"[MaskBank] {total} regions  "
              f"({len(self.small)} small, {len(self.large)} large)")

    def sample(self):
        pool = self.small * 3 + self.large
        return random.choice(pool)

    def __len__(self):
        return len(self.small) + len(self.large)


def copy_paste(img: np.ndarray, lbl: np.ndarray,
               bank: PlasticMaskBank, n: int = 2,
               scale_range=(0.5, 2.0)):
    H, W = img.shape[:2]
    img, lbl = img.copy(), lbl.copy()

    for _ in range(n):
        rgb_c, mask_c = bank.sample()
        ch, cw = rgb_c.shape[:2]
        sc = random.uniform(*scale_range)
        nh, nw = max(1, int(ch*sc)), max(1, int(cw*sc))

        rgb_r  = np.array(Image.fromarray(rgb_c).resize((nw, nh), Image.BILINEAR))
        mask_r = np.array(Image.fromarray(mask_c.astype(np.uint8)*255)
                          .resize((nw, nh), Image.NEAREST)) > 127

        nh, nw = min(nh, H), min(nw, W)
        rgb_r, mask_r = rgb_r[:nh, :nw], mask_r[:nh, :nw]

        r0 = random.randint(0, H - nh)
        c0 = random.randint(0, W - nw)

        roi_img = img[r0:r0+nh, c0:c0+nw]
        roi_lbl = lbl[r0:r0+nh, c0:c0+nw]
        roi_img[mask_r] = rgb_r[mask_r]
        roi_lbl[mask_r] = 1
        img[r0:r0+nh, c0:c0+nw] = roi_img
        lbl[r0:r0+nh, c0:c0+nw] = roi_lbl

    return img, lbl


class FastSCNNDataset(Dataset):

    def __init__(self, root: str, split: str = 'train',
                 mask_bank: PlasticMaskBank = None,
                 n_paste: int = 2, cp_prob: float = 0.4):
        self.split     = split
        self.mask_bank = mask_bank
        self.n_paste   = n_paste
        self.cp_prob   = cp_prob
        self.images, self.labels, self.names = [], [], []

        split_dir = os.path.join(root, split)
        lbl_dir   = os.path.join(root, 'testlabel') if split == 'test' else split_dir

        for fn in sorted(os.listdir(split_dir)):
            if not fn.endswith(('.jpg', '.png')):
                continue
            base = os.path.splitext(fn)[0]
            lp   = os.path.join(lbl_dir, f"{base}_trainLabelId.png")
            if os.path.exists(lp):
                self.images.append(os.path.join(split_dir, fn))
                self.labels.append(lp)
                self.names.append(fn)

        if not self.images:
            raise ValueError(f"No image-label pairs found in {split_dir}")
        print(f"[{split}] {len(self.images)} image-label pairs")

        if split == 'train':
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.4),
                A.ColorJitter(brightness=0.15, contrast=0.15,
                              saturation=0.15, hue=0.1, p=0.4),
                A.GaussianBlur(blur_limit=(3, 5), p=0.15),
                A.ToGray(p=0.05),
                A.CoarseDropout(
                    num_holes_range=(1, 4),
                    hole_height_range=(10, 40),
                    hole_width_range=(10, 40),
                    fill=0,
                    p=0.15
                ),
            ], additional_targets={'label': 'mask'})
        else:
            self.aug = A.Compose([], additional_targets={'label': 'mask'})

        self.to_tensor = transforms.ToTensor()
        self.norm      = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.images[idx]).convert('RGB'))
        lbl = np.array(Image.open(self.labels[idx]).convert('L'))
        name = self.names[idx]

        if (self.split == 'train'
                and self.mask_bank is not None
                and len(self.mask_bank) > 0
                and random.random() < self.cp_prob):
            img, lbl = copy_paste(img, lbl, self.mask_bank, self.n_paste)

        out = self.aug(image=img, label=lbl)
        img, lbl = out['image'], out['label']

        img_t  = self.norm(self.to_tensor(img))               # (3,H,W) float32
        lbl_t  = torch.from_numpy((lbl == 1).astype(np.float32)).unsqueeze(0)
        return img_t, lbl_t, name


def mixup_batch(imgs: torch.Tensor, masks: torch.Tensor, alpha: float = 0.4):
    lam = max(np.random.beta(alpha, alpha), 1 - np.random.beta(alpha, alpha))
    idx = torch.randperm(imgs.shape[0], device=imgs.device)
    mixed_imgs  = lam * imgs  + (1 - lam) * imgs[idx]
    mixed_masks = lam * masks + (1 - lam) * masks[idx]
    return mixed_imgs, mixed_masks


class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, pos_weight=10.0, clip=0.05):
        super().__init__()
        self.gp = gamma_pos
        self.gn = gamma_neg
        self.pw = pos_weight
        self.clip = clip

    def forward(self, logit, target):
        prob = torch.sigmoid(logit)
        prob_neg = (prob + self.clip).clamp(max=1)

        pos = -target * torch.log(prob.clamp(min=1e-8))
        if self.gp > 0:
            pos = pos * (1 - prob) ** self.gp

        neg = -(1 - target) * torch.log((1 - prob_neg).clamp(min=1e-8))
        if self.gn > 0:
            neg = neg * prob_neg ** self.gn

        return (self.pw * pos + neg).mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.smooth = smooth

    def forward(self, logit, target):
        p = torch.sigmoid(logit).view(-1)
        t = target.view(-1)
        tp = (p * t).sum()
        fp = ((1-t) * p).sum()
        fn = (t * (1-p)).sum()
        tv = (tp + self.smooth) / (tp + self.alpha*fn + self.beta*fp + self.smooth)
        return 1.0 - tv


class BoundaryLoss(nn.Module):
    def forward(self, bpred, target):
        k = 3
        dilated = F.max_pool2d(target, k, 1, k//2)
        eroded  = -F.max_pool2d(-target, k, 1, k//2)
        bgt     = (dilated - eroded).squeeze(1)
        return F.binary_cross_entropy_with_logits(bpred.squeeze(1), bgt)


class FastSCNNLoss(nn.Module):
    def __init__(self, pos_weight: float = 10.0):
        super().__init__()
        self.afl = AsymmetricFocalLoss(pos_weight=pos_weight)
        self.tvl = TverskyLoss()

    def _seg_loss(self, logit, target):
        return 0.5 * self.afl(logit, target) + 0.5 * self.tvl(logit, target)

    def forward(self, outputs, target):
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        return self._seg_loss(outputs, target)


def compute_metrics_binary(logit_or_prob, target_bin, threshold=0.5):
    if logit_or_prob.dim() == 4:
        logit_or_prob = logit_or_prob.squeeze(1)
    if target_bin.dim() == 4:
        target_bin = target_bin.squeeze(1)

    if logit_or_prob.min() < 0 or logit_or_prob.max() > 1:
        prob = torch.sigmoid(logit_or_prob)
    else:
        prob = logit_or_prob

    pred = (prob > threshold).float()
    tgt  = target_bin.float()

    flat_p = pred.view(-1)
    flat_t = tgt.view(-1)

    tp = (flat_p * flat_t).sum()
    fp = (flat_p * (1 - flat_t)).sum()
    fn = ((1 - flat_p) * flat_t).sum()
    tn = ((1 - flat_p) * (1 - flat_t)).sum()

    plastic_iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    bg_iou      = (tn + 1e-6) / (tn + fp + fn + 1e-6)
    mIoU        = (plastic_iou + bg_iou) / 2.0

    prec = (tp + 1e-6) / (tp + fp + 1e-6)
    rec  = (tp + 1e-6) / (tp + fn + 1e-6)
    f1   = 2 * prec * rec / (prec + rec + 1e-6)
    mPA  = ((tp / (tp + fn + 1e-6) + tn / (tn + fp + 1e-6)) / 2.0)

    return {
        'mIoU'        : mIoU.item(),
        'plastic_iou' : plastic_iou.item(),
        'bg_iou'      : bg_iou.item(),
        'mPA'         : mPA.item(),
        'f1'          : f1.item(),
        'precision'   : prec.item(),
        'recall'      : rec.item(),
    }


if __name__ == "__main__":
    import numpy as np

    logit  = torch.randn(2, 1, 64, 64)
    target = (torch.rand(2, 1, 64, 64) > 0.9).float()
    loss_fn = FastSCNNLoss(pos_weight=10.0)

    loss = loss_fn(logit, target)
    print(f"Loss (single tensor): {loss.item():.4f}")

    m = compute_metrics_binary(logit, target)
    print(f"Metrics: mIoU={m['mIoU']:.4f}  F1={m['f1']:.4f}  "
          f"Prec={m['precision']:.4f}  Rec={m['recall']:.4f}")

    imgs  = torch.randn(4, 3, 64, 64)
    masks = (torch.rand(4, 1, 64, 64) > 0.9).float()
    mi, mm = mixup_batch(imgs, masks)
    print(f"MixUp: img {tuple(mi.shape)}, mask range [{mm.min():.2f}, {mm.max():.2f}]")

    print("\nAll fastscnn_dataset.py checks passed.")
