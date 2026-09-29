import os
import cv2
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
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

            (self.small if area < self.SMALL_THRESH else self.large).append(
                (rgb_c, mask_c))

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
        nh, nw = max(1, int(ch * sc)), max(1, int(cw * sc))

        rgb_r  = np.array(Image.fromarray(rgb_c).resize((nw, nh), Image.BILINEAR))
        mask_r = np.array(
            Image.fromarray(mask_c.astype(np.uint8) * 255)
                 .resize((nw, nh), Image.NEAREST)) > 127

        nh, nw = min(nh, H), min(nw, W)
        rgb_r, mask_r = rgb_r[:nh, :nw], mask_r[:nh, :nw]

        r0 = random.randint(0, H - nh)
        c0 = random.randint(0, W - nw)

        roi_i = img[r0:r0+nh, c0:c0+nw]
        roi_l = lbl[r0:r0+nh, c0:c0+nw]
        roi_i[mask_r] = rgb_r[mask_r]
        roi_l[mask_r] = 1
        img[r0:r0+nh, c0:c0+nw] = roi_i
        lbl[r0:r0+nh, c0:c0+nw] = roi_l

    return img, lbl



class BiSeNetDataset(Dataset):
 

    def __init__(self, root: str, split: str = 'train',
                 mask_bank: PlasticMaskBank = None,
                 n_paste: int = 2, cp_prob: float = 0.4):
        self.split     = split
        self.mask_bank = mask_bank
        self.n_paste   = n_paste
        self.cp_prob   = cp_prob
        self.images, self.labels, self.names = [], [], []

        split_dir = os.path.join(root, split)
        lbl_dir   = (os.path.join(root, 'testlabel')
                     if split == 'test' else split_dir)

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
            raise ValueError(f"No image-label pairs in {split_dir}")
        print(f"[{split}] {len(self.images)} image-label pairs")

       
        if split == 'train':
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.4),
                A.ColorJitter(brightness=0.15, contrast=0.15,
                              saturation=0.15, hue=0.1, p=0.4),
                A.GaussianBlur(blur_limit=(3, 5), p=0.15),
                A.ToGray(p=0.05),
                A.CoarseDropout(max_holes=4, max_height=40, max_width=40,
                                fill_value=0, p=0.15),
            ], additional_targets={'label': 'mask'})
        else:
            self.aug = A.Compose([], additional_targets={'label': 'mask'})

        self.to_tensor = transforms.ToTensor()
        self.norm      = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225])

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

        img_t = self.norm(self.to_tensor(img))                # (3,H,W) float32
        lbl_t = torch.from_numpy((lbl == 1).astype(np.int64)) # (H,W)   int64
        return img_t, lbl_t, name



def mixup_batch(imgs: torch.Tensor, lbls: torch.Tensor, alpha: float = 0.4):
    
    import numpy as np
    lam = max(np.random.beta(alpha, alpha), 1 - np.random.beta(alpha, alpha))
    idx = torch.randperm(imgs.shape[0], device=imgs.device)
    return (lam * imgs + (1 - lam) * imgs[idx],
            lam * lbls + (1 - lam) * lbls[idx])



class FocalTverskyLoss(torch.nn.Module):
   
    def __init__(self, alpha=0.3, beta=0.7, gamma=4/3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, target):
        import torch.nn.functional as F
        if logits.shape[2:] != target.shape[1:]:
            target = F.interpolate(
                target.unsqueeze(1).float(),
                size=logits.shape[2:], mode='nearest').squeeze(1).long()

        probs = torch.softmax(logits, dim=1)
        B, C, H, W = probs.shape
        probs     = probs.view(B, C, -1)
        tgt_oh    = F.one_hot(target.view(B, -1), C).permute(0, 2, 1).float()

        TP = (probs * tgt_oh).sum(-1)
        FP = (probs * (1 - tgt_oh)).sum(-1)
        FN = ((1 - probs) * tgt_oh).sum(-1)

        tv = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth)
        return ((1 - tv) ** self.gamma).mean()


class BiSeNetLoss(torch.nn.Module):
  
    def __init__(self, plastic_weight: float = 13.3):
        super().__init__()
        self.ftl = FocalTverskyLoss()
        self.pw  = plastic_weight

    def _ce(self, logits, target):
        import torch.nn.functional as F
        if logits.shape[2:] != target.shape[1:]:
            target = F.interpolate(
                target.unsqueeze(1).float(),
                size=logits.shape[2:], mode='nearest').squeeze(1)

        if target.dtype == torch.float32:
            # Soft CE for MixUp targets — target is (B,H,W) in [0,1]
            log_p = F.log_softmax(logits, dim=1)   # (B,2,H,W)
            w_bg  = 1.0
            w_pl  = self.pw
            loss  = -(w_bg * (1 - target) * log_p[:, 0]
                    + w_pl *  target       * log_p[:, 1])
            return loss.mean()
        else:
            w = torch.tensor([1.0, self.pw], device=logits.device)
            return F.cross_entropy(logits, target.long(), weight=w)

    def _ftl(self, logits, target):
        if target.dtype == torch.float32:
            return self.ftl(logits, (target > 0.5).long())
        return self.ftl(logits, target)

    def _single(self, logits, target):
        return 0.5 * self._ce(logits, target) + 0.5 * self._ftl(logits, target)

    def forward(self, outputs, target):
        if not isinstance(outputs, (tuple, list)):
            return self._single(outputs, target)
        main, *aux = outputs
        loss = self._single(main, target)
        for a in aux:
            loss = loss + 0.4 * self._single(a, target)
        return loss




def compute_metrics(preds, target, num_classes: int = 2):
    
   
    if preds.dim() == 4:
        preds = torch.argmax(preds, dim=1)          # (B,H,W) int

    if target.dim() == 4:                            # (B,1,H,W) float
        target = (target.squeeze(1) > 0.5).long()
    elif target.dtype == torch.float32:              # (B,H,W) soft MixUp
        target = (target > 0.5).long()

    iou = torch.zeros(num_classes, device=preds.device)
    pa  = torch.zeros(num_classes, device=preds.device)

    for cls in range(num_classes):
        pred_c = (preds  == cls)
        tgt_c  = (target == cls)
        total  = tgt_c.sum().float()

        inter = (pred_c & tgt_c).sum().float()
        union = (pred_c | tgt_c).sum().float()

        pa[cls]  = inter / total.clamp(min=1e-6)
        iou[cls] = inter / union.clamp(min=1e-6)

    mIoU = iou.mean().item()
    mPA  = pa.mean().item()

    # F1 / Prec / Rec for plastic class (cls=1)
    tp = iou[1].item() * (((preds == 1) | (target == 1)).sum().float().item())
    # recompute cleanly
    tp2 = ((preds == 1) & (target == 1)).sum().float().item()
    fp  = ((preds == 1) & (target == 0)).sum().float().item()
    fn  = ((preds == 0) & (target == 1)).sum().float().item()

    prec = tp2 / (tp2 + fp + 1e-6)
    rec  = tp2 / (tp2 + fn + 1e-6)
    f1   = 2 * prec * rec / (prec + rec + 1e-6)

    return {
        'bg_iou'      : iou[0].item(),
        'plastic_iou' : iou[1].item(),
        'mIoU'        : mIoU,
        'bg_pa'       : pa[0].item(),
        'plastic_pa'  : pa[1].item(),
        'mPA'         : mPA,
        'f1'          : f1,
        'precision'   : prec,
        'recall'      : rec,
    }

if __name__ == "__main__":
    import torch

    logit  = torch.randn(2, 2, 64, 64)   # BiSeNet: 2-class output
    target = (torch.rand(2, 64, 64) > 0.9).long()

    loss_fn = BiSeNetLoss(plastic_weight=13.3)
    loss = loss_fn(logit, target)
    print(f"Loss (hard): {loss.item():.4f}")

    # Soft MixUp loss
    soft_t = target.float() * 0.9
    loss_s = loss_fn(logit, soft_t)
    print(f"Loss (soft): {loss_s.item():.4f}")

    m = compute_metrics(logit, target)
    print(f"\nMetrics:")
    print(f"  bg_iou      = {m['bg_iou']:.4f}")
    print(f"  plastic_iou = {m['plastic_iou']:.4f}")
    print(f"  mIoU        = {m['mIoU']:.4f}")
    print(f"  mPA         = {m['mPA']:.4f}")
    print(f"  F1          = {m['f1']:.4f}")
    print(f"  Precision   = {m['precision']:.4f}")
    print(f"  Recall      = {m['recall']:.4f}")

    print("\nbisenet_dataset.py OK")