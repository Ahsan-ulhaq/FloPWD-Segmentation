"""
bisenet_infer.py — Inference on new (unlabelled) dataset with trained BiSeNetV2
================================================================================
Loads best_model_bisenet.pth (raw state_dict, saved via torch.save(model.state_dict(),...))
Runs TTA (3 scales x hflip, identical to training/val) -> argmax mask.
Outputs per image:
  <name>_mask.png     binary mask (0/255, single channel)
  <name>_overlay.png  original image with red plastic overlay
Also writes a CSV with plastic pixel-fraction per image.

Requires bisenet2.py (BiSeNetV2 class) importable — place in same folder or add to PYTHONPATH.
"""

import os
import csv
import glob
import argparse

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from bisenet2 import BiSeNetV2

# =============================================================================
# CONFIG — edit paths here or pass as CLI args
# =============================================================================
DATASET_DIR = r"/home/2022bcse003/binary_semantic/wuler1/UBYK2513"
MODEL_PATH  = r"/home/2022bcse003/binary_semantic/best_model_bisenet.pth"
OUTPUT_DIR  = r"/home/2022bcse003/binary_semantic/wuler_bisenet_pred"
IMG_GLOB    = "wular_img_*.*"          # matches beach_img_0000.jpg etc.
MAX_SIDE    = None                     # images are 1024x2048 — no downsize needed; set an int to cap
USE_TTA     = True                     # 3 scales x hflip, matches training val protocol
PLASTIC_THRESH = 0.5                  # P(plastic) > thresh -> plastic (was argmax/0.5)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
TTA_SCALES    = [0.75, 1.0, 1.25]


# =============================================================================
# Preprocessing
# =============================================================================
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def load_image(path, max_side=None):
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    if max_side is not None and max(orig_w, orig_h) > max_side:
        scale = max_side / max(orig_w, orig_h)
        nw, nh = int(round(orig_w * scale)), int(round(orig_h * scale))
        img = img.resize((nw, nh), Image.BILINEAR)
    arr = np.array(img)
    tensor = normalize(to_tensor(arr)).unsqueeze(0)  # (1,3,H,W)
    return tensor, arr, (orig_w, orig_h)


# =============================================================================
# TTA predict (identical protocol to bisenet_train.py::tta_predict)
# =============================================================================
@torch.no_grad()
def tta_predict(model, img, device, use_tta=True):
    img = img.to(device)
    H, W = img.shape[2:]
    model.aux_mode = "eval"

    if not use_tta:
        out = model(img)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        return torch.softmax(logits, dim=1)

    acc = torch.zeros(img.shape[0], 2, H, W, device=device)
    n = 0
    for scale in TTA_SCALES:
        for flip in (False, True):
            x = torch.flip(img, [-1]) if flip else img
            if scale != 1.0:
                nh, nw = int(H * scale), int(W * scale)
                x = F.interpolate(x, (nh, nw), mode="bilinear", align_corners=False)
            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            prob = torch.softmax(logits, dim=1)
            prob = F.interpolate(prob, (H, W), mode="bilinear", align_corners=False)
            if flip:
                prob = torch.flip(prob, [-1])
            acc += prob
            n += 1
    return acc / n


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default=DATASET_DIR)
    ap.add_argument("--model_path", default=MODEL_PATH)
    ap.add_argument("--output_dir", default=OUTPUT_DIR)
    ap.add_argument("--max_side", type=int, default=MAX_SIDE if MAX_SIDE else 0)
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--threshold", type=float, default=PLASTIC_THRESH)
    args = ap.parse_args()

    max_side = args.max_side if args.max_side > 0 else None
    use_tta = USE_TTA and not args.no_tta
    thresh = args.threshold
    print(f"Plastic probability threshold: {thresh}")

    os.makedirs(args.output_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.dataset_dir, IMG_GLOB)))
    if not paths:
        raise FileNotFoundError(f"No images matching {IMG_GLOB} in {args.dataset_dir}")
    print(f"Found {len(paths)} images.")

    device = torch.device(DEVICE)
    print(f"Device: {device}  |  TTA: {use_tta}  |  max_side: {max_side}")

    model = BiSeNetV2(n_classes=2, aux_mode="eval").to(device)
    state = torch.load(args.model_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_issues = [k for k in list(missing) + list(unexpected)
                   if "total_ops" not in k and "total_params" not in k]
    if real_issues:
        print(f"[WARNING] state_dict mismatch: {real_issues[:5]}")
    model.eval()

    csv_path = os.path.join(args.output_dir, "plastic_fraction.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "plastic_pixel_fraction", "plastic_pixels", "total_pixels"])

        for i, path in enumerate(paths):
            name = os.path.splitext(os.path.basename(path))[0]
            img_t, orig_arr, (ow, oh) = load_image(path, max_side)

            probs = tta_predict(model, img_t, device, use_tta)   # (1,2,h,w) at processed size
            plastic_prob = probs[0, 1].cpu().numpy()              # (h,w)
            pred = (plastic_prob > thresh).astype(np.uint8)       # (h,w) thresholded, not argmax

            # resize mask back to original resolution if we downsized
            if pred.shape[:2] != (oh, ow):
                pred_img = Image.fromarray((pred * 255).astype(np.uint8)).resize((ow, oh), Image.NEAREST)
                pred = (np.array(pred_img) > 127).astype(np.uint8)
                orig_full = np.array(Image.open(path).convert("RGB"))
            else:
                orig_full = orig_arr

            mask_u8 = (pred * 255).astype(np.uint8)
            Image.fromarray(mask_u8).save(os.path.join(args.output_dir, f"{name}_mask.png"))

            overlay = orig_full.copy()
            overlay[pred == 1] = (0.5 * overlay[pred == 1] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
            Image.fromarray(overlay).save(os.path.join(args.output_dir, f"{name}_overlay.png"))

            plastic_px = int(pred.sum())
            total_px = pred.size
            frac = plastic_px / total_px
            writer.writerow([name, f"{frac:.6f}", plastic_px, total_px])

            print(f"[{i+1}/{len(paths)}] {name}  plastic_frac={frac:.4f}")

    print(f"\nDone. Masks/overlays -> {args.output_dir}")
    print(f"Summary CSV -> {csv_path}")


if __name__ == "__main__":
    main()
