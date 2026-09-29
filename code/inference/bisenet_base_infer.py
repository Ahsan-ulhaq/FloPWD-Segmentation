"""
bisenet_base_infer.py — Inference for the PLAIN-CE baseline BiSeNetV2
=====================================================================
Companion to train_bisenet.py / dal_base.py (unweighted CrossEntropyLoss,
no PAFGO augmentation, no TTA). Checkpoint: checkpoints_bisenet/best_model.pth

Val protocol in train_bisenet.py = single forward pass -> argmax, so the
default here is a plain argmax (threshold 0.5). Use --threshold to override.

Outputs per image:
  <name>_mask.png     binary mask (0/255)
  <name>_overlay.png  image with red plastic overlay
  <name>_pred.png     red-on-black map (same style as save_predicted_images)
Plus plastic_fraction.csv.

Requires bisenet2.py importable (same folder or on PYTHONPATH).
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

from mobilenet import MobileNetV2Seg  # =============================================================================
# CONFIG
# =============================================================================
DATASET_DIR = r"/home/2022bcse003/binary_semantic/validation"
MODEL_PATH  = r"/home/2022bcse003/best_model.pth"
OUTPUT_DIR  = r"/home/2022bcse003/binary_semantic/val_mobilenet_Final_preds_before"
IMG_GLOB    = "img*.*"          # matches beach_img_0000.jpg etc.
RESIZE_HW   = None          # None = native 1024x2048. Set e.g. (720, 1280) if OOM.
PLASTIC_THRESH = 0.5      # 0.5 == argmax, matches this model's own val protocol
SAVE_PRED_RGB  = True       # also write red-on-black _pred.png
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def load_image(path, resize_hw=None):
    """Returns (1,3,H,W) normalised tensor, original RGB array, (orig_w, orig_h)."""
    img = Image.open(path).convert("RGB")
    ow, oh = img.size
    if resize_hw is not None:
        img = img.resize((resize_hw[1], resize_hw[0]), Image.BILINEAR)
    arr = np.array(img)
    return normalize(to_tensor(arr)).unsqueeze(0), arr, (ow, oh)


@torch.no_grad()
def predict(model, img, device):
    """Single forward pass -> (1,2,H,W) softmax probs. Matches train_bisenet.py val."""
    img = img.to(device)
    model.aux_mode = "eval"
    out = model(img)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    return torch.softmax(logits, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default=DATASET_DIR)
    ap.add_argument("--model_path",  default=MODEL_PATH)
    ap.add_argument("--output_dir",  default=OUTPUT_DIR)
    ap.add_argument("--threshold", type=float, default=PLASTIC_THRESH,
                    help="P(plastic) > thresh -> plastic. 0.5 == argmax.")
    ap.add_argument("--resize", type=int, nargs=2, default=None, metavar=("H", "W"),
                    help="Run at this size instead of native (e.g. --resize 720 1280).")
    args = ap.parse_args()

    resize_hw = tuple(args.resize) if args.resize else RESIZE_HW
    os.makedirs(args.output_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.dataset_dir, IMG_GLOB)))
    if not paths:
        raise FileNotFoundError(f"No images matching {IMG_GLOB} in {args.dataset_dir}")

    device = torch.device(DEVICE)
    print(f"Images: {len(paths)} | Device: {device} | "
          f"Threshold: {args.threshold} | Resize: {resize_hw or 'native'}")

    model = MobileNetV2Seg(num_classes=2, pretrained=False).to(device)
    state = torch.load(args.model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    real = [k for k in list(missing) + list(unexpected)
            if "total_ops" not in k and "total_params" not in k]
    if real:
        print(f"[WARNING] state_dict mismatch ({len(real)}): {real[:5]}")
    model.eval()

    csv_path = os.path.join(args.output_dir, "plastic_fraction.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "plastic_pixel_fraction", "plastic_pixels",
                    "total_pixels", "max_plastic_prob", "mean_plastic_prob"])

        for i, path in enumerate(paths):
            name = os.path.splitext(os.path.basename(path))[0]
            img_t, proc_arr, (ow, oh) = load_image(path, resize_hw)

            probs = predict(model, img_t, device)          # (1,2,h,w)
            pprob = probs[0, 1].cpu().numpy()              # (h,w) P(plastic)
            pred  = (pprob > args.threshold).astype(np.uint8)

            # back to original resolution if we resized
            if pred.shape != (oh, ow):
                pred = (np.array(Image.fromarray(pred * 255).resize(
                    (ow, oh), Image.NEAREST)) > 127).astype(np.uint8)
                base_rgb = np.array(Image.open(path).convert("RGB"))
            else:
                base_rgb = proc_arr

            Image.fromarray((pred * 255).astype(np.uint8)).save(
                os.path.join(args.output_dir, f"{name}_mask.png"))

            ov = base_rgb.copy()
            ov[pred == 1] = (0.5 * ov[pred == 1] +
                             0.5 * np.array([255, 0, 0])).astype(np.uint8)
            Image.fromarray(ov).save(os.path.join(args.output_dir, f"{name}_overlay.png"))

            if SAVE_PRED_RGB:
                rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
                rgb[pred == 1] = [255, 0, 0]
                Image.fromarray(rgb).save(os.path.join(args.output_dir, f"{name}_pred.png"))

            n_pl, n_tot = int(pred.sum()), pred.size
            w.writerow([name, f"{n_pl / n_tot:.6f}", n_pl, n_tot,
                        f"{pprob.max():.4f}", f"{pprob.mean():.6f}"])
            print(f"[{i+1}/{len(paths)}] {name}  frac={n_pl/n_tot:.4f}  "
                  f"max_p={pprob.max():.3f}")

    print(f"\nDone -> {args.output_dir}\nCSV -> {csv_path}")


if __name__ == "__main__":
    main()
