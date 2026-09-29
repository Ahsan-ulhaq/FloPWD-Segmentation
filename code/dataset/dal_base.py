import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A


class DalLake(Dataset):
    def __init__(self, root, split='train'):
        super(DalLake, self).__init__()
        self.root = root
        self.split = split

        self.images, self.labels, self.names = [], [], []

        split_dir = os.path.join(root, split)
        label_dir = os.path.join(root, 'testlabel') if split == 'test' else split_dir

        if not os.path.exists(split_dir):
            raise ValueError(f"Directory {split_dir} does not exist.")
        if not os.path.exists(label_dir):
            raise ValueError(f"Label directory {label_dir} does not exist.")

        for img_file in sorted(os.listdir(split_dir)):
            if img_file.endswith(('.jpg', '.png')):
                img_path = os.path.join(split_dir, img_file)
                base_name = os.path.splitext(img_file)[0]
                label_file = f"{base_name}_trainLabelId.png"
                label_path = os.path.join(label_dir, label_file)
                if os.path.exists(label_path):
                    self.images.append(img_path)
                    self.labels.append(label_path)
                    self.names.append(img_file)

        if not self.images:
            raise ValueError(f"No valid images found in {split_dir}")

        print(f"[{split}] Loaded {len(self.images)} image-label pairs.")

        if split == 'train':
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ColorJitter(
                    brightness=0.1, contrast=0.1,
                    saturation=0.1, hue=0.1, p=0.3
                ),
            ], additional_targets={'label': 'mask'})
        else:
            self.transform = A.Compose(
                [], additional_targets={'label': 'mask'}
            )

        self.to_tensor  = transforms.ToTensor()
        self.normalize  = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path  = self.images[index]
        lbl_path  = self.labels[index]
        name      = self.names[index]

        img = np.array(Image.open(img_path).convert('RGB'))
        lbl = np.array(Image.open(lbl_path).convert('L'))

        out = self.transform(image=img, label=lbl)
        img, lbl = out['image'], out['label']

        img = self.normalize(self.to_tensor(img))           # (3, H, W) float32
        lbl = torch.from_numpy(lbl.astype(np.int64))        # (H, W)    int64

        return img, lbl, name


if __name__ == "__main__":
    root = "D:/ahsan/DalLake"

    for split in ('train', 'val', 'test'):
        try:
            ds = DalLake(root=root, split=split)
            img, lbl, name = ds[0]
            print(f"  split={split} | len={len(ds)} | img={tuple(img.shape)} | lbl={tuple(lbl.shape)} | name={name}")
        except ValueError as e:
            print(f"  split={split} | skipped: {e}")
