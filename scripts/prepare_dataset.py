import matplotlib.pyplot as plt
from pathlib import Path
PATCH_SIZE = 256
STRIDE = 256
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class SurfaceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)

        self.samples = []

        for folder in self.root_dir.iterdir():
            if folder.is_dir():

                texture = folder / "texture.bmp"
                height = folder / "dem.al3d"

                if texture.exists() and height.exists():
                    self.samples.append((texture, height))

    def __len__(self):
        return len(self.samples)

    def read_al3d(self, path):
        data = np.fromfile(path, dtype=np.float32)
        return data

    def __getitem__(self, idx):
        texture_path, height_path = self.samples[idx]

        img = Image.open(texture_path).convert("RGB")
        img = np.array(img) / 255.0

        height = self.read_al3d(height_path)

        img = torch.tensor(img).permute(2, 0, 1).float()
        height = torch.tensor(height).float()

        return img, height

patch_size = 256
stride = 192
out_dir = Path("E:/PT Projekt/dataset_alicona")

img_out = out_dir / "images"
height_out = out_dir / "heights"

img_out.mkdir(parents=True, exist_ok=True)
height_out.mkdir(parents=True, exist_ok=True)
dataset = SurfaceDataset("E:/PT Projekt/data/Versuchsreihe2_B91$prj")

print("数据数量:", len(dataset))

count = 0

for sample_idx in range(len(dataset)):

    x, y = dataset[sample_idx]

    h, w = x.shape[1], x.shape[2]
    expected = h * w

    height_2d = y[-expected:].reshape(h, w)
    height_2d = (height_2d - height_2d.min()) / (height_2d.max() - height_2d.min() + 1e-8)

    for top in range(0, h - patch_size + 1, stride):
        for left in range(0, w - patch_size + 1, stride):

           img_patch = x[:, top:top+patch_size, left:left+patch_size]
           height_patch = height_2d[top:top+patch_size, left:left+patch_size]

           img_patch = img_patch.half()
           height_patch = height_patch.half()

           torch.save(img_patch, img_out / f"img_{count:05d}.pt")
           torch.save(height_patch, height_out / f"height_{count:05d}.pt")

           count += 1

print("保存 patch 数量:", count)
print("保存位置:", out_dir)