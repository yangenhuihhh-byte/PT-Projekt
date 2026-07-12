from pathlib import Path
from datetime import datetime
import csv
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )

        self.pool2 = nn.MaxPool2d(2)

        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        middle = self.middle(p2)

        u2 = self.up2(middle)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.out(d1)


class PatchDataset(Dataset):
    def __init__(self, images_dir, heights_dir):
        self.image_files = sorted(Path(images_dir).glob("img_*.pt"))
        self.height_files = sorted(Path(heights_dir).glob("height_*.pt"))

        if len(self.image_files) != len(self.height_files):
            raise ValueError(
                "The number of image patches and height patches does not match."
            )

        if len(self.image_files) == 0:
            raise ValueError("No patch files were found.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = torch.load(
            self.image_files[index],
            map_location="cpu",
            weights_only=True,
        ).float()

        height = torch.load(
            self.height_files[index],
            map_location="cpu",
            weights_only=True,
        ).float()

        return image, height


PROJECT_DIR = Path("E:/PT Projekt")

MODEL_NAME = "unet_patch128_ep100"
EXPERIMENT_NAME = "experiment_05_final_100epoch"

MODEL_PATH = PROJECT_DIR / f"{MODEL_NAME}.pth"

IMAGES_DIR = PROJECT_DIR / "dataset_alicona" / "images"
HEIGHTS_DIR = PROJECT_DIR / "dataset_alicona" / "heights"

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = RESULTS_DIR / "metrics_summary.csv"

BATCH_SIZE = 16
NUM_WORKERS = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")

dataset = PatchDataset(
    images_dir=IMAGES_DIR,
    heights_dir=HEIGHTS_DIR,
)

data_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

print(f"Number of evaluated samples: {len(dataset)}")

model = SmallUNet().to(device)

model_state = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(model_state)
model.eval()

total_absolute_error = 0.0
total_squared_error = 0.0
total_elements = 0

with torch.no_grad():
    for batch_index, (images, heights) in enumerate(data_loader, start=1):
        images = images.to(device)
        heights = heights.to(device)

        predictions = model(images).squeeze(1)

        if predictions.shape != heights.shape:
            raise ValueError(
                "Prediction and target shapes do not match: "
                f"{predictions.shape} vs {heights.shape}"
            )

        difference = predictions - heights

        total_absolute_error += torch.abs(difference).sum().item()
        total_squared_error += (difference ** 2).sum().item()
        total_elements += difference.numel()

        if batch_index % 50 == 0 or batch_index == len(data_loader):
            print(
                f"Processed batch {batch_index}/{len(data_loader)}"
            )

mean_mae = total_absolute_error / total_elements
mean_mse = total_squared_error / total_elements
mean_rmse = math.sqrt(mean_mse)

print()
print(f"Mean MAE:  {mean_mae:.10f}")
print(f"Mean MSE:  {mean_mse:.10f}")
print(f"Mean RMSE: {mean_rmse:.10f}")

file_exists = METRICS_PATH.exists()

with METRICS_PATH.open(
    mode="a",
    newline="",
    encoding="utf-8",
) as csv_file:
    writer = csv.writer(csv_file)

    if not file_exists:
        writer.writerow([
            "timestamp",
            "experiment",
            "model",
            "num_samples",
            "mean_mae",
            "mean_mse",
            "mean_rmse",
        ])

    writer.writerow([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        EXPERIMENT_NAME,
        MODEL_NAME,
        len(dataset),
        mean_mae,
        mean_mse,
        mean_rmse,
    ])

print(f"Summary metrics saved to: {METRICS_PATH}")