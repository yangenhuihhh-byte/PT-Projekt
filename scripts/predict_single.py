from pathlib import Path
from datetime import datetime
import csv
import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# ============================================================
# Model definitions
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.pool2 = nn.MaxPool2d(2)

        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2,
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2,
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.out = nn.Conv2d(32, 1, kernel_size=1)

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


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path("E:/PT Projekt")

MODEL_NAME = "unet_patch128_ep100"
MODEL_PATH = PROJECT_DIR / f"{MODEL_NAME}.pth"

IMAGE_PATH = (
    PROJECT_DIR
    / "dataset_alicona"
    / "images"
    / "img_00000.pt"
)

HEIGHT_PATH = (
    PROJECT_DIR
    / "dataset_alicona"
    / "heights"
    / "height_00000.pt"
)

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_PATH = (
    RESULTS_DIR
    / f"prediction_{MODEL_NAME}.png"
)

METRICS_PATH = RESULTS_DIR / "metrics.csv"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")


# ============================================================
# Check required files
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Input image tensor not found: {IMAGE_PATH}"
    )

if not HEIGHT_PATH.exists():
    raise FileNotFoundError(
        f"Height tensor not found: {HEIGHT_PATH}"
    )


# ============================================================
# Load model
# ============================================================

model = SmallUNet().to(device)

model_state = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(model_state)
model.eval()


# ============================================================
# Load prediction sample
# ============================================================

x = torch.load(
    IMAGE_PATH,
    map_location="cpu",
    weights_only=True,
).float()

y_true = torch.load(
    HEIGHT_PATH,
    map_location="cpu",
    weights_only=True,
).float()

# Add batch dimension: [C, H, W] -> [1, C, H, W]
x_batch = x.unsqueeze(0).to(device)


# ============================================================
# Prediction
# ============================================================

with torch.no_grad():
    y_pred = model(x_batch)

# Convert prediction to CPU and remove batch/channel dimensions
y_pred = y_pred.squeeze().cpu()
y_true = y_true.squeeze().cpu()

if y_pred.shape != y_true.shape:
    raise ValueError(
        "Prediction and ground-truth shapes do not match: "
        f"{y_pred.shape} vs {y_true.shape}"
    )


# ============================================================
# Evaluation metrics
# ============================================================

difference = y_pred - y_true

mae = torch.mean(torch.abs(difference)).item()
mse = torch.mean(difference ** 2).item()
rmse = math.sqrt(mse)

print(f"MAE:  {mae:.10f}")
print(f"MSE:  {mse:.10f}")
print(f"RMSE: {rmse:.10f}")


# ============================================================
# Save metrics to CSV
# ============================================================

metrics_file_exists = METRICS_PATH.exists()

with METRICS_PATH.open(
    mode="a",
    newline="",
    encoding="utf-8",
) as csv_file:
    writer = csv.writer(csv_file)

    if not metrics_file_exists:
        writer.writerow([
            "timestamp",
            "model",
            "sample",
            "mae",
            "mse",
            "rmse",
        ])

    writer.writerow([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        MODEL_NAME,
        IMAGE_PATH.stem,
        mae,
        mse,
        rmse,
    ])

print(f"Metrics saved to: {METRICS_PATH}")


# ============================================================
# Visualization
# ============================================================

figure = plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(x.permute(1, 2, 0).numpy())
plt.title("Input texture")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(y_true.numpy(), cmap="jet")
plt.title("True height")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(y_pred.numpy(), cmap="jet")
plt.title("Predicted height")
plt.axis("off")

plt.tight_layout()

figure.savefig(
    PREDICTION_PATH,
    dpi=300,
    bbox_inches="tight",
)

print(f"Prediction saved to: {PREDICTION_PATH}")

plt.show()