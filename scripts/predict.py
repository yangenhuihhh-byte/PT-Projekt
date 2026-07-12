from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.pool2 = nn.MaxPool2d(2)

        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        m = self.middle(p2)

        u2 = self.up2(m)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.out(d1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SmallUNet().to(device)
model.load_state_dict(torch.load("unet_patch128_ep100.pth"))
model.eval()

img_path = Path("E:/PT Projekt/dataset_alicona/images/img_00000.pt")
height_path = Path("E:/PT Projekt/dataset_alicona/heights/height_00000.pt")

x = torch.load(img_path).float().unsqueeze(0).to(device)
y_true = torch.load(height_path).float()

with torch.no_grad():
    y_pred = model(x).squeeze().cpu()

mae = torch.mean(torch.abs(y_pred - y_true)).item()
print("MAE:", mae)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(x.squeeze().permute(1, 2, 0).cpu())
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

from pathlib import Path

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

save_path = results_dir / "prediction_unet_patch128_ep100.png"

plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches="tight")

print(f"Prediction saved to: {save_path.resolve()}")

plt.show()