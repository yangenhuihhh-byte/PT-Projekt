from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


# =========================
# Dataset
# =========================
class AliconaDataset(Dataset):

    def __init__(self, root_dir):

        self.image_dir = Path(root_dir) / "images"
        self.height_dir = Path(root_dir) / "heights"

        self.image_files = sorted(list(self.image_dir.glob("*.pt")))
        self.height_files = sorted(list(self.height_dir.glob("*.pt")))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        x = torch.load(self.image_files[idx]).float()
        y = torch.load(self.height_files[idx]).float()

        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        y = y - y.mean()
        y = y / (y.std() + 1e-8)

        y = y.unsqueeze(0)

        return x, y


# =========================
# Small UNet
# =========================
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
# =========================
# Main
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("使用设备:", device)

dataset = AliconaDataset("E:/PT Projekt/dataset_alicona")

print("数据集大小:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

model = SmallUNet().to(device)

criterion = nn.L1Loss()

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4
)

epochs = 100

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for x, y in loader:

        x = x.to(device).float()
        y = y.to(device).float()

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        pred = model(x)

        loss = criterion(pred, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.10f}")

torch.save(model.state_dict(), "unet_patch128_ep100.pth")

print("模型已保存")