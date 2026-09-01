from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_utils import (
    DEFAULT_PATCH_INDEX_CSV,
    DEFAULT_SPLIT_CSV,
    PROJECT_DIR,
    AliconaPatchDataset,
    ensure_prepared,
)
from models import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train height-map prediction models.")
    parser.add_argument("--model", default="small_unet", choices=["simple_cnn", "small_unet", "transunet_lite"])
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--shuffle-patches", action="store_true")
    parser.add_argument("--limit-train-patches", type=int, default=None)
    parser.add_argument("--limit-val-patches", type=int, default=None)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--patch-index-csv", type=Path, default=DEFAULT_PATCH_INDEX_CSV)
    parser.add_argument("--models-dir", type=Path, default=PROJECT_DIR / "models")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_DIR / "results")
    return parser.parse_args()


def mean_absolute_metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    difference = prediction - target
    mae = torch.mean(torch.abs(difference)).item()
    mse = torch.mean(difference * difference).item()
    rmse = mse**0.5
    return mae, mse, rmse


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_abs = 0.0
    total_squared = 0.0
    total_elements = 0

    with torch.no_grad():
        for images, heights in data_loader:
            images = images.to(device)
            heights = heights.to(device)
            predictions = model(images)
            loss = criterion(predictions, heights)

            difference = predictions - heights
            total_loss += loss.item()
            total_abs += torch.abs(difference).sum().item()
            total_squared += (difference * difference).sum().item()
            total_elements += difference.numel()

    mae = total_abs / total_elements
    mse = total_squared / total_elements
    rmse = mse**0.5
    mean_loss = total_loss / max(len(data_loader), 1)
    return mean_loss, mae, mse, rmse


def append_training_log(path: Path, row: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(
                [
                    "timestamp",
                    "model",
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "val_mae_norm",
                    "val_mse_norm",
                    "val_rmse_norm",
                ]
            )
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    ensure_prepared(
        split_csv=args.split_csv,
        patch_index_csv=args.patch_index_csv,
        patch_size=args.patch_size,
        stride=args.stride,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataset = AliconaPatchDataset(
        split="train",
        split_csv=args.split_csv,
        patch_index_csv=args.patch_index_csv,
        limit=args.limit_train_patches,
    )
    val_dataset = AliconaPatchDataset(
        split="val",
        split_csv=args.split_csv,
        patch_index_csv=args.patch_index_csv,
        limit=args.limit_val_patches,
    )
    print(f"Train patches: {len(train_dataset)}")
    print(f"Validation patches: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle_patches,
        num_workers=0,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = create_model(args.model).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    model_stem = f"{args.model}_patch{args.patch_size}_stride{args.stride}_ep{args.epochs}"
    best_path = args.models_dir / f"{model_stem}_best.pth"
    last_path = args.models_dir / f"{model_stem}_last.pth"
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for images, heights in train_loader:
            images = images.to(device)
            heights = heights.to(device)

            predictions = model(images)
            loss = criterion(predictions, heights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss, val_mae, val_mse, val_rmse = evaluate(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_mae_norm={val_mae:.6f} | "
            f"val_rmse_norm={val_rmse:.6f}"
        )

        append_training_log(
            args.results_dir / "training_log.csv",
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                args.model,
                epoch,
                train_loss,
                val_loss,
                val_mae,
                val_mse,
                val_rmse,
            ],
        )

        if val_mae < best_val:
            best_val = val_mae
            torch.save(model.state_dict(), best_path)

    torch.save(model.state_dict(), last_path)
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print("All MAE/MSE/RMSE values are computed on min-max normalized height maps.")


if __name__ == "__main__":
    main()
