from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_utils import DEFAULT_PATCH_INDEX_CSV, DEFAULT_SPLIT_CSV, PROJECT_DIR, AliconaPatchDataset, ensure_prepared
from models import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on train/val/test patches.")
    parser.add_argument("--model", default="small_unet", choices=["simple_cnn", "small_unet", "transunet_lite"])
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_DIR / "unet_patch128_ep100.pth")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-patches", type=int, default=None)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--patch-index-csv", type=Path, default=DEFAULT_PATCH_INDEX_CSV)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_DIR / "results")
    return parser.parse_args()


def append_metrics(path: Path, row: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(
                [
                    "timestamp",
                    "split",
                    "model",
                    "checkpoint",
                    "num_patches",
                    "patch_size",
                    "stride",
                    "mae_norm",
                    "mse_norm",
                    "rmse_norm",
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

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AliconaPatchDataset(
        split=args.split,
        split_csv=args.split_csv,
        patch_index_csv=args.patch_index_csv,
        limit=args.limit_patches,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = create_model(args.model).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    total_abs = 0.0
    total_squared = 0.0
    total_elements = 0

    with torch.no_grad():
        for index, (images, heights) in enumerate(loader, start=1):
            images = images.to(device)
            heights = heights.to(device)
            predictions = model(images)
            difference = predictions - heights

            total_abs += torch.abs(difference).sum().item()
            total_squared += (difference * difference).sum().item()
            total_elements += difference.numel()

            if index % 50 == 0 or index == len(loader):
                print(f"Processed batch {index}/{len(loader)}")

    mae = total_abs / total_elements
    mse = total_squared / total_elements
    rmse = mse**0.5

    print(f"Split: {args.split}")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Number of evaluated patches: {len(dataset)}")
    print(f"MAE_norm:  {mae:.10f}")
    print(f"MSE_norm:  {mse:.10f}")
    print(f"RMSE_norm: {rmse:.10f}")
    print("These errors are dimensionless because they are computed after min-max height normalization.")

    if args.limit_patches is None:
        append_metrics(
            args.results_dir / "model_comparison.csv",
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                args.split,
                args.model,
                str(args.checkpoint.relative_to(PROJECT_DIR) if args.checkpoint.is_relative_to(PROJECT_DIR) else args.checkpoint),
                len(dataset),
                args.patch_size,
                args.stride,
                mae,
                mse,
                rmse,
            ],
        )
        print(f"Saved metrics to {args.results_dir / 'model_comparison.csv'}")
    else:
        print("Limit mode was used; metrics were not appended to model_comparison.csv.")


if __name__ == "__main__":
    main()
