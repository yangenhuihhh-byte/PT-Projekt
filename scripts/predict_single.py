from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from data_utils import (
    DEFAULT_PATCH_INDEX_CSV,
    DEFAULT_SPLIT_CSV,
    PROJECT_DIR,
    AliconaPatchDataset,
    ensure_prepared,
)
from models import create_model
from predict_full_image import colorize, make_summary, save_rgb_preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict and visualize one indexed patch.")
    parser.add_argument("--model", default="small_unet", choices=["simple_cnn", "small_unet", "transunet_lite"])
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_DIR / "unet_patch128_ep100.pth")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--patch-index-csv", type=Path, default=DEFAULT_PATCH_INDEX_CSV)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_DIR / "results" / "single_patch_predictions")
    return parser.parse_args()


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

    dataset = AliconaPatchDataset(
        split=args.split,
        split_csv=args.split_csv,
        patch_index_csv=args.patch_index_csv,
    )
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"Patch index must be in [0, {len(dataset) - 1}]")

    image, height = dataset[args.index]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.model).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        prediction = model(image.unsqueeze(0).to(device)).squeeze().cpu()

    target = height.squeeze()
    difference = prediction - target
    mae = torch.mean(torch.abs(difference)).item()
    mse = torch.mean(difference * difference).item()
    rmse = mse**0.5

    output_dir = args.results_dir / f"{args.split}_idx{args.index}_{args.model}"
    output_dir.mkdir(parents=True, exist_ok=True)

    texture_np = image.permute(1, 2, 0).numpy()
    texture_image = save_rgb_preview(texture_np, output_dir / "input_texture.png")
    true_image = colorize(target.numpy(), 0.0, 1.0)
    pred_image = colorize(prediction.numpy(), 0.0, 1.0)
    error_image = colorize(np.abs(difference.numpy()), 0.0, max(1e-8, float(torch.quantile(torch.abs(difference), 0.99))))

    true_image.save(output_dir / "true_height_norm.png")
    pred_image.save(output_dir / "predicted_height_norm.png")
    error_image.save(output_dir / "absolute_error_norm.png")
    make_summary(texture_image, true_image, pred_image, error_image, output_dir / "summary.png", preview_size=256)

    metrics = Image.new("RGB", (360, 80), "white")
    draw = ImageDraw.Draw(metrics)
    draw.text((8, 8), f"MAE_norm:  {mae:.6f}", fill=(0, 0, 0))
    draw.text((8, 30), f"MSE_norm:  {mse:.6f}", fill=(0, 0, 0))
    draw.text((8, 52), f"RMSE_norm: {rmse:.6f}", fill=(0, 0, 0))
    metrics.save(output_dir / "metrics.png")

    print(f"MAE_norm:  {mae:.10f}")
    print(f"MSE_norm:  {mse:.10f}")
    print(f"RMSE_norm: {rmse:.10f}")
    print(f"Saved single-patch output to: {output_dir}")


if __name__ == "__main__":
    main()
