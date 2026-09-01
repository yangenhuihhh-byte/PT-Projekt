from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from data_utils import (
    DEFAULT_SPLIT_CSV,
    PROJECT_DIR,
    inverse_minmax,
    load_height,
    load_texture,
    normalize_minmax,
    patch_starts,
    read_split_csv,
)
from models import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a complete microscope image by patch stitching.")
    parser.add_argument("--model", default="small_unet", choices=["simple_cnn", "small_unet", "transunet_lite"])
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_DIR / "unet_patch128_ep100.pth")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample", default=None, help="Sample folder name, for example V25_O_A$3D.")
    parser.add_argument("--all-samples", action="store_true", help="Predict every sample in the selected split.")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--blend", default="gaussian", choices=["none", "uniform", "gaussian"])
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_DIR / "results" / "full_image_predictions")
    parser.add_argument("--preview-size", type=int, default=512)
    return parser.parse_args()


def gaussian_kernel(size: int, sigma_scale: float = 0.25) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    sigma = sigma_scale
    kernel = np.exp(-0.5 * (xx * xx + yy * yy) / (sigma * sigma))
    kernel = kernel / kernel.max()
    return np.maximum(kernel, 1e-3).astype(np.float32)


def colorize(values: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> Image.Image:
    array = values.astype(np.float32)
    if vmin is None:
        vmin = float(np.nanmin(array))
    if vmax is None:
        vmax = float(np.nanmax(array))
    denom = max(vmax - vmin, 1e-8)
    norm = np.clip((array - vmin) / denom, 0.0, 1.0)

    # A compact blue-cyan-yellow-red color ramp implemented without matplotlib.
    r = np.clip(1.5 * norm - 0.25, 0, 1)
    g = np.clip(1.5 - np.abs(2.0 * norm - 1.0) * 1.5, 0, 1)
    b = np.clip(1.25 - 1.5 * norm, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def save_rgb_preview(image_array: np.ndarray, path: Path) -> Image.Image:
    image = Image.fromarray(np.clip(image_array * 255.0, 0, 255).astype(np.uint8))
    image.save(path)
    return image


def resize_for_preview(image: Image.Image, size: int) -> Image.Image:
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    return image


def make_summary(
    texture: Image.Image,
    true_height: Image.Image,
    predicted_height: Image.Image,
    error_map: Image.Image,
    output_path: Path,
    preview_size: int,
) -> None:
    panels = [
        ("Input texture", resize_for_preview(texture.copy(), preview_size)),
        ("True height", resize_for_preview(true_height.copy(), preview_size)),
        ("Predicted height", resize_for_preview(predicted_height.copy(), preview_size)),
        ("Absolute error", resize_for_preview(error_map.copy(), preview_size)),
    ]
    label_height = 30
    width = sum(panel.width for _, panel in panels)
    height = max(panel.height for _, panel in panels) + label_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    x = 0
    for label, panel in panels:
        draw.text((x + 8, 8), label, fill=(0, 0, 0))
        canvas.paste(panel, (x, label_height))
        x += panel.width
    canvas.save(output_path)


def select_sample(split: str, sample_name: str | None, split_csv: Path):
    samples = [sample for sample in read_split_csv(split_csv) if sample.split == split]
    if not samples:
        raise ValueError(f"No samples found for split '{split}'.")
    if sample_name is None:
        return samples[0]
    for sample in samples:
        if sample.name == sample_name:
            return sample
    raise ValueError(f"Sample {sample_name} was not found in split '{split}'.")


def predict_full_image(
    model: torch.nn.Module,
    texture: np.ndarray,
    patch_size: int,
    stride: int,
    blend: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    rows = patch_starts(texture.shape[0], patch_size, stride)
    cols = patch_starts(texture.shape[1], patch_size, stride)
    coords = [(top, left) for top in rows for left in cols]

    prediction_sum = np.zeros(texture.shape[:2], dtype=np.float32)
    weight_sum = np.zeros(texture.shape[:2], dtype=np.float32)

    if blend == "none":
        weight = np.ones((patch_size, patch_size), dtype=np.float32)
    elif blend == "uniform":
        weight = np.ones((patch_size, patch_size), dtype=np.float32)
    else:
        weight = gaussian_kernel(patch_size)

    model.eval()
    with torch.no_grad():
        for start in range(0, len(coords), batch_size):
            batch_coords = coords[start : start + batch_size]
            patches = []
            for top, left in batch_coords:
                patch = texture[top : top + patch_size, left : left + patch_size, :]
                patches.append(torch.from_numpy(patch).permute(2, 0, 1))

            images = torch.stack(patches).float().to(device)
            predictions = model(images).squeeze(1).cpu().numpy()

            for (top, left), patch_prediction in zip(batch_coords, predictions):
                if blend == "none":
                    prediction_sum[top : top + patch_size, left : left + patch_size] = patch_prediction
                    weight_sum[top : top + patch_size, left : left + patch_size] = 1.0
                else:
                    prediction_sum[top : top + patch_size, left : left + patch_size] += patch_prediction * weight
                    weight_sum[top : top + patch_size, left : left + patch_size] += weight

            print(f"Predicted patches {min(start + batch_size, len(coords))}/{len(coords)}")

    return prediction_sum / np.maximum(weight_sum, 1e-8)


def append_full_image_metrics(path: Path, row: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(
                [
                    "timestamp",
                    "sample",
                    "split",
                    "model",
                    "checkpoint",
                    "patch_size",
                    "stride",
                    "blend",
                    "mae_norm",
                    "mse_norm",
                    "rmse_norm",
                    "mae_raw_height_units",
                    "mse_raw_height_units",
                    "rmse_raw_height_units",
                ]
            )
        writer.writerow(row)


def run_sample_prediction(
    args: argparse.Namespace,
    sample,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    texture = load_texture(sample.path / "texture.bmp")
    height_raw = load_height(sample.path / "dem.al3d", texture.shape[:2])
    height_norm, h_min, h_max = normalize_minmax(height_raw)

    predicted_norm = predict_full_image(
        model=model,
        texture=texture,
        patch_size=args.patch_size,
        stride=args.stride,
        blend=args.blend,
        batch_size=args.batch_size,
        device=device,
    )

    error_norm = predicted_norm - height_norm
    mae_norm = float(np.mean(np.abs(error_norm)))
    mse_norm = float(np.mean(error_norm * error_norm))
    rmse_norm = mse_norm**0.5

    predicted_raw = inverse_minmax(predicted_norm, h_min, h_max)
    error_raw = predicted_raw - height_raw
    mae_raw = float(np.mean(np.abs(error_raw)))
    mse_raw = float(np.mean(error_raw * error_raw))
    rmse_raw = mse_raw**0.5

    run_dir = args.results_dir / f"{sample.name}_{args.model}_p{args.patch_size}_s{args.stride}_{args.blend}"
    run_dir.mkdir(parents=True, exist_ok=True)

    np.save(run_dir / "predicted_height_norm.npy", predicted_norm)
    np.save(run_dir / "true_height_norm.npy", height_norm)
    np.save(run_dir / "absolute_error_norm.npy", np.abs(error_norm))

    texture_image = save_rgb_preview(texture, run_dir / "input_texture.png")
    true_image = colorize(height_norm, 0.0, 1.0)
    pred_image = colorize(predicted_norm, 0.0, 1.0)
    error_image = colorize(np.abs(error_norm), 0.0, max(1e-8, float(np.percentile(np.abs(error_norm), 99))))

    true_image.save(run_dir / "true_height_norm.png")
    pred_image.save(run_dir / "predicted_height_norm.png")
    error_image.save(run_dir / "absolute_error_norm.png")
    make_summary(
        texture=texture_image,
        true_height=true_image,
        predicted_height=pred_image,
        error_map=error_image,
        output_path=run_dir / "summary.png",
        preview_size=args.preview_size,
    )

    append_full_image_metrics(
        args.results_dir / "full_image_metrics.csv",
        [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            sample.name,
            args.split,
            args.model,
            str(args.checkpoint.relative_to(PROJECT_DIR) if args.checkpoint.is_relative_to(PROJECT_DIR) else args.checkpoint),
            args.patch_size,
            args.stride,
            args.blend,
            mae_norm,
            mse_norm,
            rmse_norm,
            mae_raw,
            mse_raw,
            rmse_raw,
        ],
    )

    print(f"Sample: {sample.name}")
    print(f"MAE_norm:  {mae_norm:.10f}")
    print(f"MSE_norm:  {mse_norm:.10f}")
    print(f"RMSE_norm: {rmse_norm:.10f}")
    print(f"MAE_raw_height_units:  {mae_raw:.10f}")
    print(f"MSE_raw_height_units:  {mse_raw:.10f}")
    print(f"RMSE_raw_height_units: {rmse_raw:.10f}")
    print(f"Saved full-image prediction outputs to: {run_dir}")
    return mae_norm, mse_norm, rmse_norm


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.all_samples and args.sample is not None:
        raise ValueError("Use either --all-samples or --sample, not both.")

    if args.all_samples:
        samples = [sample for sample in read_split_csv(args.split_csv) if sample.split == args.split]
        if not samples:
            raise ValueError(f"No samples found for split '{args.split}'.")
    else:
        samples = [select_sample(args.split, args.sample, args.split_csv)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(args.model).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    metrics = [run_sample_prediction(args, sample, model, device) for sample in samples]
    if len(metrics) > 1:
        values = np.asarray(metrics, dtype=np.float64)
        mean_mae, mean_mse, mean_rmse = values.mean(axis=0)
        print(f"Processed complete images: {len(metrics)}")
        print(f"Mean MAE_norm:  {mean_mae:.10f}")
        print(f"Mean MSE_norm:  {mean_mse:.10f}")
        print(f"Mean RMSE_norm: {mean_rmse:.10f}")


if __name__ == "__main__":
    main()
