from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_DIR / "Versuchsreihe2_B91$prj"
DEFAULT_DATASET_DIR = PROJECT_DIR / "dataset_alicona"
DEFAULT_SPLIT_CSV = DEFAULT_DATASET_DIR / "dataset_split.csv"
DEFAULT_PATCH_INDEX_CSV = DEFAULT_DATASET_DIR / "patch_index.csv"
ALICONA_INVALID_ABS_THRESHOLD = 1e6


@dataclass(frozen=True)
class AliconaSample:
    name: str
    path: Path
    group: int
    position: str
    variant: str
    split: str


@dataclass(frozen=True)
class PatchRecord:
    split: str
    sample: str
    top: int
    left: int
    patch_size: int
    height_min: float
    height_max: float


def parse_sample_name(name: str) -> tuple[int, str, str]:
    match = re.match(r"^V(\d+)_([OU])_([AIM])\$3D$", name)
    if not match:
        raise ValueError(f"Unexpected Alicona sample folder name: {name}")
    return int(match.group(1)), match.group(2), match.group(3)


def split_for_group(group: int) -> str:
    if 1 <= group <= 20:
        return "train"
    if 21 <= group <= 24:
        return "val"
    if 25 <= group <= 29:
        return "test"
    raise ValueError(f"No default split is defined for sample group V{group}")


def natural_sample_key(path: Path) -> tuple[int, str, str]:
    group, position, variant = parse_sample_name(path.name)
    return group, position, variant


def find_samples(raw_dir: Path = DEFAULT_RAW_DIR) -> list[AliconaSample]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    samples: list[AliconaSample] = []
    required = ("texture.bmp", "qualitymap.bmp", "info.xml", "icon.bmp", "dem.al3d")
    for folder in sorted((p for p in raw_dir.iterdir() if p.is_dir()), key=natural_sample_key):
        missing = [name for name in required if not (folder / name).exists()]
        if missing:
            raise FileNotFoundError(f"{folder.name} is missing files: {missing}")

        group, position, variant = parse_sample_name(folder.name)
        samples.append(
            AliconaSample(
                name=folder.name,
                path=folder,
                group=group,
                position=position,
                variant=variant,
                split=split_for_group(group),
            )
        )
    return samples


def write_split_csv(samples: Iterable[AliconaSample], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample", "group", "position", "variant", "split", "relative_path"])
        for sample in samples:
            writer.writerow(
                [
                    sample.name,
                    sample.group,
                    sample.position,
                    sample.variant,
                    sample.split,
                    sample.path.relative_to(PROJECT_DIR).as_posix(),
                ]
            )


def read_split_csv(split_csv: Path = DEFAULT_SPLIT_CSV) -> list[AliconaSample]:
    if not split_csv.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_csv}. Run scripts/prepare_dataset.py first."
        )

    samples: list[AliconaSample] = []
    with split_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            path = PROJECT_DIR / row["relative_path"]
            samples.append(
                AliconaSample(
                    name=row["sample"],
                    path=path,
                    group=int(row["group"]),
                    position=row["position"],
                    variant=row["variant"],
                    split=row["split"],
                )
            )
    return samples


def load_texture(texture_path: Path) -> np.ndarray:
    image = Image.open(texture_path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def read_al3d_header(height_path: Path, max_bytes: int = 4096) -> dict[str, str]:
    with height_path.open("rb") as handle:
        header_text = handle.read(max_bytes).decode("latin-1", errors="ignore")
    header_text = header_text.replace("\x00", " ")
    keys = [
        "Cols",
        "Rows",
        "DepthImageOffset",
        "InvalidPixelValue",
        "PixelSizeXMeter",
        "PixelSizeYMeter",
    ]
    header: dict[str, str] = {}
    for key in keys:
        match = re.search(rf"{key}\s+([^\r\n]+)", header_text)
        if match:
            header[key] = match.group(1).strip().split()[0]
    return header


def load_height(height_path: Path, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    expected = height * width
    header = read_al3d_header(height_path)
    depth_offset = int(float(header.get("DepthImageOffset", 0)))
    expected_bytes = expected * np.dtype(np.float32).itemsize
    if depth_offset <= 0 or depth_offset + expected_bytes > height_path.stat().st_size:
        depth_offset = height_path.stat().st_size - expected_bytes

    with height_path.open("rb") as handle:
        handle.seek(depth_offset)
        data = np.fromfile(handle, dtype=np.float32, count=expected)

    if data.size != expected:
        raise ValueError(
            f"{height_path} contains {data.size} height values, expected {expected}."
        )

    height_map = data[-expected:].reshape(height, width).astype(np.float32)
    invalid = ~np.isfinite(height_map)

    invalid_pixel_value = header.get("InvalidPixelValue")
    if invalid_pixel_value is not None:
        invalid_value = np.float32(float(invalid_pixel_value))
        invalid |= np.isclose(height_map, invalid_value, rtol=1e-6, atol=0.0)

    invalid |= np.abs(height_map) >= ALICONA_INVALID_ABS_THRESHOLD
    valid = ~invalid
    if not valid.any():
        raise ValueError(f"{height_path} does not contain finite height values.")

    fill_value = float(height_map[valid].min())
    return np.where(valid, height_map, fill_value).astype(np.float32)


def normalize_minmax(height_map: np.ndarray) -> tuple[np.ndarray, float, float]:
    h_min = float(np.min(height_map))
    h_max = float(np.max(height_map))
    denom = h_max - h_min
    if math.isclose(denom, 0.0, abs_tol=1e-12):
        return np.zeros_like(height_map, dtype=np.float32), h_min, h_max
    normalized = (height_map - h_min) / (denom + 1e-8)
    return normalized.astype(np.float32), h_min, h_max


def inverse_minmax(height_norm: np.ndarray, h_min: float, h_max: float) -> np.ndarray:
    return height_norm * (h_max - h_min) + h_min


def patch_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if patch_size > length:
        raise ValueError(f"Patch size {patch_size} is larger than image length {length}.")
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def iter_patch_records(
    samples: Iterable[AliconaSample],
    patch_size: int,
    stride: int,
) -> Iterable[PatchRecord]:
    for sample in samples:
        texture = load_texture(sample.path / "texture.bmp")
        height_raw = load_height(sample.path / "dem.al3d", texture.shape[:2])
        _, h_min, h_max = normalize_minmax(height_raw)
        rows = patch_starts(texture.shape[0], patch_size, stride)
        cols = patch_starts(texture.shape[1], patch_size, stride)
        for top in rows:
            for left in cols:
                yield PatchRecord(sample.split, sample.name, top, left, patch_size, h_min, h_max)


def write_patch_index(
    samples: Iterable[AliconaSample],
    output_csv: Path,
    patch_size: int,
    stride: int,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "sample", "top", "left", "patch_size", "height_min", "height_max"])
        for record in iter_patch_records(samples, patch_size, stride):
            writer.writerow(
                [
                    record.split,
                    record.sample,
                    record.top,
                    record.left,
                    record.patch_size,
                    record.height_min,
                    record.height_max,
                ]
            )


def read_patch_index(
    patch_index_csv: Path = DEFAULT_PATCH_INDEX_CSV,
    split: str | None = None,
    limit: int | None = None,
) -> list[PatchRecord]:
    if not patch_index_csv.exists():
        raise FileNotFoundError(
            f"Patch index not found: {patch_index_csv}. Run scripts/prepare_dataset.py first."
        )

    records: list[PatchRecord] = []
    with patch_index_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if split is not None and row["split"] != split:
                continue
            records.append(
                PatchRecord(
                    split=row["split"],
                    sample=row["sample"],
                    top=int(row["top"]),
                    left=int(row["left"]),
                    patch_size=int(row["patch_size"]),
                    height_min=float(row["height_min"]),
                    height_max=float(row["height_max"]),
                )
            )
            if limit is not None and len(records) >= limit:
                break
    return records


class AliconaPatchDataset(Dataset):
    def __init__(
        self,
        split: str,
        split_csv: Path = DEFAULT_SPLIT_CSV,
        patch_index_csv: Path = DEFAULT_PATCH_INDEX_CSV,
        limit: int | None = None,
    ) -> None:
        self.samples = {sample.name: sample for sample in read_split_csv(split_csv)}
        self.records = read_patch_index(patch_index_csv, split=split, limit=limit)

        if not self.records:
            raise ValueError(f"No patch records found for split '{split}'.")
        self._cached_sample_name: str | None = None
        self._cached_texture: np.ndarray | None = None
        self._cached_height_norm: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.records)

    def _load_sample_arrays(self, sample: AliconaSample) -> tuple[np.ndarray, np.ndarray]:
        if sample.name == self._cached_sample_name:
            assert self._cached_texture is not None
            assert self._cached_height_norm is not None
            return self._cached_texture, self._cached_height_norm

        texture = load_texture(sample.path / "texture.bmp")
        height_raw = load_height(sample.path / "dem.al3d", texture.shape[:2])
        height_norm, _, _ = normalize_minmax(height_raw)

        self._cached_sample_name = sample.name
        self._cached_texture = texture
        self._cached_height_norm = height_norm
        return texture, height_norm

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        sample = self.samples[record.sample]
        texture, height_norm = self._load_sample_arrays(sample)

        top = record.top
        left = record.left
        size = record.patch_size

        image_patch = texture[top : top + size, left : left + size, :]
        height_patch = height_norm[top : top + size, left : left + size]

        image_tensor = torch.from_numpy(image_patch).permute(2, 0, 1).float()
        height_tensor = torch.from_numpy(height_patch).unsqueeze(0).float()
        return image_tensor, height_tensor


def ensure_prepared(
    raw_dir: Path = DEFAULT_RAW_DIR,
    split_csv: Path = DEFAULT_SPLIT_CSV,
    patch_index_csv: Path = DEFAULT_PATCH_INDEX_CSV,
    patch_size: int = 128,
    stride: int = 128,
) -> None:
    if split_csv.exists() and patch_index_csv.exists():
        return

    samples = find_samples(raw_dir)
    write_split_csv(samples, split_csv)
    write_patch_index(samples, patch_index_csv, patch_size=patch_size, stride=stride)
