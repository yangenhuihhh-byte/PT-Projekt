from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from data_utils import (
    DEFAULT_DATASET_DIR,
    DEFAULT_PATCH_INDEX_CSV,
    DEFAULT_RAW_DIR,
    DEFAULT_SPLIT_CSV,
    find_samples,
    write_patch_index,
    write_split_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare split and patch-index metadata for Alicona data."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_csv = args.dataset_dir / DEFAULT_SPLIT_CSV.name
    patch_index_csv = args.dataset_dir / DEFAULT_PATCH_INDEX_CSV.name

    samples = find_samples(args.raw_dir)
    split_counts = Counter(sample.split for sample in samples)

    print(f"Found {len(samples)} complete Alicona samples in {args.raw_dir}")
    for split in ("train", "val", "test"):
        print(f"{split}: {split_counts[split]} full microscope images")

    write_split_csv(samples, split_csv)
    print(f"Saved split table: {split_csv}")

    write_patch_index(
        samples=samples,
        output_csv=patch_index_csv,
        patch_size=args.patch_size,
        stride=args.stride,
    )
    print(f"Saved patch index: {patch_index_csv}")
    print(f"Patch size: {args.patch_size} x {args.patch_size}")
    print(f"Training/evaluation patch stride: {args.stride}")
    print("Metrics computed from these patches are normalized, unitless errors.")


if __name__ == "__main__":
    main()
