# Dataset Indices

This directory stores split metadata and patch-count summaries generated from the Alicona dataset.

The split is by complete microscope image:

| Split | Full images | Sample groups |
|---|---:|---|
| Train | 120 | V1-V20 |
| Validation | 24 | V21-V24 |
| Test | 30 | V25-V29 |

Patch settings:

| Directory | Patch size | Stride |
|---|---:|---:|
| `patch64/` | 64 | 64 |
| `patch128/` | 128 | 128 |
| `patch256/` | 256 | 256 |

Full `patch_index.csv` files are generated locally by `scripts/prepare_dataset.py` and are not committed because they are large generated artifacts. The raw Alicona data folder `Versuchsreihe2_B91$prj/` is also not committed.
