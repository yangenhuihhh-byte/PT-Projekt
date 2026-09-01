# Results

This directory contains the report-ready result tables and selected figures from the post-midterm experiments.

Important files:

| File | Meaning |
|---|---|
| `model_comparison.csv` | Patch-level test metrics for SimpleCNN, Small U-Net and TransUNet-lite |
| `training_log.csv` | Training and validation logs |
| `full_image_metrics.csv` | Complete-image prediction metrics |
| `report_figures/` | Selected figures used in the current-stage report |

Key figures in `report_figures/` include the Gaussian full-image result, the stitching-mode crop comparison, and the 64/128/256 patch-resolution examples.

Only selected PNG figures are committed. Full per-sample prediction arrays (`*.npy`) are intentionally excluded.
