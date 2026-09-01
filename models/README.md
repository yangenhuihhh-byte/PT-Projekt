# Model Checkpoints

The repository includes the five selected best-validation checkpoints used in the reported model and patch-resolution comparisons:

- `simple_cnn_patch128_stride128_ep100_best.pth`
- `small_unet_patch64_stride64_ep100_best.pth`
- `small_unet_patch128_stride128_ep100_best.pth`
- `small_unet_patch256_stride256_ep100_best.pth`
- `transunet_lite_patch128_stride128_ep100_best.pth`

The corresponding `*_last.pth` files are intentionally excluded because they duplicate each training run without being the validation-selected model used for final comparison.

This update package includes the trained model checkpoint files used in the post-midterm experiments.

The best checkpoints produced in the post-midterm experiments were:

| Model | Checkpoint |
|---|---|
| SimpleCNN | `simple_cnn_patch128_stride128_ep100_best.pth` |
| Small U-Net, 128 | `small_unet_patch128_stride128_ep100_best.pth` |
| TransUNet-lite | `transunet_lite_patch128_stride128_ep100_best.pth` |
| Small U-Net, 64 | `small_unet_patch64_stride64_ep100_best.pth` |
| Small U-Net, 256 | `small_unet_patch256_stride256_ep100_best.pth` |

The corresponding `*_last.pth` checkpoints are also included for completeness.
