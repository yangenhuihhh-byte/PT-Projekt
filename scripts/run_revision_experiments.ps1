$ErrorActionPreference = "Stop"

python scripts/prepare_dataset.py --patch-size 128 --stride 128

python scripts/train.py --model simple_cnn --epochs 100 --patch-size 128 --stride 128 --batch-size 8
python scripts/train.py --model small_unet --epochs 100 --patch-size 128 --stride 128 --batch-size 8
python scripts/train.py --model transunet_lite --epochs 100 --patch-size 128 --stride 128 --batch-size 4

python scripts/evaluate.py --model simple_cnn --checkpoint models/simple_cnn_patch128_stride128_ep100_best.pth --split test
python scripts/evaluate.py --model small_unet --checkpoint models/small_unet_patch128_stride128_ep100_best.pth --split test
python scripts/evaluate.py --model transunet_lite --checkpoint models/transunet_lite_patch128_stride128_ep100_best.pth --split test

python scripts/predict_full_image.py --model small_unet --checkpoint models/small_unet_patch128_stride128_ep100_best.pth --split test --patch-size 128 --stride 128 --blend none
python scripts/predict_full_image.py --model small_unet --checkpoint models/small_unet_patch128_stride128_ep100_best.pth --split test --patch-size 128 --stride 64 --blend uniform
python scripts/predict_full_image.py --model small_unet --checkpoint models/small_unet_patch128_stride128_ep100_best.pth --split test --patch-size 128 --stride 64 --blend gaussian
python scripts/predict_full_image.py --model small_unet --checkpoint models/small_unet_patch128_stride128_ep100_best.pth --split test --all-samples --patch-size 128 --stride 64 --blend gaussian
