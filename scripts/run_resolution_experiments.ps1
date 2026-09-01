$ErrorActionPreference = "Stop"

$configs = @(
    @{ PatchSize = 64;  Stride = 64;  BatchSize = 8; DatasetDir = "dataset_alicona_patch64"  },
    @{ PatchSize = 128; Stride = 128; BatchSize = 8; DatasetDir = "dataset_alicona"          },
    @{ PatchSize = 256; Stride = 256; BatchSize = 4; DatasetDir = "dataset_alicona_patch256" }
)

foreach ($config in $configs) {
    $patchSize = $config.PatchSize
    $stride = $config.Stride
    $batchSize = $config.BatchSize
    $datasetDir = $config.DatasetDir

    python scripts/prepare_dataset.py `
        --dataset-dir $datasetDir `
        --patch-size $patchSize `
        --stride $stride

    python scripts/train.py `
        --model small_unet `
        --epochs 100 `
        --patch-size $patchSize `
        --stride $stride `
        --batch-size $batchSize `
        --split-csv "$datasetDir/dataset_split.csv" `
        --patch-index-csv "$datasetDir/patch_index.csv"

    python scripts/evaluate.py `
        --model small_unet `
        --checkpoint "models/small_unet_patch${patchSize}_stride${stride}_ep100_best.pth" `
        --split test `
        --patch-size $patchSize `
        --stride $stride `
        --split-csv "$datasetDir/dataset_split.csv" `
        --patch-index-csv "$datasetDir/patch_index.csv"
}
