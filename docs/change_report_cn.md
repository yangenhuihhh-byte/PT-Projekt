# 项目整改报告：针对老师反馈的修改说明

## 1. 修改背景

老师在中期汇报后指出，原报告中只展示了一个单一显微图片的预测结果，缺少对模型泛化能力、训练/测试数据划分、误差指标含义、不同模型定量对比、完整显微图片预测以及块状伪影优化的系统说明。根据这些意见，我们对项目的代码结构、数据读取流程、实验脚本和报告内容进行了整理和修改。

本次修改的核心目标是：把项目从“展示一个预测示例”改为“有明确数据划分、有可解释误差指标、有完整测试集泛化评估、有模型对比、有完整显微图预测流程”的实验项目。

## 2. 总体修改摘要

| 老师建议 | 我们做的修改 | 相关文件或结果 | 当前状态 |
|---|---|---|---|
| 报告中需要体现泛化分析，不应只出现一个图片结果 | 按完整显微图片划分训练集、验证集、测试集，并在文档中加入 split 表和 patch 数量 | `README.md`, `docs/report_addendum.md`, `docs/supervisor_feedback_revision_plan.md`, `dataset_alicona/dataset_split.csv`, `dataset_alicona/patch_index.csv` | 已完成数据划分说明 |
| 标明训练数据集、测试数据集等 | 明确写出 Training / Validation / Test 的样本范围、完整图像数量和用途 | 同上 | 已完成 |
| 解释 MAE、RMSE、MSE 是否正则化，为什么小于 1 | 明确说明误差是在 min-max 归一化高度图上计算的，是无量纲误差，不是物理微米误差 | `README.md`, `docs/report_addendum.md` | 已完成说明 |
| 不同模型比较要展示误差数据 | 增加统一评估脚本，将不同模型在同一 test split 上的 MAE/MSE/RMSE 写入表格 | `scripts/evaluate.py`, `results/model_comparison.csv` | 脚本已完成，最终表格需重训后填数 |
| 用已有模型对完整显微图片进行预测，先切片再结合 | 增加完整图片 sliding-window 预测和 patch stitching 流程 | `scripts/predict_full_image.py`, `results/full_image_predictions/` | 已实现 |
| 优化预测结果的离散性和马赛克感 | 增加 overlap stitching 和 Gaussian kernel blending，对比 none / uniform / gaussian 三种拼接方式 | `scripts/predict_full_image.py`, `scripts/run_revision_experiments.ps1` | 已实现，需补最终实验图和表 |
| 尝试不同分辨率 | 增加 64、128、256 patch size 对比实验脚本 | `scripts/run_resolution_experiments.ps1` | 脚本已准备，需运行实验 |
| 尝试其它模型并参考论文 | 增加 SimpleCNN、Small U-Net、TransUNet-lite 的统一模型接口，并在文档中加入 U-Net、TransUNet、Swin-Unet、DPT 参考 | `scripts/models.py`, `scripts/train.py`, `docs/report_addendum.md` | 模型接口已完成，需重训比较 |

## 3. 数据集与泛化分析的修改

原来的展示方式容易给人一种“只在一张图片上测试”的印象，无法证明模型对未见过显微图片的泛化能力。因此我们修改为先按完整显微图片划分数据集，再对每个 split 内的图片切 patch。

当前数据集包含 174 张完整 Alicona 显微样本，每个样本包含：

```text
texture.bmp
qualitymap.bmp
info.xml
icon.bmp
dem.al3d
```

修改后的数据划分如下：

| Split | 样本组 | 完整显微图片数量 | 128/128 patch 数量 | 用途 |
|---|---|---:|---:|---|
| Training | V1-V20 | 120 | 30,720 | 训练模型参数 |
| Validation | V21-V24 | 24 | 6,144 | 选择 epoch 和超参数 |
| Test | V25-V29 | 30 | 7,680 | 最终泛化评估 |

这样做的原因是：同一张显微图片中相邻 patch 高度相关。如果先把所有 patch 混在一起再随机划分，训练集和测试集可能会包含来自同一张显微图的相邻 patch，导致数据泄漏，测试误差会被低估。现在按完整图片划分，可以更真实地评估模型对新显微图片的预测能力。

## 4. MAE、MSE、RMSE 解释的修改

针对老师提出的“误差是否正则化、为什么小于 1”的问题，我们在文档中明确说明：当前报告中的 MAE、MSE 和 RMSE 是在 min-max 归一化后的高度图上计算的。

归一化公式为：

```text
h_norm = (h - h_min) / (h_max - h_min)
```

误差定义为：

```text
MAE_norm  = mean(|h_pred_norm - h_true_norm|)
MSE_norm  = mean((h_pred_norm - h_true_norm)^2)
RMSE_norm = sqrt(MSE_norm)
```

因此，这些误差是无量纲的 normalized error，不是以微米为单位的物理误差。因为目标高度图被映射到接近 `[0, 1]` 的范围内，所以误差小于 1 是正常的。但如果模型预测值超出归一化目标范围，误差也可能大于 1。

在完整显微图片预测脚本中，我们也加入了反归一化后的 raw height-unit error：

```text
h_pred_raw = h_pred_norm * (h_max - h_min) + h_min
```

报告中建议主要展示 `MAE_norm / MSE_norm / RMSE_norm`，同时说明它们是归一化误差。如果需要写物理单位误差，应先确认 Alicona 元数据中的高度单位换算关系。

## 5. Alicona 高度文件读取流程的修正

我们发现旧代码把 `.al3d` 文件当成无文件头的 float32 数组直接读取，这是不准确的。实际 `.al3d` 文件头中包含高度数据偏移量和无效像素值，例如：

```text
DepthImageOffset    845
InvalidPixelValue   3.000000028082e+15
Rows                2040
Cols                2040
```

旧流程的问题是：直接 `np.fromfile(..., dtype=np.float32)` 再取最后 `Rows * Cols` 个 float，会因为 header 长度不是 4 字节整数倍而造成 float 对齐错误，同时还会把 Alicona 的 invalid sentinel value 混入高度范围。这样会导致高度最大值异常巨大，进而影响归一化和误差解释。

我们在 `scripts/data_utils.py` 中做了如下修改：

- 读取 `.al3d` header。
- 根据 `DepthImageOffset` 精确定位高度数据开始位置。
- 根据 `Rows` 和 `Cols` 读取完整高度图。
- 过滤 `InvalidPixelValue` 和异常巨大高度值。
- 再进行 min-max 归一化。

这个修改非常关键，因为它改变了数据预处理的正确性。因此，旧的 `unet_patch128_ep100.pth` 和旧的 `results/model_comparison.csv` 中的结果只能作为历史参考，最终报告应该使用修复数据读取流程后重新训练得到的 checkpoint 和测试指标。

## 6. 模型训练与定量比较的修改

为了回应“既然做了不同模型比较，就要把误差数据表达出来”，我们把模型和评估流程整理成统一接口。

目前支持的模型包括：

| 模型 | 作用 |
|---|---|
| SimpleCNN | 简单 CNN baseline，用来证明 U-Net 结构是否真的带来提升 |
| Small U-Net | 当前主 baseline，适合 dense prediction，skip connections 有利于保留局部细节 |
| TransUNet-lite | 轻量 transformer-style 对比模型，在 CNN encoder 后加入 transformer bottleneck 建模全局上下文 |

训练命令示例：

```powershell
python scripts/train.py --model simple_cnn --epochs 100 --patch-size 128 --stride 128
python scripts/train.py --model small_unet --epochs 100 --patch-size 128 --stride 128
python scripts/train.py --model transunet_lite --epochs 100 --patch-size 128 --stride 128 --batch-size 4
```

测试命令示例：

```powershell
python scripts/evaluate.py --model simple_cnn --checkpoint models/simple_cnn_patch128_stride128_ep100_best.pth --split test
python scripts/evaluate.py --model small_unet --checkpoint models/small_unet_patch128_stride128_ep100_best.pth --split test
python scripts/evaluate.py --model transunet_lite --checkpoint models/transunet_lite_patch128_stride128_ep100_best.pth --split test
```

`scripts/evaluate.py` 会在完整 test split 上计算：

```text
MAE_norm
MSE_norm
RMSE_norm
```

并将结果追加到：

```text
results/model_comparison.csv
```

同时我们修改了脚本逻辑：如果使用 `--limit-patches` 做快速 smoke test，结果不会写入正式的 `model_comparison.csv`，避免把不完整测试误认为最终泛化结果。

最终报告中建议使用如下表格：

| Model | Patch size | Stride | Epochs | Test MAE_norm | Test MSE_norm | Test RMSE_norm |
|---|---:|---:|---:|---:|---:|---:|
| SimpleCNN | 128 | 128 | 100 | 重训后填写 | 重训后填写 | 重训后填写 |
| Small U-Net | 128 | 128 | 100 | 重训后填写 | 重训后填写 | 重训后填写 |
| TransUNet-lite | 128 | 128 | 100 | 重训后填写 | 重训后填写 | 重训后填写 |

## 7. 完整显微图片预测流程的修改

原汇报中只展示了单个 patch 或局部预测结果。根据老师建议，我们增加了完整显微图片预测流程，即先对完整 `texture.bmp` 做滑窗切片，再对每个 patch 预测高度，最后把 patch 结果拼回完整高度图。

完整流程如下：

```text
complete texture.bmp
-> sliding-window patch extraction
-> patch-wise height prediction
-> weighted stitching
-> complete predicted height map
-> complete-image MAE/MSE/RMSE
```

对应脚本为：

```text
scripts/predict_full_image.py
```

单张测试图预测命令示例：

```powershell
python scripts/predict_full_image.py --model small_unet --checkpoint models/small_unet_patch128_stride128_ep100_best.pth --split test --sample 'V25_O_A$3D' --patch-size 128 --stride 64 --blend gaussian
```

对完整 test split 中所有显微图片进行预测：

```powershell
python scripts/predict_full_image.py --model small_unet --checkpoint models/small_unet_patch128_stride128_ep100_best.pth --split test --all-samples --patch-size 128 --stride 64 --blend gaussian
```

输出结果包括：

```text
input_texture.png
true_height_norm.png
predicted_height_norm.png
absolute_error_norm.png
summary.png
predicted_height_norm.npy
true_height_norm.npy
absolute_error_norm.npy
full_image_metrics.csv
```

这样报告中既可以展示一张完整显微图片的预测可视化，也可以汇报所有测试图上的平均完整图像误差。

## 8. 马赛克感和离散性优化的修改

老师提到预测结果有离散感，像马赛克。这个问题主要来自 patch-based prediction：每个 patch 独立预测，如果直接拼接，相邻 patch 边界处容易出现不连续。

我们在完整图片预测脚本中加入了三种 stitching 方式：

| 方法 | Patch size | Stride | Blend | 目的 |
|---|---:|---:|---|---|
| No overlap | 128 | 128 | none | 作为最基础拼接方式 |
| Uniform overlap | 128 | 64 | uniform | 重叠区域取平均，减弱边界突变 |
| Gaussian overlap | 128 | 64 | gaussian | patch 中心权重大、边缘权重小，进一步减少块状伪影 |

Gaussian stitching 的表达式为：

```text
H_pred = sum(w_i * P_i) / sum(w_i)
```

其中 `P_i` 是某个 patch 的预测结果，`w_i` 是该 patch 对应的 Gaussian weight。patch 中心区域权重更高，边缘权重更低，因此可以降低 patch 边界带来的硬切换。

对比命令：

```powershell
python scripts/predict_full_image.py --model small_unet --checkpoint models/small_unet_patch128_stride128_ep100_best.pth --split test --sample 'V25_O_A$3D' --patch-size 128 --stride 128 --blend none
python scripts/predict_full_image.py --model small_unet --checkpoint models/small_unet_patch128_stride128_ep100_best.pth --split test --sample 'V25_O_A$3D' --patch-size 128 --stride 64 --blend uniform
python scripts/predict_full_image.py --model small_unet --checkpoint models/small_unet_patch128_stride128_ep100_best.pth --split test --sample 'V25_O_A$3D' --patch-size 128 --stride 64 --blend gaussian
```

最终报告中建议同时展示三种方法的误差表和预测图，说明 overlap 和 Gaussian blending 是否在视觉上减少块状边界，以及误差是否同步改善。

## 9. 不同分辨率实验的修改

为了回应“尝试不同分辨率”的建议，我们增加了 patch size 对比实验脚本：

```text
scripts/run_resolution_experiments.ps1
```

计划比较三种分辨率：

| Experiment | Patch size | Stride | 预期影响 |
|---|---:|---:|---|
| Fine patches | 64 | 64 | 捕捉更多局部细节，但训练 patch 数量增加 |
| Current baseline | 128 | 128 | 当前主设置，平衡局部细节和计算成本 |
| Large patches | 256 | 256 | 单个 patch 上下文更大，但训练样本数量减少 |

每个分辨率都应使用同一个模型，例如 Small U-Net，并在相同 test split 上评估。最终报告中可以比较不同 patch size 对 MAE/RMSE 和预测平滑性的影响。

## 10. Transformer 相关模型和论文对比的修改

项目题目中提到 transformer-based depth estimation，因此报告中不应只把 U-Net 当作最终模型。我们增加了 `TransUNet-lite` 作为轻量 transformer-style 对比模型。

报告中建议引用以下论文：

| 方法 | 论文 | 与本项目的关系 |
|---|---|---|
| U-Net | Ronneberger et al., 2015, U-Net: Convolutional Networks for Biomedical Image Segmentation | 当前强 baseline，适合像素级 dense prediction |
| TransUNet | Chen et al., 2021, TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation | 说明 CNN + Transformer + U-Net decoder 的思路 |
| Swin-Unet | Cao et al., 2021, Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation | 说明 U-shaped transformer 架构 |
| Dense Prediction Transformer | Ranftl et al., 2021, Vision Transformers for Dense Prediction | 与深度/高度图这类 dense prediction 任务直接相关 |

在报告中应说明：Small U-Net 的优势是结构简单、训练稳定、skip connections 能保留局部纹理细节；TransUNet-lite 的优势是 transformer bottleneck 能建模更大范围上下文。最终需要用同一 test split 的 MAE/RMSE 来判断 TransUNet-lite 相比当前最好 U-Net 是更好、更差，还是效果接近但计算成本更高。

## 11. 报告和 PPT 内容结构的修改建议

为了让老师的每一条意见都能在汇报中被看到，建议把原来的 9 页中期 PPT 扩展为 11 到 12 页：

1. Titel
2. Motivation und Ziel
3. Datengrundlage und Split
4. Alicona-Parsing und Normalisierung
5. Methodik: Patch-basiertes Lernen
6. Modellarchitekturen
7. Quantitativer Modellvergleich
8. Generalisierung auf dem Testdatensatz
9. Vollbildvorhersage durch Stitching
10. Reduktion von Blockartefakten
11. Transformer-basierter Vergleich
12. Fazit und Ausblick

其中第 3、4、7、8、9、10、11 页分别对应老师提出的核心问题：数据划分、误差解释、模型比较、泛化结果、完整图片预测、马赛克优化和 transformer 论文对比。

## 12. 当前已完成内容与后续待完成内容

已经完成的部分：

| 内容 | 状态 |
|---|---|
| 重新梳理项目 README，加入老师反馈对应修改 | 已完成 |
| 增加中文整改方案文档 | 已完成 |
| 增加英文 report addendum | 已完成 |
| 修正 `.al3d` header offset 和 invalid pixel 读取问题 | 已完成 |
| 重新生成 train/val/test split 和 patch index | 已完成 |
| 统一模型接口，支持 SimpleCNN、Small U-Net、TransUNet-lite | 已完成 |
| 增加完整 test split 的评估脚本 | 已完成 |
| 增加完整显微图片预测和 stitching 脚本 | 已完成 |
| 增加 Gaussian blending 和 overlap 对比 | 已完成 |
| 增加不同 patch size 分辨率实验脚本 | 已完成 |

仍需补充的部分：

| 内容 | 原因 |
|---|---|
| 用修正后的 `.al3d` loader 重新训练 SimpleCNN、Small U-Net、TransUNet-lite | 旧 checkpoint 是旧数据读取流程训练的，不能作为最终结论 |
| 重新运行完整 test split 评估并填写 `model_comparison.csv` | 最终报告需要定量模型对比 |
| 运行完整图像预测，生成新的 `full_image_metrics.csv` 和可视化图 | 最终报告需要完整显微图片预测结果 |
| 跑 none / uniform / gaussian stitching 对比 | 用来回答马赛克感是否被优化 |
| 跑 64 / 128 / 256 patch size 对比实验 | 用来回答不同分辨率是否影响结果 |
| 更新 PPT，把新增实验表格和完整图像结果放进去 | 让老师的建议在汇报中直观看到 |

## 13. 可直接写进最终报告的总结段落

根据老师反馈，我们对原项目进行了系统整改。首先，数据集不再以随机 patch 方式解释，而是按完整显微图片划分为训练集、验证集和测试集，从而避免同一显微图片中的相邻 patch 同时出现在训练和测试中造成数据泄漏。其次，我们修正了 Alicona `.al3d` 高度文件的读取流程，依据文件头中的 `DepthImageOffset` 读取真实高度数据，并过滤 `InvalidPixelValue`，保证归一化和误差计算建立在正确高度图上。第三，我们明确说明 MAE、MSE 和 RMSE 均是在 min-max 归一化高度图上计算的无量纲误差，因此数值小于 1 是合理的。第四，我们增加了统一的模型训练和测试流程，用相同 test split 对 SimpleCNN、Small U-Net 和 TransUNet-lite 进行定量比较。最后，我们实现了完整显微图片的 patch-wise prediction 与 stitching 流程，并加入 overlap 和 Gaussian kernel blending，用于减少直接拼接造成的块状伪影。

需要注意的是，旧 checkpoint 和旧误差结果来自修正 `.al3d` 读取流程之前，只能作为历史结果参考。最终报告应使用修正后重新训练的模型结果，并在同一测试集上报告 MAE_norm、MSE_norm 和 RMSE_norm。
