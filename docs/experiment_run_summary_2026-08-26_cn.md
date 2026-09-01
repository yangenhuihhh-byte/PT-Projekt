# 2026-08-26 扩展实验运行结果小结

## 1. 本次完成的实验

本次不再只做最小流程，而是围绕老师给出的建议补做了一组较完整的实验：

```text
重新检查 Alicona 高度文件读取
-> 按完整显微图片划分 train / validation / test
-> 训练 3 个模型做模型对比
-> 训练 64 / 128 / 256 三种 patch 分辨率
-> 在 test split 上计算 MAE / MSE / RMSE
-> 对 30 张完整测试显微图做 sliding-window 预测
-> 比较 none / uniform overlap / gaussian overlap 三种 stitching
-> 输出 summary.png 可视化，用于报告和 PPT
```

本次结果均使用修正后的 `.al3d` 高度文件读取流程。2026-08-11 的旧 checkpoint 结果保留在 CSV 中作为历史记录，但不作为正式结论使用。

## 2. 训练集、验证集、测试集

数据来自 `Versuchsreihe2_B91$prj`，共有 174 个完整 Alicona 样本。每个样本包含：

```text
texture.bmp      2D 显微纹理图像
dem.al3d         3D 高度图
qualitymap.bmp   质量图
info.xml         元数据
```

划分方式按“完整显微图片”划分，而不是把同一张图的 patch 同时放入训练和测试：

| Split | 完整显微图片数量 | 样本范围 | 用途 |
|---|---:|---|---|
| Train | 120 | V1-V20 | 模型训练 |
| Validation | 24 | V21-V24 | 选择 best checkpoint |
| Test | 30 | V25-V29 | 最终泛化评估 |

不同 patch 分辨率下的 patch 数量如下：

| Patch size | Stride | Train patches | Validation patches | Test patches |
|---:|---:|---:|---:|---:|
| 64 | 64 | 122,880 | 24,576 | 30,720 |
| 128 | 128 | 30,720 | 6,144 | 7,680 |
| 256 | 256 | 7,680 | 1,536 | 1,920 |

## 3. 指标说明

报告中的 `MAE_norm`、`MSE_norm`、`RMSE_norm` 都是在 min-max normalized height map 上计算的。

也就是说，每一张高度图先被归一化到 0 到 1：

```text
height_norm = (height_raw - min(height_raw)) / (max(height_raw) - min(height_raw))
```

因此这些误差是无量纲数值，小于 1 是正常的。它们不是直接的微米误差，也不是因为模型特别“完美”，而是因为高度被缩放到了 0-1 区间。

完整图像预测脚本同时输出 raw height-unit error，例如：

```text
MAE_raw_height_units
MSE_raw_height_units
RMSE_raw_height_units
```

raw error 是把预测结果反归一化回原始高度范围后计算的误差。由于每张显微图的高度范围不同，正式模型比较建议主要使用 normalized error；如果要解释实际物理误差，可以附上 raw error。

## 4. 模型对比实验

在 128x128 patch、stride 128 的同一数据集上训练并测试了 3 个模型：

| Model | 参考思想 | Test patches | MAE_norm | MSE_norm | RMSE_norm |
|---|---|---:|---:|---:|---:|
| SimpleCNN | 基础 CNN baseline | 7,680 | 0.0966449065 | 0.0161155819 | 0.1269471617 |
| TransUNet-lite | TransUNet-style CNN + Transformer encoder | 7,680 | 0.0899855863 | 0.0149214596 | 0.1221534265 |
| Small U-Net | U-Net encoder-decoder | 7,680 | 0.0852922367 | 0.0128995686 | 0.1135762678 |

结论：

1. Small U-Net 是当前 128x128 设置下最好的模型。
2. TransUNet-lite 明显好于 SimpleCNN，说明加入全局建模思想有帮助。
3. TransUNet-lite 没有超过 Small U-Net，可能原因是本项目实现的是轻量化版本，没有 ImageNet/医学数据预训练，数据量也较小，并且本任务是高度回归而不是论文中的医学图像分割。

论文参考：

| 模型 | 参考论文 |
|---|---|
| U-Net | Ronneberger, Fischer, Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015, https://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a/ |
| TransUNet | Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation", arXiv:2102.04306, https://doi.org/10.48550/arxiv.2102.04306 |

## 5. 分辨率对比实验

为了回应老师关于“尝试不同分辨率”的建议，使用 Small U-Net 分别训练了 64、128、256 三种 patch size。

| Model | Patch size | Stride | Test patches | MAE_norm | MSE_norm | RMSE_norm |
|---|---:|---:|---:|---:|---:|---:|
| Small U-Net | 64 | 64 | 30,720 | 0.0872558418 | 0.0133066573 | 0.1153544853 |
| Small U-Net | 128 | 128 | 7,680 | 0.0852922367 | 0.0128995686 | 0.1135762678 |
| Small U-Net | 256 | 256 | 1,920 | 0.0861126906 | 0.0128414407 | 0.1133200807 |

结论：

1. 128x128 在 test MAE 上最好。
2. 256x256 的 MSE 和 RMSE 略好于 128x128，但 MAE 略差，整体非常接近。
3. 64x64 训练到后期才接近其他分辨率，但最终 test error 仍略高。可能原因是小 patch 的局部细节更多，但缺少较大范围的表面形貌上下文。

## 6. 完整显微图片预测

为了回应“尝试使用已有模型对完整显微图片进行预测，先切片再结合”的建议，已经对 30 张 test split 完整显微图全部进行了 sliding-window 预测。

完整图像预测流程：

```text
读取完整 texture.bmp
-> 按 patch size 和 stride 切片
-> 每个 patch 输入模型预测高度
-> 将所有 patch 放回原始位置
-> 对重叠区域做 none / uniform / gaussian 融合
-> 输出 predicted_height.npy、error_norm.npy、summary.png
```

### 6.1 Stitching 方法对比

使用同一个 Small U-Net 128x128 checkpoint，在 30 张测试图上比较三种 stitching 设置：

| Setting | Patch size | Stride | Blend | 完整测试图数量 | Mean MAE_norm | Mean MSE_norm | Mean RMSE_norm |
|---|---:|---:|---|---:|---:|---:|---:|
| No overlap | 128 | 128 | none | 30 | 0.0852090585 | 0.0128746741 | 0.1124974950 |
| Uniform overlap | 128 | 64 | uniform | 30 | 0.0848604908 | 0.0127769265 | 0.1120548095 |
| Gaussian overlap | 128 | 64 | gaussian | 30 | 0.0846515231 | 0.0127101013 | 0.1117579066 |

结论：

1. 加入 overlap 后，完整图像平均误差略有下降。
2. Gaussian overlap 是三种方式中最好的。
3. Gaussian kernel 会降低 patch 边缘权重，让中心区域贡献更多，因此可以减弱 patch 边界造成的块状感。

### 6.2 完整图像级分辨率对比

使用各自分辨率训练出的 Small U-Net checkpoint，在 30 张 test 完整显微图上做 Gaussian stitching：

| Model | Patch size | Prediction stride | Blend | 完整测试图数量 | Mean MAE_norm | Mean MSE_norm | Mean RMSE_norm |
|---|---:|---:|---|---:|---:|---:|---:|
| Small U-Net | 64 | 32 | gaussian | 30 | 0.0862601604 | 0.0129628870 | 0.1129844417 |
| Small U-Net | 128 | 64 | gaussian | 30 | 0.0846515231 | 0.0127101013 | 0.1117579066 |
| Small U-Net | 256 | 128 | gaussian | 30 | 0.0857692685 | 0.0127286583 | 0.1119506734 |

结论：

1. 完整图像级预测中，128x128 + stride64 + Gaussian overlap 是当前最好设置。
2. 256x256 与 128x128 很接近，但 MAE 略高。
3. 64x64 + stride32 虽然 overlap 最密，但平均误差最高，说明更高局部分辨率不一定带来更好泛化。

## 7. 单张完整图像示例

典型测试图：

```text
V25_O_A$3D
```

三种 128x128 stitching 结果：

| Setting | Patch size | Stride | Blend | MAE_norm | MSE_norm | RMSE_norm |
|---|---:|---:|---|---:|---:|---:|
| No overlap | 128 | 128 | none | 0.0647623762 | 0.0096667781 | 0.0983197750 |
| Uniform overlap | 128 | 64 | uniform | 0.0641800389 | 0.0095776170 | 0.0978653005 |
| Gaussian overlap | 128 | 64 | gaussian | 0.0637224913 | 0.0094542708 | 0.0972330747 |

建议 PPT 展示下面这张图：

```text
results/full_image_predictions/V25_O_A$3D_small_unet_p128_s64_gaussian/summary.png
```

这张图包含：

```text
Input texture
True height
Predicted height
Absolute error
```

如果要展示分辨率对比，可以再加入：

```text
results/full_image_predictions/V25_O_A$3D_small_unet_p64_s32_gaussian/summary.png
results/full_image_predictions/V25_O_A$3D_small_unet_p128_s64_gaussian/summary.png
results/full_image_predictions/V25_O_A$3D_small_unet_p256_s128_gaussian/summary.png
```

## 8. 对老师建议的逐条回应

| 老师建议 | 本次完成情况 | 可以写入报告的结论 |
|---|---|---|
| 泛化分析结果要体现，不能只放单张图 | 已在 30 张 test 完整显微图和 test patches 上计算平均误差 | 报告中加入 test split 平均 MAE/MSE/RMSE，证明结果不是单图展示 |
| 标明训练数据集、测试数据集 | 已明确 train/validation/test 按完整图片划分 | Train 120 张，Validation 24 张，Test 30 张 |
| 说明 MAE/RMSE/MSE 为什么小于 1 | 已解释 normalized height map | 指标是在 0-1 高度图上计算，因此小于 1 是正常的 |
| 既然比较不同模型，也要表达误差数据 | 已训练 SimpleCNN、Small U-Net、TransUNet-lite 并给出 test metrics | Small U-Net 当前最好，TransUNet-lite 好于 SimpleCNN 但未超过 U-Net |
| 用已有模型预测完整显微图片 | 已对 30 张 test 完整图做 sliding-window + stitching | 可展示完整图像级预测和平均误差 |
| 优化离散性和马赛克感 | 已比较 none、uniform overlap、Gaussian overlap | Gaussian overlap 平均误差最低，也更适合减弱 patch 边界 |
| 尝试不同分辨率 | 已训练并评估 64、128、256 patch size | 128x128 当前综合最佳，256x256 接近，64x64 略差 |
| 尝试其它模型并分析差别 | 已加入 TransUNet-lite 对照 | 当前轻量 TransUNet-style 未超过 Small U-Net，可能与数据量、预训练和任务类型有关 |

## 9. 结果文件

关键 CSV：

```text
results/model_comparison.csv
results/full_image_predictions/full_image_metrics.csv
results/training_log.csv
```

关键 checkpoint：

```text
models/small_unet_patch128_stride128_ep100_best.pth
models/simple_cnn_patch128_stride128_ep100_best.pth
models/transunet_lite_patch128_stride128_ep100_best.pth
models/small_unet_patch64_stride64_ep100_best.pth
models/small_unet_patch256_stride256_ep100_best.pth
```

关键可视化目录：

```text
results/full_image_predictions/
```

## 10. 当前最佳结论

当前最适合作为最终报告主结果的配置是：

```text
Model: Small U-Net
Training patch size: 128
Training stride: 128
Full-image prediction patch size: 128
Full-image prediction stride: 64
Blend: Gaussian overlap
```

对应结果：

| Evaluation level | MAE_norm | MSE_norm | RMSE_norm |
|---|---:|---:|---:|
| Patch-level test split | 0.0852922367 | 0.0128995686 | 0.1135762678 |
| Full-image test split, 30 images | 0.0846515231 | 0.0127101013 | 0.1117579066 |

下一步建议把这些表格和 `summary.png` 放入最终 Word 报告和 PPT 中，并在方法章节说明 normalized error 与 raw height-unit error 的区别。
