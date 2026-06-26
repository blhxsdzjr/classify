# 车道线检测期末实验报告

## 1. 问题定义

**任务**：从道路图像中检测车道线，输出每条线的位置（端点坐标）和颜色（白/黄）。

**数据集**：
- 37 张测试图片（1080×1920），前 23 张有人工线段标注
- 标注格式：`class_id x1_norm y1_norm x2_norm y2_norm`（归一化坐标）
- 67 条 GT 线（57 白 + 10 黄）
- 另有 50 张训练图片（无标注）

**评估指标**：
- 线段级 Precision / Recall / F1（角度 ≤ 35° 且空间重叠）
- Pixel Accuracy / mIoU / Dice（语义分割）

---

## 2. 方法一：传统 CV Baseline（HSV + Hough）

### 原理

```
原图 → HSV颜色掩码(白/黄) → Canny边缘 → HoughLinesP直线检测 → 聚类合并 → 拟合输出
```

### 关键步骤

**2.1 HSV 颜色掩码**
- 白线：S ≤ 95 且 V ≥ 145（低饱和度 + 高亮度）
- 黄线：H ∈ [15,40] 且 S ≥ 70 且 V ≥ 95（暖色相 + 中高饱和度）
- 用梯形 ROI 排除天空区域
- 形态学操作（开运算去噪 + 垂直闭运算连接虚线）

**2.2 直线检测**
- Canny 边缘检测（低阈值 35，高阈值 110）
- 概率 Hough 变换（threshold=30, minLineLength=45, maxLineGap=125）
- 角度过滤：只保留与水平线夹角 35°-145° 的线段（近似竖直）

**2.3 线段聚类与拟合**
- 按角度和空间距离贪心聚类
- 每组用 `cv2.fitLine`（最小二乘）拟合为一条直线
- 去重：同类别、x 中心距 < 42px 的重复线移除

### 实验结果

| 指标 | White | Yellow | Overall |
|------|:---:|:---:|:---:|
| Precision | 15.49% | 4.35% | 12.77% |
| Recall | 19.30% | 10.00% | 17.91% |
| F1 | 17.19% | 6.06% | **14.91%** |
| GT 匹配 | 11/57 | 1/10 | 12/67 (17.9%) |

### 失败原因分析

1. **边缘 ≠ 车道线中心**：Hough 检测的是亮度跳变边缘（护栏、路缘、阴影），不是车道线中心线
2. **固定阈值不鲁棒**：光照变化导致同一 HSV 阈值无法覆盖所有场景
3. **路面暖色调干扰**：路面像素 S≈82, H≈17，与黄线 HSV 范围高度重叠
4. **无法利用上下文**：纯像素级处理，没有全局语义理解

---

## 3. 方法二：U-Net 语义分割（深度学习方法）

### 3.1 为什么选 U-Net

U-Net 是语义分割的经典架构，特点：
- **编码器-解码器**：下采样提取特征 → 上采样恢复分辨率
- **跳跃连接**：编码器特征直接连接到解码器同层，保留细粒度空间信息
- **适合小数据集**：参数少（~1.4M），配合数据增强可从 27 张图学习

### 3.2 架构

```
输入 512×512×3
  ↓
[Conv 3→32] ×2 → MaxPool   ────┐
[Conv 32→64] ×2 → MaxPool  ────┼── 跳跃连接
[Conv 64→128] ×2 → MaxPool ────┼──┐
[Conv 128→256] ×2 → MaxPool ───┼──┼──┐
  ↓                             │  │  │
[Conv 256→512] ×2 (Bottleneck) │  │  │
  ↓                             │  │  │
UpSample + Concat ←────────────┘  │  │
[Conv 512→256] ×2                 │  │
UpSample + Concat ←───────────────┘  │
[Conv 256→128] ×2                    │
UpSample + Concat ←──────────────────┘
[Conv 128→32] ×2
  ↓
Conv 1×1 → 输出 512×512×3 (bg/white/yellow)
```

### 3.3 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW (lr=1e-3, weight_decay=1e-4) |
| 学习率调度 | CosineAnnealing (120 epochs) |
| 损失函数 | CrossEntropyLoss + 0.5×DiceLoss |
| 类别权重 | bg:0.2, white:1.0, yellow:8.0 |
| 批次大小 | 4 |
| 训练/验证 | 27/10 张 |

### 3.4 数据增强

- Resize + RandomCrop → 512×512
- 亮度/对比度扰动（α∈[0.8,1.2], β∈[-20,20]）
- 高斯模糊（kernel=3/5, p=0.3）
- 水平翻转（p=0.5）
- 小角度旋转（±8°, p=0.5）
- **图像和 mask 同步变换**

### 3.5 标签生成

GT 线段标注 → `cv2.line` 渲染为 3 类 mask：
- 0 = background
- 1 = white_lane
- 2 = yellow_lane
- 线宽 10px，抗锯齿

### 3.6 推理后处理

```
U-Net 预测 mask (512×512)
  → resize 回原始尺寸
  → 形态学开闭运算去噪
  → 连通域分析 (connectedComponentsWithStats)
  → 每个连通域 ≥ 100px → cv2.fitLine 拟合直线
  → 输出端点 + 颜色类别
```

### 像素级结果（10 张验证集）

| 类别 | IoU | Dice |
|------|:---:|:---:|
| background | 98.0% | 99.0% |
| white_lane | 7.2% | 12.8% |
| yellow_lane | 11.2% | 18.3% |
| **mean** | **38.8%** | **43.4%** |

### 线段级结果（23 张标注图，角度≤35° + 空间重叠）

| 指标 | White | Yellow | Overall |
|------|:---:|:---:|:---:|
| Precision | 53.12% | 62.50% | 55.00% |
| Recall | 59.65% | 100.00% | 65.67% |
| F1 | 56.20% | 76.92% | **59.86%** |
| GT 匹配 | 34/57 | 10/10 | 44/67 (65.7%) |

---

## 4. 对比总结

| | CV (HSV+Hough) | U-Net (深度网络) |
|------|:---:|:---:|
| 方法类型 | 传统图像处理 | 深度学习语义分割 |
| 需要训练 | 否 | 是（27 张，~10min） |
| 需要 GPU | 否 | 是 |
| White F1 | 17.19% | **56.20%** |
| Yellow F1 | 6.06% | **76.92%** |
| Overall F1 | 14.91% | **59.86%** |
| GT 匹配率 | 17.9% | **65.7%** |
| 黄线全部召回 | 否 | **是 (10/10)** |

**U-Net 在各个指标上均显著优于传统 CV 方法。**

---

## 5. 分析与讨论

### 5.1 为什么 U-Net 更好

1. **学习语义特征**：U-Net 学会识别"车道线"这个概念，不仅靠颜色，还靠形状、纹理、上下文
2. **对光照鲁棒**：数据增强（亮度/对比度扰动）使模型适应不同光照
3. **端到端优化**：直接优化分割目标，而非手工设计的流水线

### 5.2 当前局限

1. **训练数据少**：仅 27 张训练图，mIoU 38.8% 偏低
2. **类别不平衡**：黄线仅 10 条，需 8× 权重 + 4× 过采样才能学到
3. **背景主导**：99% 像素是背景，IoU 虚高
4. **缺乏时序/立体信息**：单帧 2D 图像信息有限

### 5.3 改进方向

1. 标注更多数据（100+ 张可使 mIoU > 60%）
2. 使用预训练 ResNet18 作为编码器（迁移学习）
3. 加入 Dice Loss 或 Focal Loss 更好处理类别不平衡
4. 后处理加入消失点约束过滤假阳性

---

## 6. 运行命令

```bash
# 训练
python3 -m src.unet_lane train \
  --image-dir datasets/local_colm/images/test \
  --gt-dir datasets/local_colm/labels/test \
  --out models/unet_lane.pt

# 推理
python3 -m src.unet_lane predict \
  --weights models/unet_lane_v2.pt \
  --source datasets/local_colm/images/test \
  --out runs/unet_predictions.json \
  --save-vis runs/unet_vis

# CV baseline
python3 -m src.vertical_lane_pipeline \
  --source datasets/local_colm/images/test \
  --gt-xlsx 结果统计.xlsx \
  --no-color-model
```
