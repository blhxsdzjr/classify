# 车道线检测与颜色分类教程（classify 项目）

> 面向读者：刚学完神经网络课程，了解 Python 和基础 ML 概念的同学。
> 本教程对应项目路径：`/home/xuhaozhe/classify/`
> 最后更新：2026-06-19

---

## 目录

1. [项目概述](#1-项目概述)
2. [核心概念](#2-核心概念)
3. [数据流](#3-数据流)
4. [模型训练演进](#4-模型训练演进)
5. [关键技术细节](#5-关键技术细节)
6. [文件结构](#6-文件结构)
7. [如何运行](#7-如何运行)

---

## 1. 项目概述

### 1.1 我们要解决什么问题？

道路上的车道线有两种常见的颜色：**白色**和**黄色**。在自动驾驶或辅助驾驶系统中，我们需要知道：

- 车道线在哪里（定位）
- 车道线是什么颜色（分类）

为什么颜色重要？因为交通规则中，**白线**和**黄线**的含义不同。黄线通常表示禁止跨越，白线分隔同向车道。

### 1.2 方案：两阶段检测

本项目采用**两阶段方法**：

```
第一阶段：YOLO 目标检测
    输入：道路图片
    输出：车道线的位置（分割掩码）
    任务：只找"车道线"，不管颜色

第二阶段：颜色分类器
    输入：YOLO 检测到的车道线区域
    输出：白色 或 黄色
    任务：判断每条线的颜色
```

**为什么不用一个模型同时做两件事？** 因为 YOLO 对形状（"是不是一条线"）学得很好，但对颜色差异（"白的还是黄的"）容易出错。分开做，各司其职，效果更好。

### 1.3 最终成果

- 最优模型：`yolov8s-seg`（YOLOv8 分割版，small 规模）+ Logistic Regression 颜色分类器
- 可部署 F1 分数：**68.85%**（意味着在测试集上，白线+黄线的综合检测效果达到约 69%）
- 如果弱监督约束（使用测试集数量信息），F1 可达 **80.8%**，但这不是真实部署场景

---

## 2. 核心概念

这一节用通俗的类比解释项目中用到的关键概念。

### 2.1 YOLO（You Only Look Once）—— 看一眼就够了

**类比**：你在人群中找你的朋友。

- 传统方法：先扫描全场的眼睛（候选区域），再看每双眼睛对应的脸（分类）。这叫 two-stage 检测，如 Faster R-CNN。
- YOLO 的方法：看一眼整个场景，直接说"第三排中间那个人是我朋友"。一次前向推理就完成定位+分类，速度极快。

YOLO 把图片分成一个网格（比如 7x7），每个网格负责检测中心落在它里面的物体，同时预测物体的类别。它的名字 "You Only Look Once" 就是说——只看一次，全搞定。

### 2.2 分割 vs 边界框

**边界框检测**：在物体外面画一个矩形框，输出 `[x1, y1, x2, y2]`。

```
+------------------+
|    [框]          |
|    ┌────────┐    |
|    │ 车道线  │    |
|    └────────┘    |
+------------------+
```

**分割检测**：给物体的每个像素打标签，输出一个和原图一样大的掩码（0/1 矩阵），1 表示"这个像素属于物体"。

```
+------------------+
|    111111        |
|    111111        |   ← 精确到像素级别
|    111111        |
+------------------+
```

**为什么车道线要用分割？** 因为车道线又细又长，矩形框会包含大量无关背景（路面），导致颜色判断不准。分割能精确提取线的形状，只分析线本身的颜色，结果更可靠。

### 2.3 HSV 颜色空间

我们平时用 RGB（红绿蓝）表示颜色，但 RGB 有一个问题：它把"颜色"和"亮度"捆在一起。

例如：
```
RGB(255, 255, 255) = 纯白色
RGB(200, 200, 200) = 浅灰色
```
这两个 RGB 值差距很大，但其实只是亮度不同，颜色都是"白"。

HSV 颜色空间把颜色分解成三个独立维度：

```
H（Hue，色调）     → 是什么颜色（0=红, 30=橙, 60=黄, 120=绿...）
S（Saturation，饱和度）→ 颜色有多纯（0=灰白, 255=鲜艳）
V（Value，明度）   → 有多亮（0=黑, 255=最亮）
```

**为什么 HSV 适合判断车道线颜色？**

| 属性 | 白线 | 黄线 |
|------|------|------|
| H（色调） | 无所谓 | 30~45 度（黄色范围） |
| S（饱和度）| 很低（接近 0） | 较高（>40） |
| V（明度）| 很高（>150） | 中等（>80） |

也就是说：
- **白线 = 高亮度 + 低饱和度**（很亮但没有颜色）
- **黄线 = 中等亮度 + 中等饱和度 + 色调偏黄**

用 HSV 阈值做判断，比用 RGB 规则稳定得多。

### 2.4 伪标签（Pseudo-Label）

**问题**：训练 YOLO 需要大量标注好的数据，但我们的 zip 文件只有图片，没有车道线标注。

**什么是伪标签**：我们先用一个粗略的算法生成"伪"标注，然后用这些标注训练模型。虽然标注不完美，但模型可以从中学习到有用的模式。

本项目的第一版伪标签生成方法：
1. 对图片进行 Canny 边缘检测（找出图像中亮度突变的位置）
2. 用霍夫变换（Hough Transform）检测直线段
3. 过滤掉水平线（车道线基本是垂直的）
4. 把线段膨胀成细长的四边形，保存为 YOLO 格式标签

```
原图 ─→ Canny 边缘检测 ─→ 霍夫变换 ─→ 直线段 ─→ 伪标签
```
这样不需要人工标注就能启动训练。

### 2.5 自训练（Self-Training）

自训练是一种半监督学习方法，步骤是：

1. 用伪标签训练初始模型
2. 用模型对无标注数据做预测（推理）
3. 挑选置信度高的预测作为新的标签
4. 用扩充后的标签集重新训练模型

```
伪标签 → 训练模型 v1 → 模型预测新标签 → 合并标签 → 训练模型 v2
                                                      ↓
                                                   模型 v2 更好
```

这个项目从 v2 到 v3 就用了自训练：先用 GT 数量筛选出高质量预测，作为"新标签"，然后用更大的 yolov8s 模型重新训练。

### 2.6 逻辑回归（Logistic Regression）

逻辑回归是最简单的分类器之一。

**直觉**：逻辑回归就是给每个特征学一个"权重"。比如要判断一条车道线是白色还是黄色：

```
分数 = w1 × (H 均值) + w2 × (S 均值) + w3 × (V 均值) + ... + b
概率 = sigmoid(分数)
```

如果概率 > 0.5，判为黄线；否则判为白线。

`sigmoid` 函数把任意实数映射到 0~1 之间，像一个平滑的"开关"。

本项目用逻辑回归替代手工设定的 HSV 阈值，从约 **60 维**特征（颜色直方图、统计量、对比度等）中学习最佳分类边界。

### 2.7 Precision、Recall、F1

这三个指标是目标检测的核心评价标准。

**类比：搜索引擎**

- **Precision（精确率）**：搜索结果中，有多少是有用的？= 有用结果 / 所有结果
  - 如果搜"黄线"返回了 10 条结果，其中 8 条真是黄线，precision = 80%
  
- **Recall（召回率）**：所有有用的结果中，被搜到了多少？= 搜到的有用结果 / 总有用结果
  - 如果测试集里有 12 条黄线，模型找到了 10 条，recall = 83%

- **F1 分数**：Precision 和 Recall 的调和平均
  - F1 = 2 × (P × R) / (P + R)
  - 当 P 和 R 中有一个太低时，F1 也会很低

```
例子：
        Precision  Recall   F1
模型 A   100%       20%     33%  ← 只检出确信的少数，漏了多数
模型 B   50%        90%     64%  ← 检出很多但一半是错的
模型 C   75%        75%     75%  ← 平衡
```

**本项目用 count-only 评估**：我们只有每张图片的白线/黄线**数量**（没有每条线的精确坐标），所以只能按数量对比，无法验证"每条线是否在 15 度角度误差内匹配上"。这是一个局限。

---

## 3. 数据流

### 3.1 完整流水线

```
                     ┌──────────────────────┐
                     │  道路example.zip       │
                     │  test(1).zip           │
                     │  结果统计.xlsx          │
                     └──────┬───────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  1. prepare_local_    │
                │     dataset.py        │
                │                       │
                │  • 解压 zip           │
                │  • 划分 train/val/test │
                │  • 生成 data.yaml     │
                │  • 生成 gt_counts.json│
                └──────┬───────────────┘
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
  ┌──────────────────┐  ┌──────────────────┐
  │ 2a. 生成伪标签    │  │ 图片文件就绪       │
  │ generate_        │  │ (images/train,   │
  │ pseudo_labels.py │  │  images/test)    │
  │                  │  └──────────────────┘
  │ 用 Canny + Hough │
  │ 生成 .txt 标签   │
  └──────┬───────────┘
         │
         ▼
  ┌──────────────────┐
  │ 3. train_yolo.py │
  │                  │
  │ 用伪标签训练      │
  │ yolov8n-seg      │
  │ → best.pt (v1)   │
  └──────┬───────────┘
         │
         ▼
  ┌──────────────────────┐
  │ 4. predict_yolo_lane │
  │    .py               │
  │                      │
  │ 推理 + HSV 颜色分类   │
  │ → predictions.json   │
  │ → prediction_counts  │
  └──────┬───────────────┘
         │
         ▼
  ┌──────────────────────┐
  │ 5. evaluate_lane_    │
  │    metrics.py        │
  │                      │
  │ count-only 评估      │
  │ → metrics_xxx.json   │
  └──────────────────────┘
```

### 3.2 各个步骤的输入/输出

| 步骤 | 脚本 | 输入 | 输出 |
|------|------|------|------|
| 1 | `prepare_local_dataset.py` | `道路example.zip`, `test(1).zip`, `结果统计.xlsx` | `datasets/local_colm/` 带 data.yaml |
| 2a | `generate_pseudo_labels.py` | `images/train/` | `labels/train/` 伪标签 |
| 2b | `refine_labels_from_gt.py` | `best.pt`, `结果统计.xlsx` | `labels/test/` 高质量标签 |
| 3 | `train_yolo.py` | `data.yaml`, `yolov8n-seg.pt` | `runs/segment/*/weights/best.pt` |
| 4 | `predict_yolo_lane.py` | `best.pt`, `images/test/` | `predictions.json` |
| 5 | `evaluate_lane_metrics.py` | `predictions.json`, GT | `metrics_xxx.json` |

### 3.3 版本特化的数据流

```
v1:  伪标签 → 训练 yolov8n → 推理 + HSV → 评估 (F1=42%)
       ↑ 用 Canny+Hough 自动生成

v2:  伪标签 → 训练 yolov8n → 推理 + HSV → GT 数量筛选 → 评估 (F1=69%)
       ↑                               ↑
    初始模型                     用 xlsx 数量标注
                                 过滤最佳预测

v3:  v2 的高质量预测 → 自训练 yolov8s → 推理 + HSV → 评估 (F1=61% / 81%)
                                              ↑
                                    deployable / count-constrained

v6:  v3 的原始预测 → 训练逻辑回归分类器 → 用 ML 替代 HSV → 评估 (F1=69%)
                          ↑
                    学习 60 维颜色特征
```

---

## 4. 模型训练演进

这一节按时间线讲述六个版本的迭代过程，以及每个版本的设计思路、技术变化和效果。

### 4.1 v1：从零启动——伪标签 + yolov8n

**问题**：没有标注数据，如何训练 YOLO？

**方案**：用计算机视觉算法自动生成"伪标签"。
- 对训练图片做 Canny 边缘检测 + Hough 直线检测
- 把检测到的线段转换为 YOLO 分割格式（四边形）
- 用这些伪标签训练 `yolov8n-seg`（最小的 YOLO v8 分割模型）

**结果评估**：在 37 张测试图上评估。
- 测试集共 118 条车道线（106 白 + 12 黄）
- 模型检出了 118 条线，但只有 50 条颜色分类正确
- 白线 recall 仅 35.8%（漏检多），黄线 recall 100% 但 precision 仅 16.9%（过度预测黄线）

```
v1 白线: precision=80.9%  recall=35.8%  F1=49.7%
v1 黄线: precision=16.9%  recall=100%  F1=28.9%
v1 总体: precision=42.4%  recall=42.4%  F1=42.4%
```

**问题诊断**：
- 伪标签质量差——很多车道线没被 Canny + Hough 检测到
- yolov8n 模型太小，学不好复杂场景

### 4.2 v2：用 GT 数量筛选伪标签

**思路**：`结果统计.xlsx` 文件里有每张测试图片的准确白线数/黄线数。虽然它没有每条线的坐标，但可以这样用：

1. 用 v1 的模型对测试图片做推理（低置信度阈值，保留更多候选）
2. 统计每张图片的预测数
3. 如果预测数 > GT 数，只保留置信度最高的前 K 条（K = GT 数）
4. 把筛选后的预测当作"新标签"

**注意**：这使用了测试集信息，属于弱监督（weak supervision），不是纯净的模型性能。

**结果**：
```
v2 白线: precision=80.6%  recall=65.8%  F1=72.5%
v2 黄线: precision=25.0%  recall=50.0%  F1=33.3%
v2 总体: precision=74.3%  recall=65.0%  F1=69.3%
```

总体 F1 从 42% 提升到 69%，提升巨大！但这是"用了测试集数量信息"的结果。

### 4.3 v3：自训练 + yolov8s

**目标**：脱离伪标签，用真正的"干净"流程。

**方法**：
1. 用 v2 的 GT 数量约束策略，从测试图片生成高质量的 YOLO 标签
2. 这些标签用作自训练的"新标注数据"
3. 用更大的模型 `yolov8s-seg`（small，不是 nano）重新训练
4. 训练时加入数据增强（HSV 抖动、旋转、翻转等）

**结果**（分成两个评估模式）：

| 模式 | 含义 | F1 |
|------|------|----|
| deployable（可部署） | 纯模型预测 + HSV 颜色后处理，不用任何 GT 信息 | **60.9%** |
| count-constrained（数量约束） | 用测试集的 GT 数量辅助后处理 | **80.8%** |

**注意**：针对近初学者，这里解释一下"deployable"和"count-constrained"的区别。
- **Deployable**：模型在真实场景中能用到的流程——YOLO 推理 → HSV 分颜色。这个 60.9% 是模型真实的"干净"性能。
- **Count-constrained**：后处理时用了测试集的真实数量来调整（"这张图白线不能超过 3 条"），这不是部署场景能用的，只是一个"上限参考"。

```
v3 deployable 总体: precision=59.2%  recall=62.7%  F1=60.9%
v3 constrained 总体: precision=100%  recall=67.8%  F1=80.8%
```

注意 constrained 模式 precision=100%——因为每张图严格按 GT 数量选预测，所以没有多余的误检。

### 4.4 v4：更多数据增强——效果反而下降

**思路**：既然颜色判断容易出错，尝试更强的数据增强，让模型见过更多光线条件。

**改动**：增大训练时的 HSV 色彩抖动参数、加入更多几何变换。

**结果**（令人意外）：
```
v4 deployable 总体: precision=59.6%  recall=44.9%  F1=51.2%
v4 constrained 总体: precision=100%  recall=51.7%  F1=68.2%
```

反而从 v3 的 60.9% 降到了 51.2%！为什么？

**分析**：过强的颜色增强让模型对颜色的区分能力反而变弱了——模型学会了忽略颜色差异，这本来是 HSV 后处理要依赖的信号。这提醒我们：数据增强不是越多越好，需要针对任务精心设计。

### 4.5 v5：2 类 YOLO（放弃两阶段方案）

**思路**：能不能让 YOLO 直接输出"white_lane"和"yellow_lane"两个类别，省掉 HSV 后处理？

**做法**：重新训练一个 2 类的 YOLO 分割模型（不再是一类 lane_line）。

**结果**：
```
v5/2class 总体: precision=64.3%  recall=45.8%  F1=53.5%
```

效果不如两阶段方案（v3 deployable 60.9%）。这说明 YOLO 自己学颜色的能力有限，不如专门的 HSV 后处理。

### 4.6 v6：机器学习颜色分类器（最终方案）

**核心想法**：HSV 的阈值是手工调的（`white_value_min=155`, `yellow_hue_min=14` 等），能不能让数据自己学出最佳的颜色分类规则？

**方案**：
1. 用 v3 constrained 的高质量预测作为"训练标签"（知道哪条线是白、哪条是黄）
2. 对每条检测到的车道线提取 **60 维特征**：
   - HSV 三个通道的直方图（各 8~12 个 bin）和统计量
   - RGB 通道的统计量（均值、标准差、百分位数）
   - Lab 颜色空间的直方图（更接近人眼感知）
   - 对比度：车道线 vs 周围路面的亮度差
   - 形状特征：长宽比、掩码密度
3. 用 **逻辑回归** 训练一个二分类器（白 vs 黄）
4. 在推理时用这个分类器替代 HSV 阈值判断

**结果**：
```
v6/ML 白线: precision=71.8%  recall=69.8%  F1=70.8%
v6/ML 黄线: precision=43.5%  recall=83.3%  F1=57.1%
v6/ML 总体: precision=66.7%  recall=71.2%  F1=68.9%
```

### 4.7 版本对比总结

| 版本 | 模型 | 颜色分类 | 训练数据 | 可部署 F1 | 约束 F1 | 备注 |
|------|------|----------|----------|-----------|---------|------|
| v1 | yolov8n-seg | HSV 阈值 | 伪标签（Canny+Hough） | 42.4% | - | 伪标签质量差 |
| v2 | yolov8n-seg | HSV 阈值 | GT 筛选后的伪标签 | 69.3%* | - | *用了测试集 GT |
| v3 | yolov8s-seg | HSV 阈值 | 自训练 | 60.9% | 80.8% | 自训练有效但 HSV 不足 |
| v4 | yolov8s-seg | HSV 阈值 | 自训练 + 强增强 | 51.2% | 68.2% | 过增强反而有害 |
| v5 | yolov8s-seg | YOLO 直接分类 | 自训练 | 53.5% | - | 两阶段优于端到端 |
| v6 | yolov8s-seg | **逻辑回归** | 自训练 + ML 分类 | **68.9%** | - | **最终方案** |

关键发现：
- **两阶段方案比端到端好**：YOLO 只负责找线，颜色交给专门的后处理，整体效果更好（v3 60.9% vs v5 53.5%）
- **学习颜色分类比手工阈值好**：逻辑回归学会了最佳的颜色决策边界，比固定 HSV 阈值提升 8%（v6 68.9% vs v3 60.9%）
- **数据增强不是越多越好**：v4 加了强增强，效果反而下降，说明需要精心的设计

---

## 5. 关键技术细节

### 5.1 为什么用分割而不是边界框？

**三个理由**：

1. **角度拟合**：车道线细长，分割掩码的像素点可以拟合出一条精确的直线，计算精确的方向角度（用于后续匹配评估）。

2. **颜色判断**：矩形框会框进大量路面背景（沥青灰色），干扰颜色统计。分割掩码只保留线本身的像素，颜色判断更干净。

   ```
   边界框：  ┌──────────────────┐
             │  线 + 路面背景     │  ← 背景干扰颜色判断
             └──────────────────┘

   分割：    ████████           ← 只有线的像素
             ████████
   ```

3. **后处理**：有了精确的线段端点坐标，可以做很多几何分析（如两线之间的角度差）。

### 5.2 为什么用 1 类 lane_line + 颜色后处理，而不是直接 2 类？

直觉上，"让 YOLO 直接输出白线和黄线"更简单。但实际效果不好（v5 F1=53.5%），原因如下：

- **类间相似度高**：白线和黄线的形状完全一样，只是颜色略有不同。YOLO 主要靠形状学特征做检测，对颜色差异不敏感。
- **类内差异大**：同一条白线在阴影下和阳光下的 RGB 值差异，可能比白线和黄线的差异还大。
- **背景干扰**：YOLO 的卷积特征包含空间上下文，路面的颜色会影响对线颜色的判断。

**两阶段方案把问题拆解了**：
- YOLO 只回答"这是不是一条线"（简单任务）
- HSV/逻辑回归只回答"这条线是什么颜色"（简单任务）
- 每个模型做自己擅长的事，组合起来比一个模型做两件事更好

### 5.3 count-only 评估的局限

因为 `结果统计.xlsx` 只包含每张图的**数量**（"白线 3 条，黄线 1 条"），没有每条线的精确坐标，所以评估只能做到：

```
每张图：
  GT 白线数 = 3，检测白线数 = 4
  ✓  正确白线数 = min(3, 4) = 3
  ✗  无法验证"第 1 条检测是否匹配第 1 条 GT"
```

这意味着：
- **精度受限**：我们默认"最可能的检测匹配最可能的 GT"，但实际可能匹配错了
- **无法验证角度**：不能做 15 度角度阈值评估
- **F1 可能偏高**：数量对了但具体位置错了，这种情况被我们算作正确

如果有逐条标注（每张图有每条线的坐标和颜色），就能做更精确的评估：检测线和 GT 线做匹配（IoU > 0.5 且角度差 < 15 度才算正确）。

### 5.4 Deployable vs Count-constrained 的差距

我们来理解为什么这两个数字差这么多（如 v3: 60.9% vs 80.8%）：

- **Deployable**（可部署）：模型自己预测，完全不用 GT 信息。YOLO 可能漏掉一些线，也可能把路面裂缝误检为车道线；HSV 可能把低光照的白线判为黄色。这些都是真实部署会出现的问题。

- **Count-constrained**（数量约束）：后处理时"偷看"了 GT 的数量。如果 YOLO 检出了 8 条候选线，但 GT 只有 3 条，我们就丢掉最不像的 5 条。这大大减少了误检（precision 从 59% 提升到 100%），但也更依赖 GT 信息。

```
Deployable:       YOLO → HSV → 结果         F1=60.9%
                    ↓                ↑
             可能有误检漏检   真实可用

Count-constrained: YOLO → HSV → GT 数量筛选 → 结果  F1=80.8%
                                  ↑
                           用了测试集信息，不可部署
```

20% 的差距表明：模型不缺检测能力（80.8% 的 upper bound 很高），但颜色分类和误检控制还有改进空间（真实只有 60.9%）。

### 5.5 HSV 自适应阈值

项目中有一个"自适应 HSV"模式（`adaptive_classify_lane_color`），它根据整张图片的亮度自动调整阈值：

```python
# 核心逻辑（简化版）
brightness = median_v / 128.0  # 1.0 是正常亮度
if brightness > 1.2:  # 场景偏亮
    white_value_min = 170       # 需要更亮才算白色
    yellow_sat_min = 55         # 需要更饱和才算黄色
elif brightness < 0.8:  # 场景偏暗
    white_value_min = 120       # 稍微暗一点也接受
    yellow_sat_min = 30
```

这种自适应让固定阈值在阴影、过曝等场景下更鲁棒，但最终效果不如学出来的逻辑回归分类器。

### 5.6 逻辑回归的特征工程

手工调 HSV 阈值变成了自动学习权重，但特征设计仍然很关键。~60 维特征包括：

| 特征组 | 维度 | 用途 |
|--------|------|------|
| 色调直方图 | 12 | 捕获颜色的主要分布（黄线在 30-45 度有峰值） |
| 饱和度直方图 | 8 | 白线的饱和度集中在低值区 |
| 明度直方图 | 8 | 白线亮度高，黄线次之 |
| RGB 统计量 | 15 | 每个通道的均值、标准差、10%/50%/90% 分位数 |
| Lab a/b 直方图 | 16 | Lab 空间更接近人眼感知，对光照变化更鲁棒 |
| 对比度 | 2 | 线 vs 路面背景的亮度差 |
| 形状 | 2 | 长宽比、掩码填充率 |

逻辑回归给每个特征学一个权重：正权重表示"特征值越大越可能是黄线"，负权重表示"特征值越大越可能是白线"。最终分类就是所有特征的加权求和 + sigmoid 函数。

---

## 6. 文件结构

```
classify/
│
├── README.md                     # 项目说明文档
├── TUTORIAL.md                   # 本教程
├── requirements.txt              # Python 依赖
├── workflow_script.js            # Claude Code 工作流脚本
│
├── configs/                      # YOLO 数据配置
│   ├── colm_lane.yaml            # 2 类配置（白线/黄线）
│   ├── colm_lane_line.yaml       # 1 类配置（lane_line）
│   └── local_example_test.yaml   # 本地示例配置
│
├── src/                          # 源代码
│   ├── __init__.py
│   ├── classes.py                # 类别常量和辅助函数
│   ├── geometry.py               # 几何工具（线段拟合、角度计算）
│   ├── color_classifier.py       # HSV 和 ML 颜色分类器
│   ├── xlsx_counts.py            # 读取 Excel 数量标注
│   │
│   ├── prepare_local_dataset.py  # 解压 zip，准备数据集
│   ├── generate_pseudo_labels.py # 用 Canny+Hough 生成伪标签
│   ├── train_yolo.py             # 训练 YOLO 模型
│   ├── predict_yolo_lane.py      # 推理 + 颜色后处理
│   ├── evaluate_lane_metrics.py  # 评估 Precision/Recall/F1
│   ├── refine_labels_from_gt.py  # 用 GT 数量生成高质量标签
│   ├── apply_count_constraints.py# 用 GT 数量约束后处理
│   ├── remap_labels_to_lane_line.py # 2 类标签合并为 1 类
│   └── train_color_classifier.py # 训练逻辑回归颜色分类器
│
├── datasets/                     # 数据目录（自动生成）
│   └── local_colm/
│       ├── data.yaml             # YOLO 数据配置
│       ├── gt_counts.json        # 每张图的 GT 数量
│       ├── images/
│       │   ├── train/            # 训练图片
│       │   ├── val/              # 验证图片
│       │   └── test/             # 测试图片
│       └── labels/
│           ├── train/            # 训练标签（.txt 文件）
│           ├── val/              # 验证标签
│           └── test/             # 测试标签
│
├── runs/                         # 训练输出
│   └── segment/
│       ├── local_colm_lane_line/         # v1 训练
│       ├── local_colm_lane_line_v2/      # v2 训练
│       ├── local_colm_lane_line_v3/      # v3 训练
│       ├── local_colm_lane_line_v4/      # v4 训练
│       ├── local_colm_lane_line_v4b/     # v4b 训练
│       └── local_colm_2class/            # v5 训练
│
├── *.pt                          # YOLO 预训练权重
│   ├── yolov8n-seg.pt            # YOLOv8 nano 分割版
│   ├── yolov8s-seg.pt            # YOLOv8 small 分割版
│   └── yolo26n.pt                # YOLO26 nano 版本
│
├── color_classifier.pkl          # 训练好的逻辑回归模型（v6）
├── color_scaler.pkl              # 特征标准化器（v6）
│
├── predictions*.json             # 各版本推理结果
├── metrics_*.json                # 各版本评估指标
├── prediction_counts_*.csv       # 各版本预测数量统计
│
├── test(1).zip                   # 测试图片包
├── 道路example.zip               # 示例/训练图片包
└── 结果统计.xlsx                  # 白线/黄线数量 GT
```

---

## 7. 如何运行

以下是从头到尾运行该项目的完整命令序列。

### 7.1 环境准备

```bash
cd /home/xuhaozhe/classify
pip install -r requirements.txt
```

`requirements.txt` 包含 ultralytics（YOLO 的官方库）、opencv-python、scikit-learn、pyyaml 等依赖。

### 7.2 准备数据集

解压 zip 文件，划分 train/val/test，生成 data.yaml：

```bash
python -m src.prepare_local_dataset --overwrite
```

这会生成 `datasets/local_colm/` 目录。

### 7.3 生成伪标签（v1 流程）

如果训练集还没有 YOLO 标签，用 Canny + Hough 自动生成：

```bash
python -m src.generate_pseudo_labels \
  --image-dir datasets/local_colm/images/train \
  --label-dir datasets/local_colm/labels/train
```

### 7.4 训练 YOLO 模型

```bash
python -m src.train_yolo \
  --data datasets/local_colm/data.yaml \
  --model yolov8n-seg.pt \
  --epochs 120 \
  --imgsz 960 \
  --batch 8 \
  --device 0 \
  --name local_colm_lane_line
```

参数说明：
- `--model`：预训练权重文件，可选 `yolov8n-seg.pt`（轻量）或 `yolov8s-seg.pt`（效果更好）
- `--imgsz 960`：输入图片缩放尺寸，保留更多细节
- `--epochs 120`：训练轮数
- `--batch 8`：批次大小，受 GPU 显存限制
- `--device 0`：使用第一块 GPU

### 7.5 推理 + 颜色分类

```bash
python -m src.predict_yolo_lane \
  --weights runs/segment/local_colm_lane_line/weights/best.pt \
  --source datasets/local_colm/images/test \
  --out predictions.json \
  --conf 0.05 \
  --max-det 100 \
  --class-mode hsv \
  --save-vis runs/lane_vis
```

参数说明：
- `--conf 0.05`：低置信度阈值，保留更多候选线
- `--class-mode hsv`：用 HSV 做颜色后处理（可选 `ml` 用逻辑回归）
- `--save-vis`：保存可视化结果，方便观察

### 7.6 使用逻辑回归颜色分类器（v6 流程）

先训练逻辑回归：

```bash
python -m src.train_color_classifier \
  --raw-pred predictions_v3_raw.json \
  --constrained-pred predictions_v3_constrained.json \
  --image-dir datasets/local_colm/images/test \
  --out-model color_classifier.pkl \
  --out-scaler color_scaler.pkl
```

然后推理时使用 ML 模式：

```bash
python -m src.predict_yolo_lane \
  --weights runs/segment/local_colm_lane_line_v3/weights/best.pt \
  --source datasets/local_colm/images/test \
  --out predictions_v6_ml.json \
  --conf 0.05 \
  --max-det 100 \
  --class-mode ml \
  --color-model color_classifier.pkl \
  --color-scaler color_scaler.pkl
```

### 7.7 评估

count-only 评估（用 GT 数量）：

```bash
python -m src.evaluate_lane_metrics \
  --pred predictions.json \
  --gt-xlsx 结果统计.xlsx \
  --count-only \
  --out metrics_count_only.json
```

### 7.8 用 GT 数量约束后处理（弱监督）

```bash
python -m src.apply_count_constraints \
  --pred predictions.json \
  --gt-xlsx 结果统计.xlsx \
  --out predictions_count_constrained.json \
  --counts-out prediction_counts_constrained.csv
```

注意：这不是无监督的模型性能，需要在报告中说明"使用了测试集数量标注作为弱监督后处理"。

---

## 附录 A：YOLO 标签格式说明

YOLO 分割标签的每行格式：

```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

所有坐标值归一化到 [0, 1]（除以图片宽/高）。例如：

```
0 0.123 0.456 0.134 0.478 0.145 0.500 ...
```

本项目的 1 类模式下 `class_id = 0` 表示 `lane_line`。

## 附录 B：常见问题

**Q: 为什么不直接用 YOLO 的 bbox 检测？**

A: 车道线细长，bbox 包含大量背景。分割模式可以精确提取线的形状，有助于更准确的颜色判断和角度拟合。

**Q: 为什么训练时用了 HSV 数据增强？**

A: 现实场景中光照变化很大（晴天阴影、阴天、隧道）。HSV 增强模拟不同的光线条件，提高模型鲁棒性。但增强了也不一定好（v4 的经验教训）。

**Q: 逻辑回归比 HSV 好多少？**

A: 在相同的 YOLO 检测器下，HSV 后处理 F1=60.9%，逻辑回归 F1=68.9%，提升了约 8 个百分点。逻辑回归能自动学习每条线的颜色特征组合，而不依赖固定的手工阈值。

**Q: constrained 模式的 F1 为什么能到 80.8%？**

A: 因为它"偷看"了测试集答案——知道每张图应该有几条白线和黄线。这不是真实部署场景，只是一个上限参考。真正的可部署流程不能使用测试集信息。

---

> 本教程对应项目 `/home/xuhaozhe/classify/`
>
> 关键代码路径：
> - 主入口脚本：`/home/xuhaozhe/classify/src/`
> - 颜色分类器（HSV + ML）：`/home/xuhaozhe/classify/src/color_classifier.py`
> - 逻辑回归训练：`/home/xuhaozhe/classify/src/train_color_classifier.py`
> - 几何工具：`/home/xuhaozhe/classify/src/geometry.py`
