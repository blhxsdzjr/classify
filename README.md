# 非 YOLO 车道线检测：先判定线，再学习颜色

这版项目不使用 YOLO，也不依赖 `ultralytics`。整体思路是传统视觉 + 轻量颜色学习：

```text
图片
  -> Canny / Hough 找线候选
  -> 过滤人行道、横线、短线等干扰
  -> 提取每条线附近的 HSV/Lab 颜色特征
  -> 用 结果统计.xlsx 的白线/黄线数量标注弱监督学习颜色
  -> 输出白线、黄线数量和指标
```

`结果统计.xlsx` 是每张图的白线/黄线数量标注，不是每条线的坐标标注。因此它可以用来训练颜色判别和做数量统计，但不能严格验证“角度偏差 15 度以内”。

## 1. 安装

```bash
pip install -r requirements.txt
```

## 2. 准备数据

当前目录已适配：

```text
道路example.zip
test(1).zip
结果统计.xlsx
```

解压整理：

```bash
python -m src.prepare_local_dataset --overwrite
```

生成：

```text
datasets/local_colm/
  gt_counts.json
  images/train/
  images/val/
  images/test/
```

## 3. 学习白/黄颜色模型

使用 `结果统计.xlsx` 的数量标注做弱监督：先在每张图中找线候选，再按照该图的白线数、黄线数给候选线分配弱标签，最后学习两个颜色原型。

```bash
python -m src.train_color_model \
  --image-dir datasets/local_colm/images/test \
  --gt-xlsx 结果统计.xlsx \
  --out models/color_model.json \
  --samples-out models/color_samples.csv
```

如果不想“学习颜色”，也可以跳过这一步，检测脚本会退回到手写 HSV 规则。

## 4. 运行传统视觉检测

不使用数量约束，只靠线检测 + 颜色模型：

```bash
python -m src.run_classical_pipeline \
  --source datasets/local_colm/images/test \
  --color-model models/color_model.json \
  --out predictions_classical.json \
  --counts-out prediction_counts_classical.csv \
  --save-vis runs/classical_vis
```

如果允许使用 `结果统计.xlsx` 作为数量级弱监督，可以加数量约束。它会按每张图标注的白线数/黄线数，从候选线中选择对应数量的结果：

```bash
python -m src.run_classical_pipeline \
  --source datasets/local_colm/images/test \
  --color-model models/color_model.json \
  --gt-xlsx 结果统计.xlsx \
  --use-count-constraints \
  --out predictions_classical_constrained.json \
  --counts-out prediction_counts_classical_constrained.csv \
  --save-vis runs/classical_vis_constrained
```

注意：带 `--use-count-constraints` 的结果使用了测试集数量标注，应该描述为“数量标注约束后的弱监督结果”，不要说成完全独立的模型泛化性能。

## 5. 统计指标

数量级评估：

```bash
python -m src.evaluate_lane_metrics \
  --pred predictions_classical_constrained.json \
  --gt-xlsx 结果统计.xlsx \
  --count-only \
  --out metrics_classical_count_only.json
```

指标含义：

```text
Precision = 正确检测数 / 检测总数
Recall    = 正确检测数 / GT 总数
F1        = 2 * Precision * Recall / (Precision + Recall)
```

当前 Excel 没有每条线的位置和角度，所以这里的正确数按每张图、每个类别的 `min(预测数量, GT数量)` 统计。

## 6. 主要代码

- `src.prepare_local_dataset.py`：解压数据、读取 Excel 数量标注。
- `src.classical_lane.py`：Canny/Hough 找线、合并线段、提取颜色特征。
- `src.train_color_model.py`：从数量标注中弱监督学习白/黄颜色模型。
- `src.run_classical_pipeline.py`：端到端检测并输出预测 JSON、CSV、可视化。
- `src.evaluate_lane_metrics.py`：统计白线、黄线 Precision / Recall / F1。

## 7. 可以给老师解释的点

本方法没有用 YOLO，而是先做几何意义上的“线检测”。Canny 边缘和 Hough 变换会找出图片中接近直线的结构，再通过 ROI、长度、角度、颜色比例过滤掉一部分车辆、人行道和路面干扰。

颜色不是直接靠固定阈值死判，而是先从每条候选线周围提取 HSV/Lab 特征，例如白色比例、黄色比例、饱和度、亮度、Lab 的黄色通道等；再利用 `结果统计.xlsx` 中每张图的白线数、黄线数，弱监督学习白线和黄线的颜色原型。

局限也要说明清楚：Excel 只有数量，没有每条线的位置，所以无法严格判断某条预测线是否和 GT 的角度相差 15 度以内。要做严格角度评估，需要补充每条线的坐标标注。
