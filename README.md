# UFLD-inspired 车道线检测：row-anchor 响应跟踪 + 颜色学习

当前方案借鉴 [cfzd/Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection) 和论文 *Ultra Fast Structure-aware Deep Lane Detection* 的核心思想：**不做 YOLO 检测，也不做逐像素分割，而是把车道线检测变成 row-based selecting 问题**。

论文思路是：在图像中预定义若干行 `row anchors`，模型在每一行上选择车道线所在的横向 grid cell；如果该行没有车道线，就选择背景类。这样比普通分割计算量更低，也能利用全局特征处理遮挡、强光等场景。

本项目当前默认用更稳的 **row-anchor 响应跟踪**：每一行找白/黄线响应峰，再跨行用连续性约束连成车道线。这样避免小数据弱监督训练出的 row-grid 模型乱跳。

```text
图片
  -> 固定 row anchors
  -> 每行找白/黄/边缘响应峰
  -> 跨行动态跟踪连接成车道线
  -> 推理输出 white_lane / yellow_lane
  -> 用 结果统计.xlsx 做数量级评估
```

## 1. 安装

```bash
pip install -r requirements.txt
```

依赖包含 OpenCV、NumPy、PyTorch、tqdm。

## 2. 准备数据

当前目录适配：

```text
道路example.zip
test(1).zip
结果统计.xlsx
```

整理数据：

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

## 3. 推荐推理：row-anchor 响应跟踪

不需要训练模型，先跑这一版看可视化是否正常：

```bash
python -m src.predict_row_anchor \
  --source datasets/local_colm/images/test \
  --out predictions_row_anchor.json \
  --counts-out prediction_counts_row_anchor.csv \
  --save-vis runs/row_anchor_vis
```

默认可视化会画成干净的蓝色平滑曲线，接近手工标线效果；如果想调试每个 row anchor 的点和文字标签，可以加 `--debug-vis`。

如果允许使用 `结果统计.xlsx` 的每图白/黄数量做后处理约束：

```bash
python -m src.predict_row_anchor \
  --source datasets/local_colm/images/test \
  --gt-xlsx 结果统计.xlsx \
  --use-count-constraints \
  --out predictions_row_anchor_constrained.json \
  --counts-out prediction_counts_row_anchor_constrained.csv \
  --save-vis runs/row_anchor_vis_constrained
```

可调参数：

- `--peak-rel-threshold`：响应峰阈值，太多误检就调大，例如 `0.45`。
- `--max-match-distance`：跨 row 连接距离，线断开就调大，乱连就调小。
- `--min-points`：一条线至少经过多少个 row，误检多就调大。

## 4. 数量级评估

```bash
python -m src.evaluate_lane_metrics \
  --pred predictions_row_anchor_constrained.json \
  --gt-xlsx 结果统计.xlsx \
  --count-only \
  --out metrics_row_anchor_count_only.json
```

`结果统计.xlsx` 只有每张图白线/黄线数量，没有逐条线坐标，所以这里只能做数量级 Precision / Recall / F1，不能严格验证 15 度角度规则。

## 5. 可选：训练 TinyUFLD row-grid 模型

如果 row-anchor 响应跟踪可视化稳定，可以进一步把它生成的弱标签拿来训练 TinyUFLD 模型。

生成弱标签：

```bash
python -m src.generate_ufld_labels \
  --image-dir datasets/local_colm/images/test \
  --gt-xlsx 结果统计.xlsx \
  --out-dir datasets/local_colm/ufld_labels/test \
  --index-out datasets/local_colm/ufld_test_index.json
```

训练：

```bash
python -m src.train_ufld \
  --index datasets/local_colm/ufld_test_index.json \
  --out models/ufld_tiny.pt \
  --epochs 60 \
  --batch 8 \
  --device cuda:0
```

如果没有 GPU：

```bash
python -m src.train_ufld \
  --index datasets/local_colm/ufld_test_index.json \
  --out models/ufld_tiny.pt \
  --epochs 60 \
  --batch 4 \
  --device cpu
```

模型输出两部分：

- `grid_logits`：每个 lane slot、每个 row anchor 的横向 grid 分类。
- `color_logits`：每个 lane slot 的颜色分类，白线 / 黄线 / none。

训练中还加入了类似 UFLD 结构先验的平滑损失，使相邻 row 的预测位置不要剧烈跳变。

TinyUFLD 推理：

```bash
python -m src.predict_ufld \
  --weights models/ufld_tiny.pt \
  --source datasets/local_colm/images/test \
  --out predictions_ufld.json \
  --counts-out prediction_counts_ufld.csv \
  --save-vis runs/ufld_vis
```

## 6. 主要代码

- `src/predict_row_anchor.py`：默认推荐，row-anchor 响应峰检测 + 跨行跟踪。
- `src/row_anchor_detector.py`：核心 row-anchor 跟踪算法。
- `src/generate_ufld_labels.py`：生成 row-anchor 弱标签。
- `src/ufld_model.py`：TinyUFLD 模型，row-grid 分类 + 颜色头。
- `src/train_ufld.py`：训练 UFLD-style 模型。
- `src/predict_ufld.py`：推理并输出白线/黄线 JSON、CSV、可视化。
- `src/classical_lane.py`：传统线候选生成，用来启动弱标签。
- `src/evaluate_lane_metrics.py`：统计指标。

## 7. 和论文的关系

借鉴点：

- 使用预定义 row anchors。
- 把车道线位置预测改成横向 grid classification。
- 使用全局图像特征一次性预测多条 lane。
- 加入相邻 row 位置平滑的结构损失。

和原论文不同：

- 原论文在 TuSimple / CULane 这类有逐点 lane 标注的数据集上训练。
- 本项目当前只有数量标注，因此默认先用 row-anchor 响应跟踪，避免弱标签训练不稳。
- 本项目额外加了白/黄颜色头，以适配实验要求。

参考：

- GitHub: https://github.com/cfzd/Ultra-Fast-Lane-Detection
- Paper: https://arxiv.org/abs/2004.11757
