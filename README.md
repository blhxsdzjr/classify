# 车道线白/黄分类与统计

本项目用于实验里的白线、黄线检测统计：先用 YOLO 检测/分割车道线，再把结果分成 `white_lane` 和 `yellow_lane`，最后统计检测数、正确数、Precision、Recall、F1。

当前目录已经适配这三个本地文件：

```text
道路example.zip   # example 训练/示例图片
test(1).zip       # test 图片
结果统计.xlsx      # 每张 test 图片的车道线/白线/黄线数量 GT
```

注意：这两个 zip 目前只有图片，没有逐条车道线的 YOLO 标注坐标。因此：

- 可以直接整理数据、推理、按 Excel 做“数量级”统计。
- 如果要监督训练 YOLO，需要补充 `labels/train` 和 `labels/val` 下的 YOLO 标签。
- 如果要严格按“偏离 15 度以内算准确”评估，需要逐条线的 GT 坐标；只有 Excel 数量表无法验证角度。

## 1. 安装

```bash
pip install -r requirements.txt
```

## 2. 准备本地数据

自动解压 `道路example.zip`、`test(1).zip`，并把 `结果统计.xlsx` 转成 JSON：

```bash
python -m src.prepare_local_dataset --overwrite
```

生成目录：

```text
datasets/local_colm/
  data.yaml
  gt_counts.json
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
```

如果后续补了 YOLO 标注，把 `.txt` 标签放进对应的 `labels/train` 和 `labels/val`。

## 3. 训练 YOLO

有标注后训练：

```bash
python -m src.train_yolo \
  --data datasets/local_colm/data.yaml \
  --model yolov8n-seg.pt \
  --epochs 120 \
  --imgsz 960 \
  --batch 8 \
  --device 0 \
  --name local_colm_lane
```

如果直接训练白/黄两类 mAP 很低，可以先把标注合并成一类 `lane_line`，训练只负责找线，推理阶段再用 HSV 判断白/黄。

## 4. 推理

白/黄两类模型：

```bash
python -m src.predict_yolo_lane \
  --weights runs/segment/local_colm_lane/weights/best.pt \
  --source datasets/local_colm/images/test \
  --out predictions.json \
  --class-mode auto \
  --counts-out prediction_counts.csv \
  --save-vis runs/lane_vis
```

一类 `lane_line` 模型：

```bash
python -m src.predict_yolo_lane \
  --weights runs/segment/local_colm_lane_line/weights/best.pt \
  --source datasets/local_colm/images/test \
  --out predictions.json \
  --class-mode hsv \
  --counts-out prediction_counts.csv \
  --save-vis runs/lane_vis
```

`prediction_counts.csv` 会输出：

```text
文件名,车道线数,白线数,黄线数
```

## 5. 用 Excel 数量 GT 统计

适配你当前的 `结果统计.xlsx`：

```bash
python -m src.evaluate_lane_metrics \
  --pred predictions.json \
  --gt-xlsx 结果统计.xlsx \
  --count-only \
  --out metrics_count_only.json
```

或者使用准备数据时生成的 JSON：

```bash
python -m src.evaluate_lane_metrics \
  --pred predictions.json \
  --gt-counts datasets/local_colm/gt_counts.json \
  --count-only \
  --out metrics_count_only.json
```

这里的“正确数”按每张图、每个类别的 `min(预测数量, GT数量)` 估计，只能做数量级统计，不能判断角度是否在 15 度内。

## 6. 有逐条 YOLO GT 时做 15 度评估

如果测试集有逐条线的 YOLO 分割标签：

```bash
python -m src.evaluate_lane_metrics \
  --pred predictions.json \
  --data datasets/local_colm/data.yaml \
  --split test \
  --angle-thr 15 \
  --out metrics_angle.json
```

这个模式会拟合每条预测线和 GT 线的角度，并按 15 度阈值匹配。

## 7. 报告可写的算法依据

只用 YOLO bbox 做车道线检测通常不理想，因为车道线细长，遮挡、车辆、强光都会影响定位。更稳的思路是用 YOLO segmentation 做实例定位，再用颜色后处理区分白线/黄线。

相关论文：

- SCNN: https://arxiv.org/abs/1712.06080
- Ultra Fast Lane Detection: https://arxiv.org/abs/2004.11757
- CLRNet: https://arxiv.org/abs/2203.10350
