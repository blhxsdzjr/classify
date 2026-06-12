# 车道线 lane_line 检测 + 白/黄后处理

本项目现在默认采用更稳的两阶段方案：

1. YOLO 只训练一个类别：`lane_line`
2. 推理时对每条检测到的线做 HSV 颜色判断，输出 `white_lane` 或 `yellow_lane`
3. 最后按白线、黄线分别统计检测数、正确数、Precision、Recall、F1

这样能减少强光、车辆遮挡、白黄颜色差异对 YOLO mAP 的影响。YOLO 只负责“找车道线”，颜色交给后处理。

## 当前本地数据

项目已适配当前目录下的文件：

```text
道路example.zip   # example 训练/示例图片
test(1).zip       # test 图片
结果统计.xlsx      # 每张 test 图片的车道线/白线/黄线数量 GT
```

注意：这两个 zip 目前只有图片，没有逐条车道线 YOLO 标注。真正训练 YOLO 仍需要 `.txt` 标签。

## 1. 安装

```bash
pip install -r requirements.txt
```

## 2. 准备数据

默认会生成一类 `lane_line` 数据配置。如果 zip 或目标目录里有白/黄两类标签，脚本会把源类别 `0,1` 自动合并成 `0 lane_line`，并跳过其他类别：

```bash
python -m src.prepare_local_dataset --overwrite
```

生成：

```text
datasets/local_colm/
  data.yaml          # names: 0 lane_line
  gt_counts.json
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
```

如果你的原始标签里车道线类别不是 `0,1`，例如只有 `0` 是车道线：

```bash
python -m src.prepare_local_dataset --overwrite --lane-class-ids 0
```

如果仍想训练白/黄两类模型，可以显式使用：

```bash
python -m src.prepare_local_dataset --overwrite --label-mode color
```

## 3. 训练一类 YOLO

补好 `labels/train` 和 `labels/val` 后运行：

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

推荐用 `yolov8n-seg.pt` 或更大的 `yolov8s-seg.pt`，因为车道线细长，segmentation 比 bbox detection 更适合拟合角度和做颜色判断。

## 4. 推理并分白/黄

推理默认使用 HSV 后处理分白线/黄线：

```bash
python -m src.predict_yolo_lane \
  --weights runs/segment/local_colm_lane_line/weights/best.pt \
  --source datasets/local_colm/images/test \
  --out predictions.json \
  --counts-out prediction_counts.csv \
  --save-vis runs/lane_vis
```

输出：

- `predictions.json`：每条线的类别、置信度、角度、bbox、端点
- `prediction_counts.csv`：每张图的车道线数、白线数、黄线数
- `runs/lane_vis/`：可视化结果

## 5. 用 Excel 数量 GT 统计

当前 `结果统计.xlsx` 只有每张图的数量，没有逐条线坐标，所以只能做数量级统计，不能验证“15 度以内算准确”。

```bash
python -m src.evaluate_lane_metrics \
  --pred predictions.json \
  --gt-xlsx 结果统计.xlsx \
  --count-only \
  --out metrics_count_only.json
```

也可以用准备数据时生成的 JSON：

```bash
python -m src.evaluate_lane_metrics \
  --pred predictions.json \
  --gt-counts datasets/local_colm/gt_counts.json \
  --count-only \
  --out metrics_count_only.json
```

## 6. 有逐条 GT 时做 15 度评估

如果测试集补充了逐条车道线的 YOLO 分割标签，并且 GT 标签里仍区分白/黄线，可以运行：

```bash
python -m src.evaluate_lane_metrics \
  --pred predictions.json \
  --data configs/colm_lane.yaml \
  --image-dir datasets/local_colm/images/test \
  --gt-label-dir datasets/local_colm_color_gt/labels/test \
  --angle-thr 15 \
  --out metrics_angle.json
```

这个模式会拟合预测线和 GT 线的角度，并按白线/黄线类别分别匹配。

## 7. 报告写法

可以这样描述方法：

> 为降低强光、遮挡和颜色差异对检测器的影响，本文采用两阶段车道线检测方法。第一阶段使用 YOLO segmentation 仅检测 `lane_line` 一类，提高车道线定位召回率；第二阶段对检测区域进行 HSV 颜色统计，将车道线划分为白线和黄线。最终评估仍按白线、黄线分别统计 Precision、Recall 和 F1。

参考：

- SCNN: https://arxiv.org/abs/1712.06080
- Ultra Fast Lane Detection: https://arxiv.org/abs/2004.11757
- CLRNet: https://arxiv.org/abs/2203.10350
