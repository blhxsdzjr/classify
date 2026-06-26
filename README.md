# 竖直车道线检测与白黄分类

本项目用于课程测试集中的竖直/近竖直车道线检测，并按白线、黄线分别统计数量、Precision、Recall 和 F1。可视化中白线用红色描出，黄线用蓝色描出。

当前主流程是传统视觉 + 轻量颜色分类器：

```text
图片
  -> 白/黄 HSV 颜色候选
  -> ROI + 近竖直 Hough 线段
  -> 同方向、同位置断续段聚合成一条线
  -> 可选 LR/MLP 颜色模型修正白黄类别
  -> 输出 JSON / CSV / XLSX / 可视化图片
```

## 安装

```bash
pip install -r requirements.txt
```

如果电脑上 `python` 指向 Anaconda 且没有 OpenCV，可以用 Windows 的 Python launcher：

```bash
py -3 -m src.vertical_lane_pipeline --help
```

## 准备数据

当前目录默认适配：

```text
test(1).zip
结果统计.xlsx
```

整理数据：

```bash
py -3 -m src.prepare_local_dataset --overwrite
```

生成：

```text
datasets/local_colm/
  gt_counts.json
  images/test/
```

## 运行检测

默认会使用 `models/color_lr.pkl` 做颜色后处理。如果模型文件不存在，会自动退回纯 HSV/CV 结果。

```bash
py -3 -m src.vertical_lane_pipeline ^
  --source datasets/local_colm/images/test ^
  --gt-xlsx 结果统计.xlsx ^
  --out runs/vertical_lane_predictions.json ^
  --counts-out runs/vertical_lane_counts.csv ^
  --metrics-out runs/vertical_lane_metrics.json ^
  --report-xlsx runs/vertical_lane_report.xlsx ^
  --save-vis runs/vertical_lane_vis
```

不用颜色模型：

```bash
py -3 -m src.vertical_lane_pipeline --no-color-model
```

## 手动标注

如果需要给图片补充逐条 GT 车道线，用 OpenCV 标注工具：

```bash
py -3 -m src.manual_annotate_lanes ^
  --image-dir datasets/local_colm/images/test ^
  --label-dir datasets/local_colm/labels/test
```

从某张图开始：

```bash
py -3 -m src.manual_annotate_lanes --start 37.jpg
```

快捷键：

```text
鼠标左键     添加一个点
鼠标右键     结束当前车道线
Enter        结束当前车道线
w            当前类别切到白线
y            当前类别切到黄线
u            撤销当前线的最后一个点
z            撤销当前未完成线；如果没有未完成线，则删除上一条已标线
d            清空当前图片所有标注
s            保存当前图片标注
n / p        保存并切到下一张 / 上一张
q / Esc      保存当前图片并退出
```

保存格式是 `datasets/local_colm/labels/test/*.txt`，每行一条车道线：

```text
class_id x1 y1 x2 y2 ...
```

坐标是 0-1 归一化折线点；`0` 表示白线，`1` 表示黄线。这个格式可以被现有 `src.evaluate_lane_metrics` 读取，用来做逐条线角度匹配。

## 针对当前误差的优化点

- 图中黄线被画成白线：颜色模型现在会对所有候选二次判断，不只过滤黄线，也会把白色候选纠正成黄线。
- 图中多画线：黄线候选会经过颜色模型低置信过滤，并在后处理阶段做同类去重和数量上限约束。
- 图中定位不贴合：候选段聚合后会输出 `curve_points`，可视化优先画二次曲线；直线端点仍保留在 JSON 中，便于数量统计。

## 输出文件

- `runs/vertical_lane_predictions.json`：逐图、逐线检测结果。
- `runs/vertical_lane_counts.csv`：每张图检测到的白线/黄线数量。
- `runs/vertical_lane_metrics.json`：总体指标和每图计数。
- `runs/vertical_lane_report.xlsx`：可直接提交/查看的统计表。
- `runs/vertical_lane_vis/`：可视化图片。

## 评估说明

`结果统计.xlsx` 只有每张图的白线/黄线数量，没有逐条线的端点或多边形标注。因此当前“正确数”只能按每图每类的数量口径计算：

```text
correct = min(pred_count, gt_count)
```

如果要严格执行“偏离 15 度以内算准确”，需要给每条 GT 车道线补充端点、多点折线或 mask 标注，然后用角度差、中心距离和 IoU 做逐条匹配。

## 可继续提升的方向

- CV：增加消失点约束，过滤穿过车辆、箭头、斑马线的错误线段。
- ML：用当前候选的 HSV/Lab/几何特征训练二分类或三分类模型，区分白线、黄线和非车道线。
- 神经网络：用 UFLD、SCNN、LaneATT 或 CLRNet 一类 lane detector 做端到端车道线定位，再单独加颜色分类头。
