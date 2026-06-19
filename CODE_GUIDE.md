# 车道线检测与颜色分类项目 — 代码完全指南

> 本指南面向神经网络课程初学者，对 `/home/xuhaozhe/classify/src/` 中所有 12 个 Python 文件进行逐行中文注释。
> 阅读顺序建议：classes.py → geometry.py → xlsx_counts.py → color_classifier.py → prepare_local_dataset.py → generate_pseudo_labels.py → refine_labels_from_gt.py → train_color_classifier.py → train_yolo.py → predict_yolo_lane.py → evaluate_lane_metrics.py → apply_count_constraints.py

---

## 项目整体流程

```
原始数据(ZIP/Excel标注)
    │
    ▼
prepare_local_dataset.py    解压、划分训练/验证/测试集、转换标签
    │
    ▼
generate_pseudo_labels.py    Canny边缘+Hough变换 → 生成伪标签(弱监督训练用)
    │
    ▼
train_yolo.py                训练YOLO分割模型检测车道线
    │
    ▼
predict_yolo_lane.py         推理 → 检测车道线 + 判断颜色(白/黄)
    │
    ├── color_classifier.py   HSV规则 / 机器学习分类器判断颜色
    │
    ▼
evaluate_lane_metrics.py     评估精度/召回率/F1
    │
    ▼
apply_count_constraints.py   用量级标注(GT count)约束预测结果(后处理)
    │
    ▼
refine_labels_from_gt.py     用count约束后的结果生成高质量训练标签
    │
    ▼
train_color_classifier.py    训练学习的颜色分类器（替代HSV规则）
```

---

## 1. classes.py

### 功能概述

定义车道线检测中用到的类别常量（白线、黄线、车道线、路面等），以及类别名称的规范化工具函数。这个文件是整个项目的"词典"，所有其他文件都依赖它来统一类别名称的表示。

### 完整源码与注释

```python
# ============================================================
# classes.py — 类别常量定义与名称规范化工具
# ============================================================
# 这个文件虽然很短，但它是整个项目的"词汇表"。
# 所有其他文件都通过 import .classes 来引用白线/黄线等类别名称。
# 这样做的最大好处是：如果有一天要修改类别名称的字符串表示，
# 只需要改这一个文件，而不用去翻遍所有代码。

from __future__ import annotations
# ↑ Python 3.7+ 的特性：允许在类型注解中使用字符串形式的类名，
# 比如 'LineInstance' 而不是 from ... import LineInstance。
# 这可以避免循环导入（circular import）的问题。

from typing import Any, Mapping
# ↑ 导入类型提示工具。Any 表示"任何类型"，Mapping 表示"映射类型（如字典）"。
# 类型提示不会影响运行，但能让 IDE 提供更好的自动补全和错误检查。


# --- 类别常量 ---
# 整个项目中，表示"类别"的字符串只用下面这5个常量值。
# 千万不要在代码中直接写 "white_lane" 这样的字符串字面量，
# 而应该使用 WHITE 这个常量。这样如果以后要改名字，只改这里就行。

WHITE = "white_lane"        # 白色车道线
YELLOW = "yellow_lane"      # 黄色车道线
LANE_LINE = "lane_line"     # 车道线（不区分颜色）
ROAD = "road_surface"       # 路面
UNKNOWN = "unknown"         # 未知类别

# EVAL_CLASSES 定义了"评估时关心的类别"。
# 注意这里只包含 WHITE 和 YELLOW，不包含 LANE_LINE。
# 这意味着：在计算评估指标（precision/recall）时，只考虑区分了颜色的线，
# 那些没有颜色信息的 lane_line 不会被计入指标。
EVAL_CLASSES = (WHITE, YELLOW)


# --- 名称规范化的映射表 ---
# 下面三个集合用于将各种可能的输入名称映射到标准类别名。
# 为什么要这么做？因为不同的标注工具或数据源可能使用不同的命名约定，
# 比如有人写 "white_lane"，有人写 "white_line"，还有人写 "bai"（拼音）。
# 通过 normalize_class_name() 函数，我们可以兼容所有这些写法。

# 所有可能表示"白色车道线"的名称
_WHITE_NAMES = {
    "white",
    "white_lane",
    "white_line",
    "lane_white",
    "line_white",
    "bai",          # 中文拼音
    "bai_line",
}

# 所有可能表示"黄色车道线"的名称
_YELLOW_NAMES = {
    "yellow",
    "yellow_lane",
    "yellow_line",
    "lane_yellow",
    "line_yellow",
    "huang",        # 中文拼音
    "huang_line",
}

# 所有可能表示"车道线（不分颜色）"的名称
_LANE_NAMES = {
    "lane",
    "lane_line",
    "line",
    "road_line",
    "marking",
    "lane_marking",
}

# 所有可能表示"路面"的名称
_ROAD_NAMES = {
    "road",
    "road_surface",
    "surface",
    "pavement",
}


def normalize_class_name(name: Any) -> str:
    """
    将任何形式的类别名称规范化到标准类别名。

    设计思路：在实际项目中，标签来源可能五花八门（不同标注平台、不同语言），
    我们不可能要求所有数据都使用完全一致的命名。这个函数作为一种"容错机制"，
    将各种变体统一映射为常量。如果不在已知映射中，就返回原始文本（或 UNKNOWN）。

    Parameters:
        name: 可以是字符串、数字或其他类型

    Returns:
        标准类别名字符串（WHITE, YELLOW, LANE_LINE, ROAD, UNKNOWN 或原文本）
    """
    # 第一步：转成字符串、去掉首尾空格、转小写、统一分隔符
    text = str(name).strip().lower().replace("-", "_").replace(" ", "_")

    # 检查是否匹配已知的白线名称
    if text in _WHITE_NAMES:
        return WHITE

    # 检查是否匹配已知的黄线名称
    if text in _YELLOW_NAMES:
        return YELLOW

    # 检查是否匹配已知的车道线（不分颜色）名称
    if text in _LANE_NAMES:
        return LANE_LINE

    # 检查是否匹配已知的路面名称
    if text in _ROAD_NAMES:
        return ROAD

    # 没有匹配：如果原文本为空则返回 UNKNOWN，否则返回原文本
    return text or UNKNOWN


def names_from_yaml(raw_names: Any) -> dict[int, str]:
    """
    从 YAML 配置文件中读取类别名称映射。

    在 YOLO 的训练配置中，names 字段可以有两种格式：
    1. 字典格式：{0: "person", 1: "car"}
    2. 列表格式：["person", "car"]

    这个函数统一处理这两种情况，确保输出总是 {id: name} 字典。

    Parameters:
        raw_names: YAML 中读出的 names 字段

    Returns:
        {class_id: class_name} 的字典
    """
    if raw_names is None:
        return {}  # 没有类别信息，返回空字典

    if isinstance(raw_names, Mapping):
        # 已经是字典格式：{0: "white_lane", 1: "yellow_lane"}
        # 确保 key 是 int 类型（YAML 可能读出字符串或整数）
        return {int(k): str(v) for k, v in raw_names.items()}

    if isinstance(raw_names, (list, tuple)):
        # 列表格式：["white_lane", "yellow_lane"]
        # 索引就是类别 ID
        return {idx: str(name) for idx, name in enumerate(raw_names)}

    # 不支持的格式，抛出异常
    raise TypeError(f"Unsupported names format: {type(raw_names)!r}")


def class_id_to_name(class_id: int, names: Mapping[int, str] | None) -> str:
    """
    将 YOLO 输出的类别 ID 转换为规范化的类别名。

    这是 YOLO 推理结果解析的桥梁：
    - YOLO 模型输出的是类别 ID（如 0, 1）
    - 我们需要将其转换为标准类别名（如 "white_lane"）

    Parameters:
        class_id: YOLO 输出的类别 ID
        names:    {class_id: class_name} 的映射字典

    Returns:
        规范化的类别名字符串
    """
    if names and class_id in names:
        # 先查表得到原始名称，再规范化
        return normalize_class_name(names[class_id])
    # 查不到就返回 ID 的字符串形式
    return str(class_id)


def is_eval_class(name: str) -> bool:
    """
    判断一个类别是否属于"评估类别"（即白线或黄线）。

    这个函数被 evaluate_lane_metrics.py 调用，
    用于过滤出那些需要参与评估的检测结果。
    非评估类别的预测（如 LANE_LINE、ROAD）不会计入指标计算。

    Parameters:
        name: 类别名字符串

    Returns:
        是否属于 EVAL_CLASSES
    """
    return normalize_class_name(name) in EVAL_CLASSES
```

### 关键设计决策

1. **为什么用常量而不是字符串字面量？** 为了可维护性。如果有一天要将 "white_lane" 改为 "white_lane_marking"，只需要改一行代码。
2. **为什么要做名称规范化？** 现实中的数据标注来源复杂，不同工具使用不同的命名习惯。规范化是一种防御性编程实践。
3. **为什么 EVAL_CLASSES 只包含 WHITE 和 YELLOW？** 评估时，我们需要知道颜色分类是否正确。如果检测结果是 "lane_line"（未分颜色），无法判断颜色分类的准确性。

---

## 2. geometry.py

### 功能概述

提供车道线的几何表示（LineInstance 数据类）以及各种几何计算工具：角度差、IoU、边界框中心距离、通过 SVD 拟合直线、YOLO 标签解析等。这个文件是"几何工具箱"，被 predict_yolo_lane.py 和 evaluate_lane_metrics.py 大量调用。

### 完整源码与注释

```python
# ============================================================
# geometry.py — 车道线几何表示与计算工具
# ============================================================
# 这个文件包含了两类核心内容：
# 1. LineInstance 数据类：表示一条检测到的车道线（类别、置信度、角度、端点、边界框等）
# 2. 各种几何计算函数：角度差、IoU、直线拟合、坐标变换等
#
# 交叉文件依赖：
# - 被 predict_yolo_lane.py 调用（推理时构建 LineInstance）
# - 被 evaluate_lane_metrics.py 调用（评估时进行匹配）

from __future__ import annotations
# ↑ 允许在类型注解中使用尚未定义的类名

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
# ↑ numpy 是 Python 科学计算的核心库。几乎所有涉及数值计算的地方都会用到。
# 这里的 np.ndarray 是 NumPy 的多维数组类型，比 Python 列表快很多。

from .classes import class_id_to_name, is_eval_class
# ↑ 相对导入，导入同一包（src 目录）下的 classes.py 中的函数


# ============================================================
# 第1部分：LineInstance 数据类
# ============================================================

@dataclass
class LineInstance:
    """
    表示一条检测到的车道线实例。

    @dataclass 是 Python 3.7+ 的特性，它会自动生成 __init__、__repr__、__eq__ 等方法。
    也就是说，我们不需要手动写 __init__(self, cls, conf, ...)，数据类会自动处理好。

    为什么用 dataclass 而不直接用 dict？
    - 固定的字段结构：不容易写错 key 名称
    - 类型注解：IDE 可以自动补全
    - 可以定义方法（如 to_dict()、from_dict()）
    - 性能比 dict 略好
    """

    cls: str                # 类别名（如 "white_lane"、"yellow_lane"）
    conf: float             # 置信度（0~1），表示模型对这个检测结果的把握有多大
    angle_deg: float        # 线段角度（度），范围 0~180。0度=水平，90度=垂直
    endpoints: list[list[float]]    # 线段端点坐标 [[x1, y1], [x2, y2]]
    bbox: list[float]       # 边界框 [x1, y1, x2, y2]（左上角和右下角的坐标）
    source_class: str | None = None     # YOLO 原始输出的类别（未经过颜色分类器处理前的类别）
    color_score: float | None = None    # 颜色分类的置信度分数
    white_fraction: float | None = None # 在检测区域内，被判定为白色的像素比例
    yellow_fraction: float | None = None # 在检测区域内，被判定为黄色的像素比例

    def to_dict(self) -> dict:
        """
        将 LineInstance 转换为字典格式，用于 JSON 序列化。

        为什么需要这个函数？因为 JSON 只能序列化基本类型（dict、list、str、int、float、bool、None），
        不能直接序列化 dataclass 对象。所以在保存到 JSON 文件之前，需要先转成字典。

        注意：只有非 None 的可选字段才会被包含在输出字典中，这样可以减小 JSON 文件体积。
        """
        # 必填字段：所有 LineInstance 都必须有这些信息
        data = {
            "class": self.cls,
            "conf": float(self.conf),
            "angle_deg": float(self.angle_deg),
            "endpoints": self.endpoints,
            "bbox": self.bbox,
        }
        # 可选字段：只在有值时添加
        if self.source_class is not None:
            data["source_class"] = self.source_class
        if self.color_score is not None:
            data["color_score"] = float(self.color_score)
        if self.white_fraction is not None:
            data["white_fraction"] = float(self.white_fraction)
        if self.yellow_fraction is not None:
            data["yellow_fraction"] = float(self.yellow_fraction)
        return data

    @classmethod
    def from_dict(cls, data: Mapping) -> "LineInstance":
        """
        从字典反序列化为 LineInstance 对象。

        @classmethod 表示这是类方法，调用方式为 LineInstance.from_dict(data)。
        与之相对的是实例方法（如 to_dict），调用方式为 instance.to_dict()。

        这个函数是 to_dict 的逆操作，用于从 JSON 文件中读取数据。
        """
        return cls(
            cls=str(data["class"]),          # 类别名
            conf=float(data.get("conf", 1.0)),  # 置信度，如果缺失则默认为 1.0
            angle_deg=float(data["angle_deg"]), # 角度
            endpoints=[[float(v) for v in pt] for pt in data["endpoints"]],  # 端点列表
            bbox=[float(v) for v in data["bbox"]],  # 边界框
            source_class=data.get("source_class"),    # 以下都是可选字段，用 .get() 获取
            color_score=data.get("color_score"),
            white_fraction=data.get("white_fraction"),
            yellow_fraction=data.get("yellow_fraction"),
        )


# ============================================================
# 第2部分：角度和几何计算工具
# ============================================================

def angle_diff_deg(a: float, b: float) -> float:
    """
    计算两条线段的角度差（考虑了角度的周期性）。

    为什么需要特殊处理？因为角度是周期性的：
    - 0度 和 180度 实际上是同一条线（水平线）
    - 170度 和 10度 的差是 20度，而不是 160度

    数学原理：
    1. (a - b) % 180 将差值映射到 [0, 180) 范围
    2. min(diff, 180 - diff) 取两个方向中较小的角度差

    Parameters:
        a, b: 两个角度（度）

    Returns:
        最小的角度差（0~90度）
    """
    diff = abs((a - b) % 180.0)
    return min(diff, 180.0 - diff)


def bbox_iou_xyxy(a: Iterable[float], b: Iterable[float]) -> float:
    """
    计算两个边界框的 IoU（Intersection over Union，交并比）。

    IoU 是目标检测中最核心的评估指标之一：
    IoU = 两个框的交集面积 / 两个框的并集面积

    取值范围：0（完全不重叠）~ 1（完全重合）。

    这个函数被 evaluate_lane_metrics.py 调用，用于判断检测结果和真实标注是否匹配。

    Parameters:
        a, b: 边界框，格式为 [x1, y1, x2, y2]

    Returns:
        IoU 值（float）
    """
    # 解析坐标
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]

    # 计算交集矩形的坐标
    # 交集左上角 = 两个框左上角的最大值
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    # 交集右下角 = 两个框右下角的最小值
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    # 计算交集面积（如果没有重叠就为0）
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih

    # 计算两个框各自的面积
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    # 并集 = 面积A + 面积B - 交集
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0  # 防止除以零

    return inter / union


def bbox_center(bbox: Iterable[float]) -> tuple[float, float]:
    """
    计算边界框的中心点坐标。

    中心点 = ((x1 + x2) / 2, (y1 + y2) / 2)
    也就是矩形两个对角点的中点。

    为什么需要这个？在匹配预测和真实标注时，中心点距离是一个重要的判断依据。
    """
    x1, y1, x2, y2 = [float(x) for x in bbox]
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def center_distance(a: Iterable[float], b: Iterable[float]) -> float:
    """
    计算两个边界框的中心点距离（欧氏距离）。

    使用 math.hypot 来计算 √(Δx² + Δy²)，它比手动计算更精确且不易溢出。

    这个距离被用于匹配算法中：距离越近，越可能是同一个车道线。
    """
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return math.hypot(ax - bx, ay - by)


# ============================================================
# 第3部分：从分割掩码(mask)中提取直线参数
# ============================================================

def points_from_mask(mask: np.ndarray, max_points: int = 20000) -> np.ndarray:
    """
    从二值掩码（binary mask）中提取所有前景像素的坐标。

    分割模型输出的掩码是一个二维数组，每个像素的值表示属于目标的概率。
    这个函数找到所有"前景"像素的位置，返回它们的坐标列表。

    参数 max_points 有什么用？如果图像很大，前景像素可能很多（比如几十万），
    处理所有点会非常慢。限制最大点数可以保证性能。

    Parameters:
        mask: 二值掩码 (H x W)，True 表示前景（车道线区域）
        max_points: 最大采样点数

    Returns:
        N x 2 的数组，每行是一个前景像素的 (x, y) 坐标
    """
    # np.where 返回所有满足条件的索引
    # mask.astype(bool) 确保输入是布尔类型
    ys, xs = np.where(mask.astype(bool))

    if len(xs) == 0:
        # 没有前景像素，返回空数组
        return np.empty((0, 2), dtype=np.float32)

    # 将 (xs, ys) 组合成 N x 2 的坐标数组
    points = np.column_stack([xs, ys]).astype(np.float32)

    # 如果点数超过限制，进行降采样
    if len(points) > max_points:
        # 比如有 50000 个点，max_points=20000，step = 50000//20000 = 2
        # 也就是每隔一个点取一个，这样大约取 25000 个点
        step = max(1, len(points) // max_points)
        points = points[::step]  # ::step 是 Python 的切片语法，表示每隔 step 取一个

    return points


def fit_line_from_points(points: np.ndarray) -> tuple[float, list[list[float]], list[float]] | None:
    """
    通过一组点用 SVD（奇异值分解）拟合一条直线。

    这是本项目的核心几何算法之一。
    给定一组散点（车道线上的像素），找到最能代表这些点的直线。

    数学原理：
    我们使用"主成分分析（PCA）"的思想：
    1. 将数据中心化（减去均值）
    2. SVD 分解得到主方向
    3. 主方向就是直线的方向
    4. 将点投影到主方向上，得到端点

    SVD（奇异值分解）是线性代数中非常重要的矩阵分解方法：
    X = U * Σ * V^T
    其中 V 的第一行就是数据方差最大的方向（主成分）。

    Returns:
        (角度, 端点坐标列表, 边界框) 或 None（点太少）
    """
    if points is None or len(points) < 2:
        # 至少需要2个点才能确定一条直线
        return None

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    # ↑ 确保输入是 N x 2 的形状

    # 计算边界框（所有点的最小和最大坐标）
    x1, y1 = np.min(pts, axis=0)   # axis=0 表示按列（特征）取最小值
    x2, y2 = np.max(pts, axis=0)
    bbox = [float(x1), float(y1), float(x2), float(y2)]

    # 特殊情况：恰好只有两个点
    if len(pts) == 2:
        p0, p1 = pts
        vx, vy = p1 - p0   # 方向向量
        if abs(float(vx)) + abs(float(vy)) < 1e-6:
            return None  # 两个点重合，无法确定方向
        # atan2 计算方向角（弧度），然后转换为度
        angle = math.degrees(math.atan2(float(vy), float(vx))) % 180.0
        return angle, [[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]], bbox

    # 一般情况：通过 SVD 拟合直线
    # 第一步：数据中心化（减去均值）
    origin = np.mean(pts, axis=0)  # 所有点的中心
    centered = pts - origin        # 将点平移到原点附近

    # 第二步：SVD 分解
    try:
        # full_matrices=False 表示计算经济型分解（更高效）
        # U: 左奇异向量, S: 奇异值（对角矩阵的对角线）, Vh: 右奇异向量的转置
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None  # SVD 可能失败（比如所有点重合）

    # Vh 的第一行是主方向（数据方差最大的方向）
    direction = vh[0].astype(np.float32)
    vx, vy = float(direction[0]), float(direction[1])

    if abs(vx) + abs(vy) < 1e-6:
        return None  # 方向向量几乎为零，无法确定方向

    # 第三步：将所有点投影到主方向上
    # t = dot(centered, direction) 表示每个点在主方向上的"位置"
    t = centered @ direction  # @ 是 NumPy 的矩阵乘法

    # 在投影轴上找最小和最大的两个点作为端点
    p_start = origin + direction * float(np.min(t))
    p_end = origin + direction * float(np.max(t))

    # 计算直线的角度
    angle = math.degrees(math.atan2(float(vy), float(vx))) % 180.0
    endpoints = [
        [float(p_start[0]), float(p_start[1])],
        [float(p_end[0]), float(p_end[1])],
    ]
    return angle, endpoints, bbox


def line_from_bbox_xyxy(bbox: Iterable[float]) -> tuple[float, list[list[float]], list[float]]:
    """
    当无法从掩码中拟合直线时，用边界框生成一个简化的直线表示。

    这个函数是 fit_line_from_points 的备用方案。
    判断逻辑很简单：
    - 如果框是竖长的（高度 ≥ 宽度），认为是一条竖线，取中轴线
    - 如果框是横宽的（宽度 > 高度），认为是一条横线，取中轴线

    这样即使没有分割掩码，我们至少能有一个粗略的直线表示。

    Returns:
        (角度, 端点, 边界框)
    """
    x1, y1, x2, y2 = [float(x) for x in bbox]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    if height >= width:
        # 竖直线：取水平中心线，角度 90 度
        cx = (x1 + x2) * 0.5
        angle = 90.0
        endpoints = [[cx, y1], [cx, y2]]
    else:
        # 水平线：取垂直中心线，角度 0 度
        cy = (y1 + y2) * 0.5
        angle = 0.0
        endpoints = [[x1, cy], [x2, cy]]

    return angle, endpoints, [x1, y1, x2, y2]


# ============================================================
# 第4部分：YOLO 坐标格式转换
# ============================================================

def polygon_to_points(values: list[float], image_width: int, image_height: int) -> np.ndarray:
    """
    将 YOLO 分割格式的多边形坐标（归一化的扁平列表）转换为像素坐标数组。

    YOLO 分割标签格式：
    - 所有坐标都是相对于图像宽高的比例（归一化到 [0, 1]）
    - 存储为一个扁平列表：[x1, y1, x2, y2, x3, y3, ...]
    - 例如 [0.5, 0.3, 0.6, 0.4, 0.4, 0.4] 表示一个三角形的三个顶点

    这个函数将归一化坐标转换为实际像素坐标：
    pixel_x = norm_x * image_width
    pixel_y = norm_y * image_height
    """
    # reshape(-1, 2) 将扁平列表变为 N x 2 的数组
    coords = np.asarray(values, dtype=np.float32).reshape(-1, 2)

    # 反归一化
    coords[:, 0] *= float(image_width)    # x 坐标
    coords[:, 1] *= float(image_height)   # y 坐标

    return coords


def bbox_yolo_to_xyxy(values: list[float], image_width: int, image_height: int) -> list[float]:
    """
    将 YOLO 边界框格式（中心点+宽高）转换为 xyxy 格式（左上角+右下角）。

    YOLO 边界框格式：
    - (xc, yc) = 边界框中心点坐标（归一化比例）
    - (bw, bh) = 边界框的宽和高（归一化比例）
    - 例如 [0.5, 0.5, 0.2, 0.3] 表示图像中心一个宽 20%、高 30% 的框

    转换为 xyxy 格式：
    - x1 = xc - bw/2
    - y1 = yc - bh/2
    - x2 = xc + bw/2
    - y2 = yc + bh/2
    """
    xc, yc, bw, bh = values
    # 反归一化
    xc *= image_width
    yc *= image_height
    bw *= image_width
    bh *= image_height

    return [
        float(xc - bw / 2.0),   # x1: 左上角 x
        float(yc - bh / 2.0),   # y1: 左上角 y
        float(xc + bw / 2.0),   # x2: 右下角 x
        float(yc + bh / 2.0),   # y2: 右下角 y
    ]


# ============================================================
# 第5部分：读取 YOLO 标签文件
# ============================================================

def read_yolo_label_file(
    label_path: Path,               # YOLO 标签文件路径
    image_width: int,               # 对应图像的宽度
    image_height: int,              # 对应图像的高度
    names: Mapping[int, str] | None, # 类别 ID 到名称的映射
    *,
    keep_only_eval_classes: bool = True,  # 是否只保留评估类别的实例
) -> list[LineInstance]:
    """
    读取一个 YOLO 格式的标签文件，返回 LineInstance 列表。

    YOLO 标签文件格式（每行一个实例）：
    - 边界框：<class_id> <xc> <yc> <bw> <bh>
    - 分割多边形：<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>

    这个函数被 evaluate_lane_metrics.py 调用，用于加载真实标注（Ground Truth）。

    设计决策：为什么需要 keep_only_eval_classes 参数？
    因为真实标注可能包含路面、路肩等其他类别，但在评估车道线检测时，
    我们只关心白线和黄线。这个参数让我们可以过滤掉不相关类别的标注。
    """
    instances: list[LineInstance] = []
    if not label_path.exists():
        # 文件不存在，返回空列表（有些图片可能没有标注）
        return instances

    # enumerate 从第1行开始计数，方便出错时定位
    for line_no, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue  # 跳过空行

        parts = stripped.split()
        try:
            # 第一个值是类别 ID，后面是坐标值
            class_id = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_no}: invalid YOLO label line: {raw!r}") from exc

        # 将类别 ID 转换为规范化的类别名
        class_name = class_id_to_name(class_id, names)

        # 如果设置为只保留评估类别，则过滤掉非评估类别
        if keep_only_eval_classes and not is_eval_class(class_name):
            continue

        # 根据坐标数量判断是边界框还是分割多边形
        if len(values) == 4:
            # 边界框格式：4个值 (xc, yc, bw, bh)
            bbox = bbox_yolo_to_xyxy(values, image_width, image_height)
            # 从边界框生成直线表示
            angle, endpoints, bbox = line_from_bbox_xyxy(bbox)
        elif len(values) >= 4 and len(values) % 2 == 0:
            # 分割多边形格式：每2个值一组坐标
            points = polygon_to_points(values, image_width, image_height)
            # 从多边形点拟合直线
            fitted = fit_line_from_points(points)
            if fitted is None:
                continue  # 拟合失败，跳过
            angle, endpoints, bbox = fitted
        else:
            # 既不是边界框也不是多边形，抛出异常
            raise ValueError(
                f"{label_path}:{line_no}: expected YOLO bbox or segmentation polygon, got {len(values)} values"
            )

        # 构建 LineInstance
        instances.append(
            LineInstance(
                cls=class_name,
                conf=1.0,  # 真实标注的置信度设为 1.0（100% 确定）
                angle_deg=angle,
                endpoints=endpoints,
                bbox=bbox,
                source_class=class_name,
            )
        )

    return instances


def find_image_by_stem(image_dir: Path, stem: str) -> Path | None:
    """
    在目录中查找与给定文件名（不含扩展名）匹配的图像文件。

    为什么需要这个函数？因为 YOLO 标签文件是 .txt 扩展名，图像可能是 .jpg、.png 等。
    给定标签的文件名（不含扩展名），我们需要找到对应的图像文件。

    查找策略：
    1. 先在目录中直接查找常见扩展名（性能优先）
    2. 如果找不到，再递归搜索所有子目录（容错）

    Parameters:
        image_dir: 图像目录
        stem: 文件名（不含扩展名）

    Returns:
        图像文件路径，如果找不到则返回 None
    """
    # 常见图像扩展名（按频率排序，以便快速命中）
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    # 策略1：直接在目录中查找（最快）
    for ext in extensions:
        direct = image_dir / f"{stem}{ext}"
        if direct.exists():
            return direct

    # 策略2：递归搜索所有子目录（较慢但更全面）
    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions and path.stem == stem:
            return path

    # 都找不到
    return None
```

### 关键概念讲解

1. **SVD（奇异值分解）拟合直线**：给定一堆散点，找到最"代表"这些点的直线。核心思想是 PCA（主成分分析）：数据方差最大的方向就是直线的方向。使用 `np.linalg.svd` 比直接计算协方差矩阵更数值稳定。

2. **IoU（交并比）**：目标检测中最基本的评估指标。两个边界框的重叠程度 = 交集面积 / 并集面积。值越大表示检测越准。

3. **角度周期性**：直线的方向是模 180 度的（而不是 360 度），因为一条线从两端看角度差 180 度。`angle_diff_deg` 函数正确处理了这一点。

4. **YOLO 坐标归一化**：YOLO 所有坐标都归一化到 [0, 1] 范围，与实际图像尺寸无关。这样做的好处是模型不依赖于输入图像的分辨率。

---

## 3. xlsx_counts.py

### 功能概述

读取 Excel 文件中的车道线数量统计（文件名、白线数、黄线数），并提供 JSON 格式的读写函数。Excel 文件是 .xlsx 格式（Office Open XML），本质是一个 ZIP 压缩包，内部包含 XML 文件。这个函数直接解析 ZIP 中的 XML，不依赖任何 Excel 库。

### 完整源码与注释

```python
# ============================================================
# xlsx_counts.py — 读取 Excel 中的车道线数量统计
# ============================================================
# 本项目的训练数据包含一个 Excel 文件（结果统计.xlsx），其中人工标注了
# 每张图片中车道线的数量（总条数、白线数、黄线数）。
# 这个文件提供了解析这个 Excel 的工具函数。
#
# 为什么不用 pandas 或 openpyxl？为了减少依赖。
# .xlsx 文件本质上是 ZIP 压缩包，内部是 XML 格式。
# 我们直接用 Python 标准库（zipfile + xml.etree.ElementTree）解析，
# 不需要安装额外的 Excel 库。
#
# 交叉文件依赖：
# - 被 prepare_local_dataset.py 调用（准备数据集时读取标注）
# - 被 evaluate_lane_metrics.py 调用（评估时读取 GT 数量）
# - 被 apply_count_constraints.py 调用（约束预测时读取 GT 数量）
# - 被 refine_labels_from_gt.py 调用（生成标签时读取 GT 数量）

from __future__ import annotations

import json
import zipfile
import xml.etree.ElementTree as ET   # Python 标准库的 XML 解析器
from pathlib import Path
from typing import Any

from .classes import WHITE, YELLOW
# ↑ 引用标准类别名，确保和项目其他部分一致


# OOXML 命名空间前缀
# .xlsx 文件内部使用 XML 命名空间 http://schemas.openxmlformats.org/spreadsheetml/2006/main，
# 我们在查找 XML 元素时需要用这个命名空间前缀 'a'。
NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _shared_strings(zf: zipfile.ZipFile) -> list[str]:
    """
    读取 .xlsx 文件中的共享字符串表（Shared String Table）。

    Excel 的存储优化：如果有大量重复的字符串，Excel 会将这些字符串存入
    共享字符串表，然后在单元格中只存储一个索引。单元格类型 's' 表示
    该单元格的值是共享字符串表的索引。

    这个过程称为 SST（Shared String Table），是 OOXML 格式的重要特性。

    为什么需要这个函数？因为我们要读取"文件名"、"白线数"等字符串，
    它们可能存储在共享字符串表中。

    Parameters:
        zf: 打开的 ZIP 文件对象

    Returns:
        共享字符串列表，按索引访问
    """
    # 检查共享字符串表是否存在
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []  # 没有共享字符串，返回空列表

    # 解析 XML
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))

    values = []
    for item in root.findall("a:si", NS):
        # 每个 <si> 元素可能包含多个 <t> 元素（文本片段）
        # 将所有 <t> 元素的文本拼接起来
        values.append("".join(node.text or "" for node in item.findall(".//a:t", NS)))

    return values


def _cell_text(cell: ET.Element, shared: list[str]) -> str:
    """
    获取单元格的文本值。

    Excel 单元格有多种类型：
    - 's': 共享字符串类型，值存储在共享字符串表中
    - 'inlineStr': 内联字符串类型，XML 中直接包含文本
    - 其他类型：数值、日期等，直接读取 <v> 元素

    Parameters:
        cell: XML 单元格元素
        shared: 共享字符串表

    Returns:
        单元格的文本内容
    """
    cell_type = cell.attrib.get("t")  # 获取单元格类型属性
    value = cell.find("a:v", NS)      # 查找 <v> 元素（value 的缩写）

    if cell_type == "inlineStr":
        # 内联字符串：直接提取 <t> 元素中的文本
        return "".join(node.text or "" for node in cell.findall(".//a:t", NS)).strip()

    if value is None or value.text is None:
        return ""  # 没有值

    text = value.text.strip()
    if cell_type == "s" and text:
        # 共享字符串：text 是索引，从 shared 列表中取值
        return shared[int(text)].strip()

    return text  # 直接返回数值的字符串形式


def _column_name(ref: str) -> str:
    """
    提取单元格引用中的列名（字母部分）。

    在 Excel 中，单元格引用如 "A1"、"B2"、"AA10"。
    - 字母部分（A, B, AA）表示列
    - 数字部分（1, 2, 10）表示行

    例如：
    - "A1" → "A"
    - "B2" → "B"
    - "AA10" → "AA"

    Parameters:
        ref: 单元格引用字符串

    Returns:
        列名字母
    """
    return "".join(ch for ch in ref if ch.isalpha())


def _to_int(value: Any) -> int:
    """
    安全地将任意值转换为整数。

    为什么需要这个函数？Excel 单元格中的数字可能以浮点数的形式存储
    （即使是整数，Excel 也可能存为 3.0 而不是 3），
    所以需要先转 float 再转 int。

    Parameters:
        value: 任意值

    Returns:
        整数，转换失败返回 0
    """
    text = str(value).strip()
    if not text:
        return 0
    return int(float(text))  # 先转 float 再转 int

def read_count_xlsx(path: Path) -> dict[str, dict[str, int]]:
    """
    从 .xlsx 文件中读取车道线数量统计。

    Excel 表格的预期结构：
    - 第一行是表头（含"文件名"、"车道线数"、"白线数"、"黄线数"等列）
    - 后续每行是一条数据

    返回值格式：{文件名: {"lane_line": 总数, "white_lane": 白线数, "yellow_lane": 黄线数}}

    设计决策：为什么返回值用字典嵌套而不是列表？
    因为我们要通过文件名快速查找，字典的查找速度是 O(1)，列表是 O(n)。
    在后续的评估和约束步骤中，这种按文件名查找的操作非常频繁。

    Parameters:
        path: .xlsx 文件路径

    Returns:
        {文件名: {类别名: 数量}} 的字典
    """
    # .xlsx 文件是一个 ZIP 包，先用 zipfile 打开
    with zipfile.ZipFile(path) as zf:
        # 读取共享字符串表
        shared = _shared_strings(zf)

        # 找到第一个工作表的 XML 文件
        # 工作表文件位于 xl/worksheets/sheet1.xml, sheet2.xml, ...
        sheet_names = [
            name for name in zf.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        ]
        if not sheet_names:
            raise ValueError(f"No worksheet found in {path}")

        # 只读取第一个工作表
        root = ET.fromstring(zf.read(sheet_names[0]))

    # 表头映射：列字母 → 列名
    # 例如 {"A": "文件名", "B": "车道线数", "C": "白线数", "D": "黄线数"}
    header_by_col: dict[str, str] = {}

    # 结果数据：{文件名: {类别名: 数量}}
    rows: dict[str, dict[str, int]] = {}

    # 遍历所有行
    for row in root.findall(".//a:row", NS):
        # 收集当前行所有单元格的值
        row_values: dict[str, str] = {}
        for cell in row.findall("a:c", NS):
            ref = cell.attrib.get("r", "")  # 单元格引用，如 "A1"
            col = _column_name(ref)         # 提取列名，如 "A"
            if col:
                row_values[col] = _cell_text(cell, shared)

        if not row_values:
            continue  # 跳过空行

        # 如果是表头行（包含"文件名"列）
        if not header_by_col and any(value == "文件名" for value in row_values.values()):
            header_by_col = {col: value for col, value in row_values.items()}
            continue  # 表头行不包含数据，跳过

        # 如果没有找到表头，跳过所有行（直到表头出现）
        if not header_by_col:
            continue

        # 将列字母映射为列名
        values_by_header = {
            header_by_col[col]: value
            for col, value in row_values.items()
            if col in header_by_col
        }

        # 获取文件名
        filename = values_by_header.get("文件名", "").strip()
        if not filename:
            continue  # 跳过没有文件名的行

        # 读取各个计数值
        white = _to_int(values_by_header.get("白线数", 0))
        yellow = _to_int(values_by_header.get("黄线数", 0))
        # 如果"车道线数"列存在则用它，否则使用 white + yellow
        total = _to_int(values_by_header.get("车道线数", white + yellow))

        # 保存结果
        rows[filename] = {
            "lane_line": total,
            WHITE: white,
            YELLOW: yellow,
        }

    return rows


def write_count_json(counts: dict[str, dict[str, int]], path: Path) -> None:
    """
    将车道线数量统计写入 JSON 文件。

    JSON 格式便于后续程序读取，比 Excel 更轻量、更快速。

    ensure_ascii=False: 允许非 ASCII 字符（如中文文件名）正常存储
    indent=2: 美化输出，每层缩进 2 个空格，便于人工阅读
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")


def read_count_json(path: Path) -> dict[str, dict[str, int]]:
    """
    从 JSON 文件中读取车道线数量统计。

    兼容性处理：有些旧版本 JSON 文件可能使用 "white"/"yellow" 作为 key，
    而不是标准化的 "white_lane"/"yellow_lane"。这里做了兼容转换。
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(filename): {
            "lane_line": _to_int(values.get("lane_line", 0)),
            WHITE: _to_int(values.get(WHITE, values.get("white", 0))),   # 兼容旧格式
            YELLOW: _to_int(values.get(YELLOW, values.get("yellow", 0))), # 兼容旧格式
        }
        for filename, values in raw.items()
    }
```

### 关键概念讲解

1. **OOXML 格式**：.xlsx 文件是 Office Open XML 格式的缩写，本质上是一个 ZIP 包。内部包含了多个 XML 文件，分别存储工作簿结构、工作表数据、共享字符串、样式等信息。

2. **共享字符串表（SST）**：Excel 的一个优化机制。当一个工作簿中有大量重复的文本时（比如"文件名"这个字符串在每行都出现），Excel 只在 SST 中存一次，然后每个单元格只记录一个索引。这样可以显著减小文件体积。

3. **命名空间（XML Namespace）**：OOXML 使用 XML 命名空间来避免元素名称冲突。`NS = {"a": "..."}` 定义了一个前缀 'a' 指向命名空间 URI，在查找元素时使用 `a:row` 而不是 `row`。

---

## 4. color_classifier.py

### 功能概述

实现车道线颜色的分类逻辑（白线 vs 黄线）。提供三种方案：固定阈值 HSV、自适应阈值 HSV（根据光照条件动态调整）、以及机器学习分类器（逻辑回归）。这是项目中"区分白线和黄线"的核心算法文件。

### 完整源码与注释

```python
# ============================================================
# color_classifier.py — 车道线颜色分类（白线 vs 黄线）
# ============================================================
# 这个文件是项目中最核心的算法文件之一，实现了三种颜色分类方案：
#
# 方案1：固定阈值 HSV（classify_lane_color）
#   - 在 HSV 颜色空间设置固定阈值
#   - 简单、快速，但光照变化大时效果不稳定
#
# 方案2：自适应阈值 HSV（adaptive_classify_lane_color）
#   - 根据图像统计信息动态调整阈值
#   - 更鲁棒，适用于各种光照条件
#
# 方案3：机器学习分类器（learned_classify_lane_color）
#   - 使用 Logistic Regression 学习特征
#   - 最准确，但需要训练（用 train_color_classifier.py）
#
# 交叉文件依赖：
# - 被 predict_yolo_lane.py 调用（推理时判断颜色）
# - train_color_classifier.py 是它的训练脚本（训练 ML 版本）

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .classes import UNKNOWN, WHITE, YELLOW


# ============================================================
# 第1部分：ColorDecision 数据类
# ============================================================

@dataclass(frozen=True)
# ↑ frozen=True 表示这个数据类的实例是不可变的（immutable）。
# 创建后不能修改字段值。为什么？因为颜色分类结果应该是确定性的，
# 不可变性可以防止意外修改。

class ColorDecision:
    """
    颜色分类的决策结果。

    字段说明：
    - cls: 最终的分类结果（WHITE / YELLOW / UNKNOWN）
    - score: 分类置信度（0~1）
    - white_fraction: 被判定为白色的像素比例
    - yellow_fraction: 被判定为黄色的像素比例
    - valid_pixels: 有效像素数（排除了过暗的像素）

    white_fraction 和 yellow_fraction 可以同时非零（如果像素同时满足两个条件），
    但最终分类会按逻辑规则选择其中一个。
    """
    cls: str           # 分类结果
    score: float       # 置信度
    white_fraction: float  # 白色像素比例
    yellow_fraction: float # 黄色像素比例
    valid_pixels: int   # 有效像素数


# ============================================================
# 第2部分：辅助函数
# ============================================================

def _safe_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    """
    安全地处理掩码（mask），确保其形状与图像匹配。

    为什么需要这个函数？
    - 掩码可能为 None（如果没有分割掩码）
    - 掩码的形状可能与图像不匹配（如果来自不同分辨率的处理步骤）
    - 掩码可能是 uint8 类型（0 或 255）而不是 bool 类型

    Parameters:
        mask: 可选的二值掩码
        shape: 目标形状 (height, width)

    Returns:
        与 shape 匹配的布尔掩码
    """
    if mask is None:
        # 没有掩码：返回全 True 掩码（认为所有像素都有效）
        return np.ones(shape, dtype=bool)

    if mask.shape[:2] != shape:
        # 形状不匹配：用最近邻插值缩放掩码
        # 使用 INTER_NEAREST（最近邻插值）而不是默认的双线性插值，
        # 因为掩码是二值的，最近邻插值可以保持尖锐的边缘。
        import cv2
        mask = cv2.resize(mask.astype("uint8"), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

    return mask.astype(bool)


def _image_stats(image_bgr: np.ndarray) -> dict:
    """
    计算整张图像的全局统计信息，用于自适应阈值调整。

    核心思想：车道线主要出现在道路区域，而道路通常位于图像的下半部分。
    所以我们只关注图像的下半部分（road_region），这样：
    - 减少天空、树木等无关区域的干扰
    - 使统计信息更聚焦在路面环境

    计算的统计量：
    - median_v: 亮度中位数（对极端值不敏感）
    - p10_v: 亮度第10百分位（暗部）
    - p90_v: 亮度第90百分位（亮部）
    - mean_s: 平均饱和度
    - std_v: 亮度标准差
    - dyn_range: 动态范围（p90 - p10），反映图像对比度

    为什么用百分位数而不是最大最小值？
    因为最大/最小值容易受到噪声点和极端值的影响，不稳定。
    百分位数更鲁棒。

    Parameters:
        image_bgr: BGR 格式的 OpenCV 图像

    Returns:
        包含上述统计量的字典
    """
    import cv2

    h, w = image_bgr.shape[:2]
    # 只取下半部分（h//2 开始到结尾）
    road_region = image_bgr[h // 2:, :]
    if road_region.size == 0:
        road_region = image_bgr  # 容错：如果图像太小

    # 转换为 HSV 颜色空间
    hsv = cv2.cvtColor(road_region, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)  # V 通道（亮度）
    s = hsv[:, :, 1].astype(np.float32)  # S 通道（饱和度）

    # 计算亮度的统计量
    v_sorted = np.sort(v.ravel())  # 排序，方便计算百分位数
    n = len(v_sorted)

    median_v = float(np.median(v))      # 中位数
    p10_v = float(v_sorted[int(n * 0.10)])  # 第10百分位（暗部）
    p90_v = float(v_sorted[int(n * 0.90)])  # 第90百分位（亮部）
    mean_s = float(s.mean())             # 平均饱和度
    std_v = float(v.std())               # 亮度标准差
    dyn_range = p90_v - p10_v            # 动态范围（对比度）

    return {
        "median_v": median_v,
        "p10_v": p10_v,
        "p90_v": p90_v,
        "mean_s": mean_s,
        "std_v": std_v,
        "dyn_range": dyn_range,
    }


def _region_stats(image_bgr: np.ndarray, mask: np.ndarray) -> dict:
    """
    计算检测到的车道线区域及其周围区域的 HSV 统计信息。

    这个函数计算两个关键信息：
    1. 车道线本身的 HSV 统计（亮度、饱和度）
    2. 车道线与周围环境的对比度

    对比度的计算方法是：
    - 对 mask 做膨胀（dilate），得到 mask 的"周围区域"
    - 取周围区域的亮度中位数
    - 对比度 = 车道线亮度 - 周围亮度

    为什么需要对比度？因为一条白色车道线在明亮路面上的绝对亮度可能不高，
    但只要它比周围路面亮，仍然可以判定为白色。
    """
    import cv2

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)  # 拆分为三个通道

    # 车道线区域的 HSV 统计
    lane_v = v[mask].astype(np.float32)
    lane_s = s[mask].astype(np.float32)

    if lane_v.size == 0:
        return {"lane_median_v": 0, "lane_median_s": 0, "contrast_vs_surround": 0}

    lane_median_v = float(np.median(lane_v))
    lane_median_s = float(np.median(lane_s))

    # 计算周围区域的对比度
    # 膨胀操作：将 mask 扩大（kernel=15x15 的矩形）
    kernel = np.ones((15, 15), dtype=np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    # 周围区域 = 膨胀后的区域 - 原始 mask
    surround = dilated & ~mask

    if surround.sum() > 0:
        surround_v = v[surround].astype(np.float32)
        surround_median_v = float(np.median(surround_v))
        contrast = lane_median_v - surround_median_v  # 正数表示比周围亮
    else:
        contrast = 0

    return {
        "lane_median_v": lane_median_v,
        "lane_median_s": lane_median_s,
        "contrast_vs_surround": contrast,
    }


# ============================================================
# 第3部分：固定阈值 HSV 颜色分类
# ============================================================

def classify_lane_color(
    image_bgr: np.ndarray,             # BGR 图像（OpenCV 默认格式）
    mask: np.ndarray | None = None,     # 车道线区域掩码
    *,
    # 以下是默认阈值参数，通过大量实验调优得到
    min_value: int = 70,                # 最小亮度（过滤过暗的噪声）
    white_sat_max: int = 85,           # 白色最大饱和度（白色饱和度应该低）
    white_value_min: int = 155,        # 白色最小亮度（白色反射更多光）
    yellow_hue_min: int = 14,          # 黄色最小色相
    yellow_hue_max: int = 45,          # 黄色最大色相
    yellow_sat_min: int = 45,          # 黄色最小饱和度
    yellow_value_min: int = 90,        # 黄色最小亮度
    min_color_fraction: float = 0.04,  # 最少需要多大比例的像素才能判定颜色
) -> ColorDecision:
    """
    使用固定 HSV 阈值判断车道线颜色。

    HSV 颜色空间简介：
    - H（Hue，色相）：表示颜色的种类，用角度度量（0°~180°在 OpenCV 中）
      - 红色: 0° 附近
      - 黄色: 15°~45° 附近
      - 绿色: 45°~90° 附近
      - 蓝色: 90°~135° 附近
    - S（Saturation，饱和度）：颜色的纯度/鲜艳程度（0~255，越低越灰）
      - 白色饱和度很低（接近 0）
      - 黄色饱和度较高
    - V（Value，明度）：颜色的明亮程度（0~255，越高越亮）
      - 白色亮度很高（反射所有光）
      - 黄色亮度中等

    为什么用 HSV 而不是 RGB？
    在 RGB 空间中，"白色"和"黄色"的边界不明显，受光照影响很大。
    HSV 将颜色（H）和明暗（V）分离，更符合人类对颜色的感知，
    对光照变化也更鲁棒。
    """
    return _classify_impl(
        image_bgr, mask,
        min_value=min_value, white_sat_max=white_sat_max,
        white_value_min=white_value_min, yellow_hue_min=yellow_hue_min,
        yellow_hue_max=yellow_hue_max, yellow_sat_min=yellow_sat_min,
        yellow_value_min=yellow_value_min, min_color_fraction=min_color_fraction,
    )


# ============================================================
# 第4部分：自适应阈值 HSV 颜色分类
# ============================================================

def adaptive_classify_lane_color(
    image_bgr: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    min_color_fraction: float = 0.04,
) -> ColorDecision:
    """
    根据图像光照条件自适应调整 HSV 阈值进行分类。

    这是固定阈值版本的改进版，核心思路是：
    - 亮的场景 🡒 提高亮度和饱和度阈值（因为需要更亮才算白/黄）
    - 暗的场景 🡒 降低所有阈值（因为线本身就不亮）
    - 低对比度场景 🡒 更宽容（阴天、黄昏等）
    - 高对比度场景 🡒 依靠相对对比度而不是绝对亮度

    自适应策略是基于大量经验数据的启发式规则，不是机器学习。
    虽然不如 ML 版本准，但不需要训练，即开即用。

    Parameters:
        image_bgr: BGR 图像
        mask: 可选掩码
        min_color_fraction: 最小颜色像素比例

    Returns:
        ColorDecision 对象
    """
    if image_bgr.size == 0:
        return ColorDecision(UNKNOWN, 0.0, 0.0, 0.0, 0)

    # 计算图像全局统计信息
    stats = _image_stats(image_bgr)
    median_v = stats["median_v"]    # 亮度中位数
    mean_s = stats["mean_s"]        # 平均饱和度
    dyn_range = stats["dyn_range"]  # 动态范围

    # --- 根据场景亮度调整阈值 ---
    # brightness 因子：以 median_v=128 为基准
    # - 128 的中位数亮度被定义为"标准"场景（factor=1.0）
    # - 更暗（median_v < 128）则 factor < 1.0
    # - 更亮（median_v > 128）则 factor > 1.0
    # - clamp 到 [0.5, 1.8] 防止极端值
    brightness = np.clip(median_v / 128.0, 0.5, 1.8)

    # 白色亮度阈值：基准 140，每超出 1.0 的亮度因子增加 30
    # 亮的场景需要更高的亮度才能算"白色"
    base_white_v = int(np.clip(140 + 30 * (brightness - 1.0), 110, 190))

    # 黄色亮度阈值：同理，基准 75
    base_yellow_v = int(np.clip(75 + 25 * (brightness - 1.0), 50, 115))

    # 饱和度因子：以 mean_s=45 为基准
    # 阴天等低饱和度场景需要更宽容的阈值
    sat_factor = np.clip(mean_s / 45.0, 0.5, 1.5)

    # 黄色饱和度阈值：低饱和度场景降低要求
    base_yellow_s = int(np.clip(40 / max(sat_factor, 0.6), 20, 70))

    # 白色最大饱和度：低饱和度场景更宽容
    base_white_s_max = int(np.clip(90 * sat_factor, 55, 130))

    # 最小亮度阈值：基于动态范围
    # 动态范围大的场景有更多光照变化，需要提高最低亮度以过滤阴影
    base_min_v = int(np.clip(50 + 0.15 * dyn_range, 35, 85))

    # 低对比度场景（阴天、黎明等）：更加宽容
    if dyn_range < 40:
        base_min_v = max(30, base_min_v - 15)
        base_white_v = max(100, base_white_v - 20)

    # --- 局部区域精细调整 ---
    mask_bool = _safe_mask(mask, image_bgr.shape[:2])
    region_stats = _region_stats(image_bgr, mask_bool)
    lane_v = region_stats["lane_median_v"]
    contrast = region_stats["contrast_vs_surround"]
    lane_s = region_stats["lane_median_s"]

    # 如果车道线比周围明显亮，可能是白色（即使绝对亮度不高）
    if contrast > 30 and lane_v > 100:
        base_white_v = min(base_white_v, int(lane_v - 5))

    # 如果车道线饱和度很高，很可能是黄色
    if lane_s > 80:
        base_yellow_s = max(25, base_yellow_s - 15)

    # 使用调整后的阈值进行分类
    return _classify_impl(
        image_bgr, mask,
        min_value=base_min_v, white_sat_max=base_white_s_max,
        white_value_min=base_white_v, yellow_hue_min=14,
        yellow_hue_max=48, yellow_sat_min=base_yellow_s,
        yellow_value_min=base_yellow_v, min_color_fraction=min_color_fraction,
    )


# ============================================================
# 第5部分：核心分类逻辑
# ============================================================

def _classify_impl(
    image_bgr: np.ndarray,
    mask: np.ndarray | None,
    *,
    # 以下是可调阈值，让固定阈值和自适应阈值两个版本共用同一套核心逻辑
    min_value: int,
    white_sat_max: int,
    white_value_min: int,
    yellow_hue_min: int,
    yellow_hue_max: int,
    yellow_sat_min: int,
    yellow_value_min: int,
    min_color_fraction: float,
) -> ColorDecision:
    """
    核心分类逻辑。

    步骤：
    1. 转换到 HSV 空间
    2. 用掩码和最小亮度过滤出有效像素
    3. 分别统计白色像素和黄色像素的比例
    4. 根据比例和规则做最终决策

    决策规则：
    - 黄色占比 ≥ min_color_fraction 且 黄色 > 白色×1.2 → 黄色
    - 白色占比 ≥ min_color_fraction → 白色
    - 黄色 > 白色 → 黄色
    - 白色 > 0 → 白色（即使很少）
    - 否则 → UNKNOWN

    为什么黄色需要超过白色 1.2 倍才能判定？
    因为白色检测更宽松（低饱和度+高亮度），可能误检。
    黄色检测更严格（必须在特定色相范围内），一旦检测到可信度更高。
    所以当两者相近时，优先选择白色。
    """
    if image_bgr.size == 0:
        return ColorDecision(UNKNOWN, 0.0, 0.0, 0.0, 0)

    import cv2

    # 确保掩码有效
    mask_bool = _safe_mask(mask, image_bgr.shape[:2])

    # 转换到 HSV 空间
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)  # H: 色相, S: 饱和度, V: 亮度

    # 有效像素：在掩码内 且 亮度 >= min_value
    # 为什么要过滤低亮度像素？因为阴影区域的颜色信息不可靠
    valid = mask_bool & (v >= min_value)
    valid_pixels = int(valid.sum())

    # 没有有效像素，无法判断
    if valid_pixels == 0:
        return ColorDecision(UNKNOWN, 0.0, 0.0, 0.0, 0)

    # 白色条件：低饱和度 + 高亮度
    white_pixels = valid & (s <= white_sat_max) & (v >= white_value_min)

    # 黄色条件：色相在黄色范围 + 一定饱和度 + 一定亮度
    yellow_pixels = (
        valid
        & (h >= yellow_hue_min)
        & (h <= yellow_hue_max)
        & (s >= yellow_sat_min)
        & (v >= yellow_value_min)
    )

    # 计算比例
    white_fraction = float(white_pixels.sum() / valid_pixels)
    yellow_fraction = float(yellow_pixels.sum() / valid_pixels)

    # --- 决策逻辑 ---
    if yellow_fraction >= min_color_fraction and yellow_fraction > white_fraction * 1.2:
        # 黄色显著多于白色：判黄
        return ColorDecision(YELLOW, yellow_fraction, white_fraction, yellow_fraction, valid_pixels)

    if white_fraction >= min_color_fraction:
        # 白色达到最小比例：判白
        return ColorDecision(WHITE, white_fraction, white_fraction, yellow_fraction, valid_pixels)

    if yellow_fraction > white_fraction:
        # 黄色多于白色但都不够 min_color_fraction：还是判黄
        return ColorDecision(YELLOW, yellow_fraction, white_fraction, yellow_fraction, valid_pixels)

    if white_fraction > 0:
        # 只有少量白色像素：勉强判白
        return ColorDecision(WHITE, white_fraction, white_fraction, yellow_fraction, valid_pixels)

    # 什么都没有：未知
    return ColorDecision(UNKNOWN, 0.0, white_fraction, yellow_fraction, valid_pixels)


# ============================================================
# 第6部分：机器学习颜色分类器
# ============================================================
# 下面的代码实现了一个基于 Logistic Regression 的学习型颜色分类器。
# 它提取丰富的特征（HSV直方图、RGB统计、Lab色空间、对比度、形状等），
# 然后用训练好的模型预测颜色。
#
# 优势：比固定HSV规则更准确，能处理复杂光照条件。
# 缺点：需要先训练模型（用 train_color_classifier.py）。

# 全局变量，用于缓存已加载的模型和缩放器
# 为什么要缓存？因为模型加载需要读磁盘，如果每张图片都加载一次会非常慢。
_ml_model = None
_ml_scaler = None


def _load_ml_model(model_path: str = "color_classifier.pkl",
                   scaler_path: str = "color_scaler.pkl") -> tuple:
    """
    懒加载（Lazy Loading）机器学习模型和缩放器。

    首次调用时从磁盘加载，后续调用直接使用缓存的全局变量。

    pickle 是 Python 的序列化库，可以将 Python 对象保存到文件并在之后恢复。
    .pkl 文件是 pickle 格式的序列化文件。

    Parameters:
        model_path: 模型文件路径
        scaler_path: 缩放器文件路径

    Returns:
        (model, scaler) 元组
    """
    global _ml_model, _ml_scaler
    if _ml_model is None:
        import pickle
        with open(model_path, "rb") as f:
            _ml_model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            _ml_scaler = pickle.load(f)
    return _ml_model, _ml_scaler


def _extract_ml_features(image_bgr: np.ndarray, mask: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """
    为机器学习分类器提取特征（约 60 维）。

    特征设计理念：综合使用多种颜色空间，让分类器可以从不同角度理解颜色：

    1. HSV 直方图+统计（12+8+8 维柱状图 + 均值/标准差/中位数 ×3 通道 = 39 维）
       HSV 是最直觉的颜色空间，H 通道区分颜色，S 和 V 反映饱和度和亮度

    2. RGB 统计（5维 ×3 通道 = 15 维）
       RGB 是原始颜色空间，保留了最完整的信息

    3. Lab a,b 通道直方图（8维 ×2 通道 = 16 维）
       Lab 是感知均匀的颜色空间，a 通道代表绿-红，b 通道代表蓝-黄
       对于区分白/黄特别有用（黄色在 b 通道有较强响应）

    4. 对比度特征（2维）
       车道线与周围环境的亮度对比

    5. 形状特征（2维）
       长宽比和掩码密度

    总计约 74 维特征。

    Parameters:
        image_bgr: BGR 图像
        mask: 二值掩码
        bbox: 边界框 [x1, y1, x2, y2]

    Returns:
        特征向量 (numpy 数组)
    """
    import cv2

    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
    x2 = min(w, x2); y2 = min(h, y2)  # 确保不超出图像边界
    if x2 <= x1 or y2 <= y1:
        return np.zeros(60, dtype=np.float32)

    # 裁剪到边界框区域（减少计算量）
    crop = image_bgr[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2] if mask.shape[:2] == (h, w) else np.ones(crop.shape[:2], dtype=bool)
    if crop_mask.shape != crop.shape[:2]:
        crop_mask = cv2.resize(crop_mask.astype('uint8'), (crop.shape[1], crop.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)

    features = []

    # --- HSV 特征 ---
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask_flat = crop_mask.ravel()

    # 遍历 H(0), S(1), V(2) 三个通道
    for ch_idx, bins in [(0, 12), (1, 8), (2, 8)]:
        vals = hsv[:, :, ch_idx].ravel()[mask_flat]
        if len(vals) == 0:
            features.extend([0.0] * (bins + 3))
            continue
        # 直方图：将值域分成 bins 个区间，统计每个区间的像素比例
        # density=True 让直方图的和为 1（概率分布）
        rng = (0, 180) if ch_idx == 0 else (0, 256)
        hist, _ = np.histogram(vals, bins=bins, range=rng, density=True)
        features.extend(hist.astype(np.float32))
        # 添加统计量：均值、标准差、中位数
        features.extend([float(np.mean(vals)), float(np.std(vals)), float(np.median(vals))])

    # --- RGB 特征 ---
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    for ch_idx in range(3):
        vals = rgb[:, :, ch_idx].ravel()[mask_flat]
        if len(vals) == 0:
            features.extend([0.0] * 5)
            continue
        # 均值、标准差、10%百分位、中位数、90%百分位
        features.extend([float(np.mean(vals)), float(np.std(vals)),
                         float(np.percentile(vals, 10)), float(np.median(vals)),
                         float(np.percentile(vals, 90))])

    # --- Lab 颜色空间特征 ---
    # Lab 色彩空间比 RGB 更接近人类视觉感知
    # L 通道：亮度（与 HSV 的 V 类似）
    # a 通道：绿色到红色的渐变
    # b 通道：蓝色到黄色的渐变 ← 对黄色检测特别有用
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
    for ch_idx in [1, 2]:  # a 通道和 b 通道
        vals = lab[:, :, ch_idx].ravel()[mask_flat]
        if len(vals) == 0:
            features.extend([0.0] * 8)
            continue
        hist, _ = np.histogram(vals, bins=8, range=(0, 256), density=True)
        features.extend(hist.astype(np.float32))

    # --- 对比度特征 ---
    kernel = np.ones((10, 10), dtype=np.uint8)
    dilated = cv2.dilate(crop_mask.astype(np.uint8), kernel).astype(bool)
    surround = dilated & ~crop_mask  # 周围区域
    lane_v = hsv[:, :, 2].ravel()[mask_flat]
    surround_v = hsv[:, :, 2].ravel()[surround.ravel()] if surround.sum() > 0 else lane_v
    # 相对亮度比（车道线/周围）
    features.append(float(np.median(lane_v)) / max(float(np.median(surround_v)), 1.0))
    # 绝对亮度差（车道线 - 周围）
    features.append(float(np.mean(lane_v)) - float(np.mean(surround_v)))

    # --- 形状特征 ---
    bbox_w, bbox_h = x2 - x1, y2 - y1
    features.append(bbox_h / max(bbox_w, 1.0))  # 长宽比（竖长的更像是车道线）
    features.append(float(mask_flat.sum()) / max(bbox_w * bbox_h, 1.0))  # 掩码密度

    return np.asarray(features, dtype=np.float32)


def learned_classify_lane_color(
    image_bgr: np.ndarray,
    mask: np.ndarray | None = None,
    bbox: np.ndarray | None = None,
    *,
    model_path: str = "color_classifier.pkl",
    scaler_path: str = "color_scaler.pkl",
) -> ColorDecision:
    """
    使用训练好的机器学习模型分类车道线颜色。

    这是 HSV 方法的替代方案，理论上更准确，但需要先训练模型。

    工作流程：
    1. 检查模型文件是否存在，如果不存在则回退到 HSV
    2. 提取特征
    3. 用 StandardScaler 归一化特征（使其均值为0、方差为1）
    4. 用 Logistic Regression 预测概率
    5. 根据概率做出决策

    为什么需要 StandardScaler？
    不同特征的数值范围差异很大（比如色相在 0~180，亮度在 0~255），
    如果不归一化，数值范围大的特征会主导模型的预测，这是不对的。
    归一化确保所有特征对预测的贡献是公平的。

    Parameters:
        image_bgr: BGR 图像
        mask: 可选掩码
        bbox: 边界框 [x1, y1, x2, y2]（可选，如果没提供则从 mask 估算）
        model_path: 模型文件路径
        scaler_path: 缩放器文件路径

    Returns:
        ColorDecision 对象
    """
    if image_bgr.size == 0:
        return ColorDecision(UNKNOWN, 0.0, 0.0, 0.0, 0)

    import cv2
    from pathlib import Path

    # 如果模型文件不存在，回退到 HSV 方法
    if not Path(model_path).exists() or not Path(scaler_path).exists():
        return classify_lane_color(image_bgr, mask)

    h, w = image_bgr.shape[:2]
    mask_bool = _safe_mask(mask, (h, w))

    # 如果没有提供边界框，从掩码中估算
    if bbox is None:
        ys, xs = np.where(mask_bool)
        if len(xs) == 0:
            return ColorDecision(UNKNOWN, 0.0, 0.0, 0.0, 0)
        bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()])

    # 提取特征
    feats = _extract_ml_features(image_bgr, mask_bool, bbox)
    # 处理 NaN 和 Inf（防止数值不稳定）
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        model, scaler = _load_ml_model(model_path, scaler_path)
        # 归一化特征
        X = scaler.transform(feats.reshape(1, -1))
        # 预测概率
        proba = model.predict_proba(X)[0]
        # 根据模型 classes_ 确定白/黄对应的概率
        # 假设 classes_[0] = white (0), classes_[1] = yellow (1)
        white_prob = proba[0] if model.classes_[0] == 0 else proba[1]
        yellow_prob = proba[1] if model.classes_[1] == 1 else proba[0]

        # 决策（和 HSV 类似但使用 ML 概率）
        if yellow_prob > 0.5:
            return ColorDecision(YELLOW, yellow_prob, white_prob, yellow_prob, int(mask_bool.sum()))
        elif white_prob > 0.5:
            return ColorDecision(WHITE, white_prob, white_prob, yellow_prob, int(mask_bool.sum()))
        elif yellow_prob > white_prob:
            return ColorDecision(YELLOW, yellow_prob, white_prob, yellow_prob, int(mask_bool.sum()))
        else:
            return ColorDecision(WHITE, white_prob, white_prob, yellow_prob, int(mask_bool.sum()))

    except Exception:
        # 任何错误回退到 HSV
        return classify_lane_color(image_bgr, mask)
```

### 关键概念讲解

1. **HSV 颜色空间**：HSV（Hue, Saturation, Value）比 RGB 更适合做颜色分类，因为它把"颜色"（Hue）和"亮度"（Value）分开了。在 RGB 空间中，同样的白色物体在阴影下可能看起来偏蓝或偏灰，但在 HSV 中，只要 H 不变，颜色就不变。

2. **自适应阈值**：固定阈值最大的问题是：在不同光照条件下，同一条车道线在 HSV 空间中的位置会变化。自适应方法通过分析图像统计量（亮度中位数、动态范围等）来动态调整阈值，适应不同环境。

3. **Logistic Regression（逻辑回归）**：虽然名字叫"回归"，但实际上是分类算法。它用 Sigmoid 函数将线性模型的输出映射到 [0, 1] 区间，作为属于某个类别的概率。简单、可解释、计算快，适合二分类问题（白 vs 黄）。

4. **膨胀（Dilation）**：形态学操作，将掩码向外扩展。用 `cv2.dilate` 实现。这里用于计算"车道线周围区域"——膨胀后的掩码减去原始掩码就是周围区域。

5. **特征工程**：_extract_ml_features 提取了约 74 维特征，涵盖多个颜色空间（HSV、RGB、Lab）。不同颜色空间从不同角度描述颜色，组合使用让分类器有更丰富的信息。

---

## 5. prepare_local_dataset.py

### 功能概述

准备训练数据集：从 ZIP 文件中解压图像和标签、划分训练/验证/测试集、可选地将颜色标签（白/黄）合并为统一的车道线标签、生成 YOLO 所需的 data.yaml 配置文件，并将 Excel 中的数量统计转换为 JSON 格式。

### 完整源码与注释

```python
# ============================================================
# prepare_local_dataset.py — 数据集准备与预处理
# ============================================================
# 这是整个项目的数据准备入口。原始数据以 ZIP 包的形式提供，
# 本脚本负责解压、整理、划分数据集。
#
# 整个流程：
# 1. 自动发现或手动指定 ZIP 文件（示例数据 + 测试数据）和 Excel 统计表
# 2. 从 ZIP 中解压图像和 YOLO 标签
# 3. 按比例划分训练集和验证集
# 4. 可选：将多类标签（白/黄/路面）合并为单类车道线标签
# 5. 生成 data.yaml（YOLO 需要的配置文件）
# 6. 将 Excel 中的数量统计转换为 JSON
#
# 交叉文件依赖：
# - 调用 xlsx_counts.py 的 read_count_xlsx 和 write_count_json

from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from pathlib import Path

import yaml  # YAML 文件解析库，用于生成 data.yaml

from .xlsx_counts import read_count_xlsx, write_count_json


# 支持的图像和标签文件扩展名
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXTENSIONS = {".txt"}


# ============================================================
# 命令行参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the local example/test zip files for this project.")
    # --root: 项目根目录，默认当前目录
    parser.add_argument("--root", default=".", help="Project root containing zip/xlsx files.")
    # --example-zip: 示例数据 ZIP（含训练图像和标签）
    parser.add_argument("--example-zip", default=None, help="Training/example zip. Auto-detected when omitted.")
    # --test-zip: 测试数据 ZIP（仅含图像）
    parser.add_argument("--test-zip", default=None, help="Test zip. Auto-detected when omitted.")
    # --gt-xlsx: Excel 统计表
    parser.add_argument("--gt-xlsx", default=None, help="Count GT spreadsheet. Auto-detected when omitted.")
    # --out: 输出目录
    parser.add_argument("--out", default="datasets/local_colm")
    # --val-ratio: 验证集比例
    parser.add_argument("--val-ratio", type=float, default=0.2)
    # --seed: 随机种子，确保可重复性
    parser.add_argument("--seed", type=int, default=42)
    # --label-mode: 标签模式
    #   "lane-line": 合并白/黄为 lane_line（单类检测）
    #   "color": 保留白/黄分类（两/三类检测）
    parser.add_argument("--label-mode",
        choices=("lane-line", "color"),
        default="lane-line",
        help="lane-line merges white/yellow labels into class 0; color keeps white/yellow classes.",
    )
    # --lane-class-ids: 哪些源类别ID被视为"车道线"
    parser.add_argument("--lane-class-ids", default="0,1", help="Source class ids treated as lane lines.")
    # --overwrite: 覆盖已存在的输出目录
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


# ============================================================
# 文件查找和 ZIP 操作工具
# ============================================================

def auto_find(root: Path, kind: str) -> Path | None:
    """
    自动查找 ZIP 或 XLSX 文件。

    设计思路：用户不需要手动指定每个文件路径，脚本会自动识别。
    识别规则：
    - 示例 ZIP：文件名包含 "example"
    - 测试 ZIP：文件名包含 "test"
    - Excel 表：所有 .xlsx 文件

    Parameters:
        root: 搜索目录
        kind: 文件类型 ("example_zip" / "test_zip" / "gt_xlsx")

    Returns:
        找到的文件路径，如果没找到返回 None
    """
    if kind == "example_zip":
        candidates = [p for p in root.glob("*.zip") if "example" in p.name.lower()]
    elif kind == "test_zip":
        candidates = [p for p in root.glob("*.zip") if "test" in p.name.lower()]
    elif kind == "gt_xlsx":
        candidates = list(root.glob("*.xlsx"))
    else:
        raise ValueError(kind)
    # 按文件名排序，取第一个
    return sorted(candidates, key=lambda p: p.name)[0] if candidates else None


def zip_members(path: Path, extensions: set[str]) -> list[zipfile.ZipInfo]:
    """
    获取 ZIP 文件中所有指定扩展名的成员列表。

    Parameters:
        path: ZIP 文件路径
        extensions: 要匹配的扩展名集合

    Returns:
        排序后的 ZipInfo 对象列表
    """
    with zipfile.ZipFile(path) as zf:
        members = [
            info for info in zf.infolist()
            if not info.is_dir() and Path(info.filename).suffix.lower() in extensions
        ]
    return sorted(members, key=lambda info: info.filename)


def extract_member(zf: zipfile.ZipFile, member: zipfile.ZipInfo, dst: Path) -> None:
    """
    从 ZIP 中提取一个成员到指定路径。

    使用 shutil.copyfileobj 而不是 zf.extract() 的原因：
    - extract() 参数是目标目录，可能引发路径遍历安全漏洞
    - copyfileobj 完全控制输出路径，更安全

    Parameters:
        zf: 打开的 ZIP 文件对象
        member: 要提取的文件成员
        dst: 目标路径
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, dst.open("wb") as out:
        shutil.copyfileobj(src, out)

def copy_zip_images(
    zip_path: Path, out_dir: Path, split: str,
    image_members: list[zipfile.ZipInfo],
    *,
    clear: bool,
) -> list[str]:
    """
    从 ZIP 包中复制图像文件到指定分集目录。

    Parameters:
        zip_path: ZIP 文件路径
        out_dir: 输出根目录
        split: 分集名称（"train" 或 "val" 或 "test"）
        image_members: 要复制的文件列表
        clear: 是否清空已存在的目录

    Returns:
        复制的文件名列表
    """
    image_dir = out_dir / "images" / split
    label_dir = out_dir / "labels" / split

    # 如果需要清空且目录存在，则删除重建
    if clear and image_dir.exists():
        shutil.rmtree(image_dir)
    if clear and label_dir.exists():
        shutil.rmtree(label_dir)

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in image_members:
            filename = Path(member.filename).name  # 只取文件名，忽略目录结构
            extract_member(zf, member, image_dir / filename)
            copied.append(filename)
    return copied


def copy_zip_labels(
    zip_path: Path, out_dir: Path, split: str,
    image_filenames: list[str],
) -> int:
    """
    从 ZIP 包中复制与图像对应的标签文件。

    标签文件的位置：和图像同目录或不同目录，只要有相同的文件名主名（stem）。
    例如图像 "IMG001.jpg" 对应标签 "IMG001.txt"。

    Parameters:
        zip_path: ZIP 文件路径
        out_dir: 输出根目录
        split: 分集名称
        image_filenames: 图像文件名列表

    Returns:
        复制的标签文件数
    """
    labels = zip_members(zip_path, LABEL_EXTENSIONS)
    if not labels:
        return 0  # ZIP 中没有标签文件

    # 提取需要保留的文件名主名
    wanted_stems = {Path(name).stem for name in image_filenames}
    labels = [member for member in labels if Path(member.filename).stem in wanted_stems]

    label_dir = out_dir / "labels" / split
    label_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in labels:
            extract_member(zf, member, label_dir / Path(member.filename).name)
    return len(labels)


# ============================================================
# 标签合并工具
# ============================================================

def parse_class_ids(raw: str) -> set[int]:
    """
    解析逗号分隔的类别 ID 字符串，如 "0,1,2" → {0, 1, 2}。

    用于指定哪些原始类别应该被合并为车道线。
    """
    ids: set[int] = set()
    for item in raw.split(","):
        stripped = item.strip()
        if stripped:
            ids.add(int(stripped))
    return ids


def remap_label_text_to_lane_line(text: str, lane_class_ids: set[int]) -> tuple[str, int]:
    """
    将 YOLO 标签文本中指定类别 ID 的实例重新映射为 class_id 0（lane_line）。

    为什么需要这个功能？
    原始数据可能标注了白线（class_id=0）和黄线（class_id=1），
    但如果只想检测"车道线"而不区分颜色，就需要把两者都映射到 class_id=0。

    这个函数是 --label-mode lane-line 的核心实现。

    Parameters:
        text: YOLO 标签文本（多行）
        lane_class_ids: 要映射为车道线的类别 ID 集合

    Returns:
        (转换后的文本, 跳过的行数)
    """
    converted: list[str] = []
    skipped = 0
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        # 按第一个空格分割：前半部分是 class_id，后半部分是坐标
        parts = stripped.split(maxsplit=1)
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            skipped += 1
            continue
        if class_id not in lane_class_ids:
            skipped += 1
            continue  # 不是车道线类别，跳过
        # 将 class_id 改为 0，坐标保持不变
        rest = parts[1] if len(parts) > 1 else ""
        converted.append(f"0 {rest}".rstrip())
    return "\n".join(converted) + ("\n" if converted else ""), skipped


def convert_existing_labels(
    out_dir: Path, split: str, lane_class_ids: set[int]
) -> tuple[int, int]:
    """
    对已解压的标签文件进行类别映射转换。

    遍历指定分集的所有 .txt 标签文件，逐个进行 remap_label_text_to_lane_line 转换。

    Parameters:
        out_dir: 输出根目录
        split: 分集名称
        lane_class_ids: 要合并的类别 ID

    Returns:
        (转换的行数, 跳过的行数)
    """
    label_dir = out_dir / "labels" / split
    if not label_dir.exists():
        return 0, 0

    converted = 0
    skipped = 0
    for label_path in label_dir.rglob("*.txt"):
        text = label_path.read_text(encoding="utf-8")
        new_text, skipped_here = remap_label_text_to_lane_line(text, lane_class_ids)
        label_path.write_text(new_text, encoding="utf-8")
        converted += sum(1 for line in new_text.splitlines() if line.strip())
        skipped += skipped_here
    return converted, skipped


# ============================================================
# 生成 data.yaml
# ============================================================

def write_data_yaml(out_dir: Path, label_mode: str) -> Path:
    """
    生成 YOLO 训练所需的 data.yaml 配置文件。

    YOLO 需要数据配置来知道：
    - 数据集路径
    - 训练/验证/测试集的位置
    - 类别名称列表

    两种标签模式对应不同的类别定义：
    - lane-line: {0: "lane_line"} — 只检测车道线，不区分颜色
    - color: {0: "white_lane", 1: "yellow_lane", 2: "road_surface"} — 区分颜色

    Parameters:
        out_dir: 数据集根目录
        label_mode: "lane-line" 或 "color"

    Returns:
        data.yaml 文件的路径
    """
    if label_mode == "lane-line":
        names = {0: "lane_line"}
    else:
        names = {
            0: "white_lane",
            1: "yellow_lane",
            2: "road_surface",
        }

    data = {
        "path": str(out_dir.as_posix()),  # 数据集根目录（使用 POSIX 路径格式）
        "train": "images/train",           # 训练图像目录（相对路径）
        "val": "images/val",               # 验证图像目录
        "test": "images/test",             # 测试图像目录
        "names": names,                    # 类别名称映射
    }

    yaml_path = out_dir / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return yaml_path


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()  # resolve 将相对路径转换为绝对路径
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()

    # 自动或手动定位各个文件
    example_zip = Path(args.example_zip).resolve() if args.example_zip else auto_find(root, "example_zip")
    test_zip = Path(args.test_zip).resolve() if args.test_zip else auto_find(root, "test_zip")
    gt_xlsx = Path(args.gt_xlsx).resolve() if args.gt_xlsx else auto_find(root, "gt_xlsx")

    if example_zip is None or not example_zip.exists():
        raise FileNotFoundError("Could not find example zip. Pass --example-zip explicitly.")
    if test_zip is None or not test_zip.exists():
        raise FileNotFoundError("Could not find test zip. Pass --test-zip explicitly.")

    # 列出所有图像文件
    example_images = zip_members(example_zip, IMAGE_EXTENSIONS)
    test_images = zip_members(test_zip, IMAGE_EXTENSIONS)
    if not example_images:
        raise ValueError(f"No images found in {example_zip}")
    if not test_images:
        raise ValueError(f"No images found in {test_zip}")

    # 划分训练集和验证集
    rng = random.Random(args.seed)  # 使用指定种子的随机数生成器
    shuffled = example_images[:]
    rng.shuffle(shuffled)  # 打乱顺序
    val_count = max(1, int(round(len(shuffled) * args.val_ratio))) if len(shuffled) > 1 else 0
    val_images = sorted(shuffled[:val_count], key=lambda info: info.filename)
    train_images = sorted(shuffled[val_count:], key=lambda info: info.filename)

    # 复制图像
    train_files = copy_zip_images(example_zip, out_dir, "train", train_images, clear=args.overwrite)
    val_files = copy_zip_images(example_zip, out_dir, "val", val_images, clear=args.overwrite)
    test_files = copy_zip_images(test_zip, out_dir, "test", test_images, clear=args.overwrite)

    # 复制标签
    train_labels = copy_zip_labels(example_zip, out_dir, "train", train_files)
    val_labels = copy_zip_labels(example_zip, out_dir, "val", val_files)
    test_labels = copy_zip_labels(test_zip, out_dir, "test", test_files)

    # 如果需要合并标签（lane-line 模式）
    lane_label_stats = None
    if args.label_mode == "lane-line":
        lane_class_ids = parse_class_ids(args.lane_class_ids)
        train_converted, train_skipped = convert_existing_labels(out_dir, "train", lane_class_ids)
        val_converted, val_skipped = convert_existing_labels(out_dir, "val", lane_class_ids)
        test_converted, test_skipped = convert_existing_labels(out_dir, "test", lane_class_ids)
        lane_label_stats = {
            "train": (train_converted, train_skipped),
            "val": (val_converted, val_skipped),
            "test": (test_converted, test_skipped),
        }

    # 生成 data.yaml
    data_yaml = write_data_yaml(out_dir, args.label_mode)

    # 读取并保存 Excel 中的数量统计
    gt_json = None
    if gt_xlsx is not None and gt_xlsx.exists():
        counts = read_count_xlsx(gt_xlsx)
        gt_json = out_dir / "gt_counts.json"
        write_count_json(counts, gt_json)

    # 打印统计信息
    print(f"Prepared dataset: {out_dir}")
    print(f"Data yaml: {data_yaml}")
    print(f"Train images: {len(train_files)}")
    print(f"Val images: {len(val_files)}")
    print(f"Test images: {len(test_files)}")
    print(f"Copied labels: train={train_labels}, val={val_labels}, test={test_labels}")
    print(f"Label mode: {args.label_mode}")
    if lane_label_stats is not None:
        for split, (converted, skipped) in lane_label_stats.items():
            print(f"{split} lane_line labels: converted={converted}, skipped_non_lane={skipped}")
    if gt_json is not None:
        print(f"GT count json: {gt_json}")
    if train_labels == 0 and val_labels == 0:
        print("WARNING: example zip has images only. Add YOLO labels before supervised YOLO training.")


if __name__ == "__main__":
    main()
```

### 关键概念讲解

1. **数据集划分**：将数据分为训练集（train）、验证集（val）和测试集（test）。训练集用于训练模型，验证集用于调参，测试集用于最终评估。比例通常是 70% / 15% / 15% 或 80% / 20%。

2. **随机种子（Random Seed）**：机器学习中的"确定性随机"。设置 seed=42 可以确保每次运行都得到相同的随机划分，保证实验结果可复现。

3. **YAML 配置文件**：YOLO 使用 YAML 格式的配置文件。YAML 是一种人类可读的数据序列化格式，比 JSON 更容易阅读和编辑。

4. **shutil.rmtree**：Python 的目录删除函数，会递归删除目录下所有内容。使用时要特别小心，但结合 `--overwrite` 参数使用有助于确保输出目录是最新状态。

---

## 6. generate_pseudo_labels.py

### 功能概述

为没有标注的图像生成"伪标签"（pseudo-labels），使用计算机视觉传统方法——Canny 边缘检测 + Hough 变换检测车道线。这是弱监督学习的起点：先用传统方法生成粗略的标签，然后用这些标签训练 YOLO，再用 YOLO 的预测结果迭代优化。

### 完整源码与注释

```python
# ============================================================
# generate_pseudo_labels.py — 生成 YOLO 分割伪标签
# ============================================================
# 为什么要生成伪标签？
# 原始数据只有 Excel 中的车道线数量统计（每张图有几条线），
# 没有精确的 YOLO 格式像素级标注。为了训练 YOLO 分割模型，
# 我们需要先用传统视觉方法生成粗糙的标签。
#
# 使用的方法：
# 1. 高斯模糊降噪
# 2. Canny 边缘检测找到图像中的边缘
# 3. ROI 裁剪（只关注下半部分——道路区域）
# 4. Hough 变换检测直线段
# 5. 过滤出接近垂直的线段（车道线通常是竖长的）
# 6. 合并相邻线段
# 7. 将线段转换为 YOLO 分割格式的多边形
#
# OpenCV 传统方法 vs 深度学习：
# - 优点：不需要训练数据，即开即用
# - 缺点：精度有限，容易受噪声和光照影响
# - 用途：作为 YOLO 训练的"预标签"，后续会通过迭代提升质量
#
# 交叉文件依赖：
# - 生成的文件被 train_yolo.py 用于训练

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2          # OpenCV 计算机视觉库
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pseudo-labels for lane_line training.")
    parser.add_argument("--image-dir", default="datasets/local_colm/images/train")
    parser.add_argument("--label-dir", default="datasets/local_colm/labels/train")
    # Canny 边缘检测参数
    parser.add_argument("--canny-low", type=int, default=50)     # 低阈值
    parser.add_argument("--canny-high", type=int, default=150)   # 高阈值
    # Hough 变换参数
    parser.add_argument("--hough-thr", type=int, default=40)     # 阈值（越高越严格）
    parser.add_argument("--min-line-len", type=int, default=80)  # 最短线段长度（像素）
    parser.add_argument("--max-line-gap", type=int, default=30)  # 线段间隙允许值
    # 输出参数
    parser.add_argument("--line-width", type=int, default=12, help="Half-width for polygon around line.")
    # ROI（感兴趣区域）参数：只处理图像下半部分
    parser.add_argument("--roi-fraction", type=float, default=0.55,
                        help="Only process the bottom fraction of the image (road area).")
    return parser.parse_args()


def detect_lane_lines(image_bgr: np.ndarray, args: argparse.Namespace) -> list[np.ndarray]:
    """
    使用 Canny 边缘检测 + Hough 变换检测车道线。

    这是传统计算机视觉中最经典的"直线检测"流水线。
    虽然方法简单，但在清晰的道路图像上效果不错。

    处理流程：
    1. 灰度化：彩色→灰度（减少计算量，边缘信息保留）
    2. 高斯模糊：降噪（防止噪声被检测为边缘）
    3. Canny 边缘检测：找到图像中的边缘
    4. ROI 裁剪：只保留下半部分（道路区域）
    5. Hough 变换：从边缘图中检测直线
    6. 角度过滤：保留接近垂直的线段

    Parameters:
        image_bgr: BGR 图像
        args: 命令行参数

    Returns:
        线段列表，每个为 [x1, y1, x2, y2] 数组
    """
    # 1. 灰度化
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 2. 高斯模糊，核大小 5x5
    # 高斯模糊是一种平滑滤波器，每个像素的值被周围像素的加权平均替代，
    # 权重根据高斯分布计算。sigma=0 让 OpenCV 自动计算。
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Canny 边缘检测
    # Canny 算法的步骤：
    #   a. 高斯滤波降噪
    #   b. 计算梯度幅值和方向（用 Sobel 算子）
    #   c. 非极大值抑制（NMS）：细化边缘到单像素宽
    #   d. 双阈值检测：高于高阈值的是强边缘，低于低阈值的是非边缘，
    #      介于两者之间的是弱边缘（如果与强边缘连接则保留）
    # 参数 low=50, high=150 的意思是：
    #   - 梯度幅值 > 150: 确定为边缘
    #   - 梯度幅值 < 50: 确定为非边缘
    #   - 50~150: 如果与强边缘相连则为边缘
    edges = cv2.Canny(blurred, args.canny_low, args.canny_high)

    # 4. ROI（感兴趣区域）：只保留下半部分
    # 车道线通常出现在图像的下半部分（道路区域）
    h, w = edges.shape
    roi_top = int(h * (1 - args.roi_fraction))  # 上半部分的截止线
    edges[:roi_top, :] = 0  # 将上半部分清零

    # 5. Hough 变换检测线段
    # HoughLinesP 是"概率霍夫变换"，检测线段（有端点）。
    # 相比标准霍夫变换（检测无限长的直线），概率版本更快且给出线段。
    #
    # 参数说明：
    #   rho=1: 距离分辨率（1像素）
    #   theta=np.pi/180: 角度分辨率（1度）
    #   threshold=40: 至少需要40个边缘点才能形成一条直线
    #   minLineLength=80: 最短线段长度（像素）
    #   maxLineGap=30: 线段上允许的最大间隙
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=args.hough_thr,
        minLineLength=args.min_line_len,
        maxLineGap=args.max_line_gap,
    )

    if lines is None:
        return []  # 没有检测到线段

    # 提取线段坐标
    segments = [line[0].astype(np.float32) for line in lines]

    # 6. 角度过滤：车道线通常是接近垂直的
    filtered = []
    for seg in segments:
        x1, y1, x2, y2 = seg
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        # 计算线段与垂直方向的夹角
        angle = math.degrees(math.atan2(dx, dy)) if dy > 1 else 90
        if angle < 45:  # 与垂直方向夹角小于45度 = 更垂直而不是水平
            filtered.append(seg)

    return filtered


def line_to_polygon(
    x1: float, y1: float, x2: float, y2: float,
    width: float, h: int, w: int
) -> list[tuple[float, float]]:
    """
    将一条线段转换为一个薄四边形（polygon）。

    YOLO 分割格式需要多边形而不是直线。所以我们将线段"扩展"成一个细长的四边形。

    算法：
    1. 计算线段的方向向量 (dx, dy)
    2. 计算法线向量（垂直方向）(-dy, dx) 归一化后乘以宽度
    3. 在线段两侧各扩展 width 像素，形成四边形
    4. 沿线段方向再扩展 0.5*width，使覆盖更好

    Parameters:
        x1, y1, x2, y2: 线段端点
        width: 扩展半宽度（最终四边形宽度 = 2 * width）
        h, w: 图像尺寸（用于边界检查）

    Returns:
        多边形的顶点列表
    """
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)  # 线段长度（欧氏距离）
    if length < 1:
        return []  # 线段太短，无法生成有效多边形

    # 计算法线方向（与线段垂直的单位向量）
    nx = -dy / length * width
    ny = dx / length * width

    # 基本四边形的四个顶点
    pts = [
        (x1 + nx, y1 + ny),   # 端点1的上方
        (x1 - nx, y1 - ny),   # 端点1的下方
        (x2 - nx, y2 - ny),   # 端点2的下方
        (x2 + nx, y2 + ny),   # 端点2的上方
    ]

    # 沿线段方向再延伸 0.5 * width，使端部覆盖更好
    ex = dx / length * width * 0.5
    ey = dy / length * width * 0.5
    pts[0] = (pts[0][0] - ex, pts[0][1] - ey)
    pts[1] = (pts[1][0] - ex, pts[1][1] - ey)
    pts[2] = (pts[2][0] + ex, pts[2][1] + ey)
    pts[3] = (pts[3][0] + ex, pts[3][1] + ey)

    return pts


def polygon_to_yolo_line(polygon: list[tuple[float, float]], w: int, h: int) -> str:
    """
    将多边形像素坐标转换为 YOLO 分割格式的文本行。

    YOLO 分割格式：
    <class_id> <x1_norm> <y1_norm> <x2_norm> <y2_norm> ...

    坐标归一化：所有坐标值除以图像宽/高，范围 [0, 1]。
    这样做的好处是：无论原始图像尺寸多大，模型看到的坐标都在 0~1 范围内。
    """
    norm = []
    for px, py in polygon:
        norm.append(f"{float(px) / w:.6f}")   # x 坐标归一化，保留 6 位小数
        norm.append(f"{float(py) / h:.6f}")   # y 坐标归一化
    return "0 " + " ".join(norm)  # class_id=0（lane_line）


def merge_nearby_lines(
    segments: list[np.ndarray],
    image_shape: tuple[int, int],
    angle_thr: float = 10.0,
    dist_thr: float = 80.0,
) -> list[np.ndarray]:
    """
    合并角度相近且位置接近的线段。

    为什么要合并？因为 Hough 变换可能将一条完整的车道线
    检测为多条断裂的短线段。合并后能获得更完整的车道线。

    合并算法：
    1. 遍历所有线段，找到一个种子线段
    2. 找到所有与种子线段角度差 < angle_thr
       且中心点距离 < dist_thr 的线段
    3. 将这些线段的所有端点合并，用 cv2.fitLine 拟合一条直线
    4. 标记已合并的线段，避免重复处理

    Parameters:
        segments: 线段列表
        image_shape: 图像形状 (h, w)
        angle_thr: 角度差阈值（度）
        dist_thr: 中心点距离阈值（像素）

    Returns:
        合并后的线段列表
    """
    if len(segments) < 2:
        return segments

    h, w = image_shape

    def line_angle(seg):
        """计算线段的角度（相对于垂直方向）"""
        dx = seg[2] - seg[0]
        dy = seg[3] - seg[1]
        return math.degrees(math.atan2(dx, dy))

    def line_center(seg):
        """计算线段的中点"""
        return np.array([(seg[0] + seg[2]) / 2, (seg[1] + seg[3]) / 2])

    merged = []
    used = [False] * len(segments)  # 跟踪哪些线段已被合并

    for i, seg_i in enumerate(segments):
        if used[i]:
            continue  # 已被合并，跳过

        angle_i = line_angle(seg_i)
        center_i = line_center(seg_i)
        group = [seg_i]
        used[i] = True

        # 找所有与 seg_i 相似的线段
        for j, seg_j in enumerate(segments):
            if used[j]:
                continue
            angle_j = line_angle(seg_j)
            center_j = line_center(seg_j)
            angle_diff = abs(angle_i - angle_j)
            dist = np.linalg.norm(center_i - center_j)  # 欧氏距离

            if angle_diff < angle_thr and dist < dist_thr:
                group.append(seg_j)
                used[j] = True

        if len(group) == 1:
            # 没有可合并的线段
            merged.append(group[0])
        else:
            # 合并：将所有端点收集起来，拟合一条直线
            all_pts = np.vstack([g.reshape(2, 2) for g in group])
            # cv2.fitLine 通过最小二乘法（L2距离）拟合直线
            # 返回 (vx, vy, cx, cy) — 方向向量和线上一点
            vx, vy, cx, cy = cv2.fitLine(all_pts, cv2.DIST_L2, 0, 0.01, 0.01)
            # 将所有点投影到直线上，找到最远的两个点作为端点
            proj = []
            for pt in all_pts:
                t = (pt[0] - cx) * vx + (pt[1] - cy) * vy
                proj.append(t)
            t_min, t_max = min(proj), max(proj)
            p1 = np.array([cx + t_min * vx[0], cy + t_min * vy[0]])
            p2 = np.array([cx + t_max * vx[0], cy + t_max * vy[0]])
            merged.append(np.array([p1[0], p1[1], p2[0], p2[1]], dtype=np.float32))

    return merged


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    # 找到所有图像文件
    image_files = sorted(p for p in image_dir.rglob("*")
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})

    total_labels = 0
    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  SKIP cannot read: {img_path.name}")
            continue
        h, w = image.shape[:2]

        # 检测车道线
        segments = detect_lane_lines(image, args)
        if not segments:
            # 没有检测到车道线，创建空标签文件
            label_path = label_dir / f"{img_path.stem}.txt"
            label_path.write_text("", encoding="utf-8")
            continue

        # 合并相邻线段
        merged = merge_nearby_lines(segments, (h, w))

        # 转换为 YOLO 分割格式
        yolo_lines = []
        for seg in merged:
            x1, y1, x2, y2 = seg
            poly = line_to_polygon(x1, y1, x2, y2, args.line_width, h, w)
            if poly:
                yolo_lines.append(polygon_to_yolo_line(poly, w, h))

        # 写入标签文件
        label_path = label_dir / f"{img_path.stem}.txt"
        label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        total_labels += len(yolo_lines)

        if len(image_files) <= 10 or len(yolo_lines) > 0:
            print(f"  {img_path.name}: {len(merged)} segments -> {len(yolo_lines)} labels")

    print(f"\nTotal: {total_labels} labels across {len(image_files)} images in {label_dir}")


if __name__ == "__main__":
    main()
```

### 关键概念讲解

1. **Canny 边缘检测**：1986 年提出的经典边缘检测算法。它找到图像中亮度变化剧烈的像素点（即"边缘"）。算法包含多步：
   - 高斯滤波降噪
   - Sobel 算子计算梯度（亮度变化的方向和强度）
   - 非极大值抑制：将宽边缘细化到单像素
   - 双阈值检测：区分强边缘、弱边缘和噪声

2. **Hough 变换**：检测直线的经典算法。核心思想是：图像空间中的一条直线对应参数空间中的一个点。通过在参数空间中"投票"，找到票数最高的参数组合就是检测到的直线。概率霍夫变换（HoughLinesP）是优化版本，直接检测线段而不是无限长直线。

3. **ROI（Region of Interest）**：感兴趣区域。车道线通常位于图像下半部分（道路区域），通过只处理下半部分可以减少计算量并避免误检（如天空中的电线）。

4. **伪标签（Pseudo-Label）**：半监督学习中的概念。先用传统方法或已训练的模型为无标签数据生成标签，然后用这些"伪标签"训练模型。在项目迭代中，伪标签的质量会随着模型能力提升而提升。

---

## 7. refine_labels_from_gt.py

### 功能概述

利用 Excel 中的车道线数量标注（GT counts）来筛选 YOLO 模型在测试集上的预测结果，生成高质量的 YOLO 分割训练标签。核心思路：虽然我们不知道每条线的精确位置，但我们知道有多少条线（GT count），利用这个"弱监督"信息可以从模型的多个候选预测中挑选出最可靠的。

### 完整源码与注释

```python
# ============================================================
# refine_labels_from_gt.py — 用 GT 数量筛选预测生成高质量标签
# ============================================================
# 场景：测试集图片在 Excel 中有"这条图有几条车道线"的标注，
# 但没有 YOLO 格式的精确位置标注。
#
# 我们的方案：
# 1. 用已经训练好的 YOLO 模型预测测试集图片
# 2. 对每张图片，YOLO 会输出多个候选检测（可能多于实际数量）
# 3. 用 GT count（正确答案中的数量）来"掐头去尾"：
#    只保留最可信的 top-K 个检测（K = GT count）
# 4. 将这些检测转换为 YOLO 分割标签
#
# 这就是"弱监督学习"的一种形式——用弱标注（数量）来生成强标注（位置和形状）。
#
# 交叉文件依赖：
# - 调用 xlsx_counts.py 的 read_count_xlsx
# - 生成的文件被 train_yolo.py 用于训练（迭代训练）

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .xlsx_counts import read_count_xlsx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert filtered YOLO predictions to training labels.")
    parser.add_argument("--weights", required=True, help="Path to trained YOLO best.pt.")
    parser.add_argument("--image-dir", default="datasets/local_colm/images/test")
    parser.add_argument("--label-dir", default="datasets/local_colm/labels/test")
    parser.add_argument("--gt-xlsx", default="结果统计.xlsx")
    parser.add_argument("--conf", type=float, default=0.1, help="Low conf threshold to include more candidates.")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    return parser.parse_args()

def mask_to_polygon(mask: np.ndarray, simplify_eps: float = 2.0) -> list[tuple[float, float]] | None:
    """
    将分割掩码（mask）转换为简化的多边形。

    分割模型的输出是一个逐像素的掩码，我们需要将其转换为 YOLO 格式的多边形。
    转换步骤：
    1. 用 cv2.findContours 找到掩码的轮廓
    2. 选取面积最大的轮廓（排除小噪声）
    3. 用 cv2.approxPolyDP 简化轮廓（减少顶点数量）
    4. 返回多边形顶点列表

    cv2.approxPolyDP 使用 Douglas-Peucker 算法简化多边形：
    - 找到轮廓上距离最远的两个点
    - 递归地添加距离最大的点，直到最大距离 < epsilon
    - 这样可以保留轮廓的大致形状，同时大幅减少顶点数

    Parameters:
        mask: 二值掩码 (H x W)
        simplify_eps: 简化精度（值越小，多边形越精细）

    Returns:
        多边形顶点列表，如果失败返回 None
    """
    import cv2

    # 查找轮廓
    # RETR_EXTERNAL: 只提取最外层的轮廓（忽略内孔）
    # CHAIN_APPROX_SIMPLE: 只保存端点（压缩轮廓）
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    # 选取面积最大的轮廓
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 3:
        return None  # 至少需要3个点才能形成多边形

    # Douglas-Peucker 简化
    epsilon = simplify_eps / 1000.0 * cv2.arcLength(largest, True)  # 基于周长的比例
    approx = cv2.approxPolyDP(largest, epsilon, True)

    # 提取坐标
    pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
    return pts if len(pts) >= 3 else None


def bbox_to_polygon(xyxy: np.ndarray) -> list[tuple[float, float]]:
    """
    当掩码不可用时，将边界框转换为多边形（降级方案）。

    边界框只有 4 个角，所以多边形就是矩形本身。
    虽然不如掩码精确，但总比没有好。
    """
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def line_score(xyxy: np.ndarray, conf: float, mask: np.ndarray | None = None) -> float:
    """
    对检测结果进行评分，用于选择"最可能"的检测。

    评分标准：
    1. conf（置信度）：模型自己的置信度
    2. log1p(length)：线越长越好（log1p 防止极端大值主导）
    3. min(aspect/15, 1)：长宽比越大越好（车道线应该是细长的）
    4. density：掩码密度（填充比例），越大越好

    总分 = conf * log1p(length) * min(aspect/15, 1) * density

    设计思路：一条"好的"车道线检测应该是：
    - 模型有信心的（高 conf）
    - 足够长的（高 length）
    - 细长的（高 aspect ratio）
    - 掩码紧密的（高 density，不是松散的点）

    Parameters:
        xyxy: 边界框 [x1, y1, x2, y2]
        conf: 检测置信度
        mask: 分割掩码

    Returns:
        综合评分
    """
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    length = max(w, h)          # 线的"长度"取宽高中的较大值
    width = min(w, h)           # 线的"宽度"取宽高中的较小值
    aspect = length / max(width, 1.0)  # 长宽比（越大约好）

    # 掩码面积或边界框面积
    mask_area = float(mask.sum()) if mask is not None else (w * h)
    density = mask_area / max(w * h, 1.0)  # 掩码密度（面积填充比例）

    # log1p(length) = log(1 + length)，对数转换使长度的影响是亚线性的
    return float(conf) * math.log1p(length) * min(aspect / 15.0, 1.0) * density


def polygon_to_yolo_line(polygon: list[tuple[float, float]], w: int, h: int) -> str:
    """
    将多边形转换为 YOLO 分割格式文本行。

    格式：<class_id> <x1_norm> <y1_norm> <x2_norm> <y2_norm> ...
    """
    norm = []
    for px, py in polygon:
        norm.append(f"{float(px) / w:.6f}")
        norm.append(f"{float(py) / h:.6f}")
    return "0 " + " ".join(norm)


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO  # Ultralytics YOLO 库

    # 读取 Excel 中的 GT 数量标注
    gt_counts = read_count_xlsx(Path(args.gt_xlsx))
    print(f"Loaded GT counts for {len(gt_counts)} images")

    # 加载 YOLO 模型
    model = YOLO(args.weights)
    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有测试图像
    image_files = sorted(p for p in image_dir.rglob("*")
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})

    import cv2

    total_labels = 0
    matched = 0

    for img_path in image_files:
        stem = img_path.stem  # 文件名（不含扩展名）

        # 查找 GT 数量：先用完整文件名查，再用文件名主名查
        gt = gt_counts.get(stem) or gt_counts.get(img_path.name)
        target_count = gt.get("lane_line", 0) if gt else None

        # YOLO 预测
        results = model.predict(
            str(img_path), imgsz=args.imgsz, conf=args.conf,
            device=args.device, verbose=False, stream=True,
        )
        result = next(results)  # 取第一个（也是唯一一个）结果

        # 读取原始图像
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            label_dir.joinpath(f"{stem}.txt").write_text("", encoding="utf-8")
            continue

        xyxy = boxes.xyxy.cpu().numpy()  # 边界框坐标
        confs = boxes.conf.cpu().numpy()  # 置信度

        # 处理分割掩码
        masks = None
        if result.masks is not None and result.masks.data is not None:
            masks = result.masks.data.cpu().numpy()
            # 如果掩码尺寸与图像不匹配，进行缩放
            if masks.shape[1:3] != (h, w):
                resized = []
                for m in masks:
                    resized.append(cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST))
                masks = np.asarray(resized)

        # 评分并排序
        scored = []
        for idx in range(len(xyxy)):
            mask = masks[idx] > 0.5 if masks is not None else None
            score = line_score(xyxy[idx], confs[idx], mask)
            scored.append((score, idx))
        scored.sort(key=lambda x: x[0], reverse=True)  # 按评分从高到低排序

        # 用 GT count 截取前 K 个
        if target_count is not None and target_count > 0:
            keep_indices = [idx for _, idx in scored[:target_count]]
            matched += 1
        else:
            # 没有 GT count：保留评分 > 0.1 的检测
            keep_indices = [idx for score, idx in scored if score > 0.1]

        # 转换为 YOLO 标签
        yolo_lines = []
        for idx in keep_indices:
            if masks is not None:
                mask = masks[idx] > 0.5
                poly = mask_to_polygon(mask)
            else:
                poly = None

            if poly is None:
                poly = bbox_to_polygon(xyxy[idx])  # 降级：使用边界框

            yolo_lines.append(polygon_to_yolo_line(poly, w, h))

        # 写入文件
        label_path = label_dir / f"{stem}.txt"
        label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        total_labels += len(yolo_lines)

        status = f"GT={target_count}" if target_count is not None else "no_GT"
        print(f"  {img_path.name}: {len(yolo_lines)} labels ({status})")

    print(f"\nTotal: {total_labels} labels across {len(image_files)} images")
    print(f"GT-matched images: {matched}/{len(image_files)}")


if __name__ == "__main__":
    main()
```

### 关键概念讲解

1. **弱监督学习（Weakly Supervised Learning）**：在监督学习中，标签的"强度"可以是不同级别的：
   - 强监督：每个实例的精确标注（如像素级分割）
   - 弱监督：每个实例的粗略标注（如边界框、数量）
   
   本项目用"车道线数量"（弱监督）来筛选 YOLO 的预测结果（强监督），是一种典型的弱监督到强监督的转换。

2. **Douglas-Peucker 简化算法**：一种曲线简化算法。给定一条曲线，算法找到曲线上距离最远的两个点，然后递归地添加距离最大的中间点，直到所有点到简化曲线的距离小于 epsilon。在 OpenCV 中通过 `cv2.approxPolyDP` 实现。

3. **轮廓查找（cv2.findContours）**：从二值图像中提取轮廓。`RETR_EXTERNAL` 只获取最外层轮廓（排除内部空洞），`CHAIN_APPROX_SIMPLE` 对轮廓进行压缩（只保存转折点）。

4. **评分函数设计**：`line_score` 综合考虑了置信度、长度、长宽比和掩码密度。多因素评分比单一的置信度排序更可靠，因为高置信度的检测可能是小片段的误检，而长且细的检测更有可能是完整的车道线。

---

## 8. train_color_classifier.py

### 功能概述

训练一个逻辑回归分类器来替代 HSV 规则进行颜色判断（白线 vs 黄线）。流程是：从约束后的预测中提取特征和标签 → 用交叉验证评估 → 训练最终模型并保存。

### 完整源码与注释

```python
# ============================================================
# train_color_classifier.py — 训练机器学习颜色分类器
# ============================================================
# 为什么需要机器学习分类器？
# HSV 固定阈值方法虽然简单，但在复杂光照条件下（阴影、阳光直射、黄昏等）
# 容易出错。机器学习可以学习更复杂的颜色判断规则。
#
# 训练数据的构建：
# 1. 用 YOLO 检测车道线（用低置信度阈值获取大量候选）
# 2. 用量级约束（apply_count_constraints.py）得到"最可靠"的预测
# 3. 约束后的预测作为训练标签（白线/黄线）
# 4. 原始的候选检测作为特征提取对象
# 5. 通过 IoU 匹配将候选检测映射到约束结果
#
# 这个过程叫做"自训练（self-training）"或"弱监督训练"。
#
# 交叉文件依赖：
# - 生成 color_classifier.pkl 和 color_scaler.pkl，被 color_classifier.py 的
#   learned_classify_lane_color() 调用

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # 交叉验证
from sklearn.preprocessing import StandardScaler  # 特征标准化
from sklearn.metrics import classification_report  # 分类报告


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a color classifier for lane white/yellow.")
    parser.add_argument("--raw-pred", default="predictions_v3_raw.json",
                        help="Raw predictions with low conf, many candidates.")
    parser.add_argument("--constrained-pred", default="predictions_v3_constrained.json",
                        help="Count-constrained predictions with perfect color labels.")
    parser.add_argument("--image-dir", default="datasets/local_colm/images/test")
    parser.add_argument("--out-model", default="color_classifier.pkl")
    parser.add_argument("--out-scaler", default="color_scaler.pkl")
    return parser.parse_args()


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算两个边界框的 IoU。

    和 geometry.py 中的 bbox_iou_xyxy 功能相同，
    但这里是 NumPy 数组版本的简化实现。
    """
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def extract_features(image_bgr: np.ndarray, mask: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """
    为机器学习分类器提取颜色和形状特征。

    这个函数与 color_classifier.py 中的 _extract_ml_features 功能完全相同，
    二者是"训练时"和"推理时"的对应关系。

    特征向量包含（约 74 维）：
    1. HSV 三通道的直方图 + 均值/标准差/中位数（39维）
    2. RGB 三通道的均值/标准差/10%/50%/90% 百分位（15维）
    3. Lab a/b 通道直方图（16维）
    4. 对比度特征（2维）
    5. 形状特征（2维）

    详细注释请参考 color_classifier.py 中的 _extract_ml_features。
    """
    import cv2

    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
    x2 = min(w, x2); y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return np.zeros(60, dtype=np.float32)

    crop = image_bgr[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2] if mask.shape[:2] == (h, w) else np.ones(crop.shape[:2], dtype=bool)
    if crop_mask.shape != crop.shape[:2]:
        crop_mask = cv2.resize(crop_mask.astype('uint8'), (crop.shape[1], crop.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)

    features = []

    # HSV features
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    mask_flat = crop_mask.ravel()

    for ch, bins, name in [(h_ch, 12, 'hue'), (s_ch, 8, 'sat'), (v_ch, 8, 'val')]:
        vals = ch.ravel()[mask_flat]
        if len(vals) == 0:
            features.extend([0.0] * (bins + 3))
            continue
        hist, _ = np.histogram(vals, bins=bins, range=(0, 256 if name != 'hue' else 180), density=True)
        features.extend(hist.astype(np.float32))
        features.extend([float(np.mean(vals)), float(np.std(vals)), float(np.median(vals))])

    # RGB features
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    for ch_idx in range(3):
        vals = rgb[:, :, ch_idx].ravel()[mask_flat]
        if len(vals) == 0:
            features.extend([0.0] * 5)
            continue
        features.extend([float(np.mean(vals)), float(np.std(vals)),
                         float(np.percentile(vals, 10)), float(np.median(vals)),
                         float(np.percentile(vals, 90))])

    # Lab features
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
    a_ch, b_ch = lab[:, :, 1], lab[:, :, 2]
    for ch, bins in [(a_ch, 8), (b_ch, 8)]:
        vals = ch.ravel()[mask_flat]
        if len(vals) == 0:
            features.extend([0.0] * bins)
            continue
        hist, _ = np.histogram(vals, bins=bins, range=(0, 256), density=True)
        features.extend(hist.astype(np.float32))

    # Contrast
    kernel = np.ones((10, 10), dtype=np.uint8)
    dilated = cv2.dilate(crop_mask.astype(np.uint8), kernel).astype(bool)
    surround = dilated & ~crop_mask
    lane_v = v_ch.ravel()[mask_flat]
    surround_v = v_ch.ravel()[surround.ravel()] if surround.sum() > 0 else lane_v
    features.append(float(np.median(lane_v)) / max(float(np.median(surround_v)), 1.0))
    features.append(float(np.mean(lane_v)) - float(np.mean(surround_v)))

    # Shape
    bbox_w, bbox_h = x2 - x1, y2 - y1
    aspect = bbox_h / max(bbox_w, 1.0)
    density = float(mask_flat.sum()) / max(bbox_w * bbox_h, 1.0)
    features.extend([aspect, density])

    return np.asarray(features, dtype=np.float32)


def main() -> None:
    args = parse_args()
    import cv2

    # 加载两种预测结果：
    # raw: 低置信度阈值，包含大量候选检测
    # constrained: 经过 count 约束的"最佳"检测
    raw = json.loads(Path(args.raw_pred).read_text(encoding="utf-8"))
    constrained = json.loads(Path(args.constrained_pred).read_text(encoding="utf-8"))

    # 构建文件名主名 → 数据的快速查找表
    raw_lookup = {Path(k).stem: v for k, v in raw["images"].items()}
    constrained_lookup = {Path(k).stem: v for k, v in constrained["images"].items()}

    image_dir = Path(args.image_dir)

    X_list, y_list = [], []  # 特征矩阵和标签向量
    matched_count = 0
    total_raw = 0

    for stem in sorted(raw_lookup):
        raw_payload = raw_lookup[stem]
        const_payload = constrained_lookup.get(stem)
        if const_payload is None:
            continue  # 没有约束结果，跳过（通常不会发生）

        raw_insts = raw_payload.get("instances", [])
        const_insts = const_payload.get("instances", [])

        # 构建约束实例的边界框列表
        const_bboxes = []
        for inst in const_insts:
            bbox = inst.get("bbox")
            if bbox and len(bbox) == 4:
                const_bboxes.append((np.array(bbox), inst.get("class", "")))

        # 查找对应的图像文件
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            p = image_dir / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]

        # 将原始候选检测与约束结果进行匹配
        for raw_inst in raw_insts:
            total_raw += 1
            raw_bbox = raw_inst.get("bbox")
            if not raw_bbox or len(raw_bbox) != 4:
                continue

            raw_bbox_np = np.array(raw_bbox)

            # 找到 IoU 最高的约束实例
            best_iou, best_label = 0.0, None
            for const_bbox_np, const_label in const_bboxes:
                iou = bbox_iou(raw_bbox_np, const_bbox_np)
                if iou > best_iou:
                    best_iou = iou
                    best_label = const_label

            # 只有 IoU 足够高且颜色明确时才作为训练样本
            if best_iou < 0.3 or best_label not in ("white_lane", "yellow_lane"):
                continue

            matched_count += 1

            # 构建掩码（用边界框作为简化掩码）
            x1, y1, x2, y2 = [max(0, int(v)) for v in raw_bbox_np]
            x2, y2 = min(w, x2), min(h, y2)
            mask = np.zeros((h, w), dtype=bool)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = True

            # 提取特征
            feats = extract_features(image, mask, raw_bbox_np)
            if np.isnan(feats).any():
                continue  # 跳过包含 NaN 的特征

            X_list.append(feats)
            y_list.append(0 if best_label == "white_lane" else 1)  # 0=白, 1=黄

    # 转换为 NumPy 数组
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f"Total raw detections: {total_raw}")
    print(f"Matched to constrained labels: {matched_count}")
    print(f"White samples: {(y == 0).sum()}, Yellow samples: {(y == 1).sum()}")

    if len(np.unique(y)) < 2:
        print("ERROR: Only one class in training data, cannot train classifier.")
        return

    # 处理数值异常
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # --- 交叉验证评估 ---
    scaler = StandardScaler()  # 标准化：减去均值除以标准差
    X_scaled = scaler.fit_transform(X)

    # StratifiedKFold：分层 K 折交叉验证
    # 保持每折中白/黄样本比例与原始数据集相同
    n_folds = min(5, int((y == 1).sum()))  # 避免折数超过少数类样本数
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # 逻辑回归参数：
    # max_iter=1000: 最大迭代次数（默认100不够，有时不收敛）
    # class_weight="balanced": 自动平衡类别权重（处理样本不均衡）
    # C=1.0: 正则化强度的倒数（越小正则化越强）
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42)
    y_pred = cross_val_predict(clf, X_scaled, y, cv=skf, method="predict")

    print("\n=== Cross-Validation Results ===")
    print(classification_report(y, y_pred, target_names=["white_lane", "yellow_lane"], digits=4))

    # --- 用全部数据训练最终模型 ---
    clf_final = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42)
    clf_final.fit(X_scaled, y)

    # 保存模型和缩放器
    with open(args.out_model, "wb") as f:
        pickle.dump(clf_final, f)
    with open(args.out_scaler, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Model saved to {args.out_model}")
    print(f"Scaler saved to {args.out_scaler}")

    # 查看特征重要性
    if hasattr(clf_final, "coef_"):
        top_idx = np.argsort(np.abs(clf_final.coef_[0]))[::-1][:15]
        print("\nTop 15 feature indices by importance:")
        for i in top_idx:
            print(f"  feat[{i}]: {clf_final.coef_[0][i]:+.4f}")


if __name__ == "__main__":
    main()
```

### 关键概念讲解

1. **逻辑回归（Logistic Regression）**：虽然是二分类算法，但它输出的是概率值（通过 Sigmoid 函数）。在本项目中，逻辑回归的输入是 ~74 维特征向量，输出是"是白色"和"是黄色"的概率。

2. **交叉验证（Cross-Validation）**：将数据分成 K 份（这里 K=5），每次用 K-1 份训练、1 份验证，重复 K 次。这样可以更可靠地评估模型性能，避免"过拟合评估"（模型恰好对测试集表现好）。`StratifiedKFold` 确保每份中白/黄比例与总体一致。

3. **StandardScaler（标准化）**：将每个特征减去均值、除以标准差，使所有特征的均值为 0、方差为 1。逻辑回归对特征尺度敏感，标准化是必要的预处理步骤。

4. **类别权重平衡（class_weight="balanced"）**：当白线和黄线的样本数量相差很大时，模型会偏向多数类。balanced 选项自动给少数类分配更大的权重，使模型更关注少数类。

5. **自训练（Self-Training）**：先用 YOLO 生成候选检测，再用 count 约束生成"可靠"标签，然后用这些标签训练颜色分类器。这种"模型生成自己的训练数据"的方法就是自训练。

---

## 9. train_yolo.py

### 功能概述

使用 Ultralytics YOLO 库训练车道线分割模型。支持各种 YOLO 参数配置、数据增强设置和训练策略调整。

### 完整源码与注释

```python
# ============================================================
# train_yolo.py — 训练 YOLO 车道线检测/分割模型
# ============================================================
# 这个文件是整个项目的模型训练入口。
# 它是对 Ultralytics YOLO API 的一层封装，简化了命令行调用。
#
# 支持的 YOLO 模型：
# - yolov8n-seg.pt:   YOLOv8 Nano 分割版（最小、最快）
# - yolov8s-seg.pt:   YOLOv8 Small 分割版
# - yolov8m-seg.pt:   YOLOv8 Medium 分割版（推荐）
# - yolov8l-seg.pt:   YOLOv8 Large 分割版
# - yolov8x-seg.pt:   YOLOv8 X-Large 分割版（最大、最准）
#
# 为什么使用分割模型（-seg）而不是检测模型？
# 车道线需要精确的形状信息，不仅仅是边界框。
# 分割模型能输出像素级掩码，对车道线形状有更好的描述。
#
# 关键训练策略：
# - close_mosaic=15: 最后 15 个 epoch 关闭马赛克增强（让模型收敛更好）
# - cos_lr: 使用余弦退火学习率调度
# - 数据增强针对车道线优化（小角度旋转、平移、缩放适中）

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO lane-line detector/segmenter.")
    parser.add_argument("--data", default="datasets/local_colm/data.yaml", help="YOLO data yaml.")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="Pretrained YOLO model.")
    parser.add_argument("--epochs", type=int, default=120)     # 训练轮数
    parser.add_argument("--imgsz", type=int, default=960)      # 输入图像大小（车道线需要较高分辨率）
    parser.add_argument("--batch", type=int, default=8)        # 批次大小（受 GPU 显存限制）
    parser.add_argument("--device", default=None, help="CUDA device such as 0, 0,1 or cpu.")
    parser.add_argument("--workers", type=int, default=8)      # 数据加载进程数
    parser.add_argument("--project", default="runs/segment")   # 输出目录
    parser.add_argument("--name", default="colm_lane")         # 实验名称
    parser.add_argument("--patience", type=int, default=40)    # 早停耐心值（40轮无改善则停止）
    parser.add_argument("--seed", type=int, default=42)        # 随机种子
    parser.add_argument("--resume", action="store_true")       # 从断点恢复训练
    parser.add_argument("--cache", action="store_true")        # 缓存图像到内存（加速训练）
    parser.add_argument("--close-mosaic", type=int, default=15) # 最后N轮关闭马赛克增强
    parser.add_argument("--cos-lr", action="store_true")       # 使用余弦学习率
    parser.add_argument("--optimizer", default="auto")         # 优化器（auto=自动选择）
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Ultralytics train args, e.g. hsv_v=0.6 degrees=2.0",
    )
    return parser.parse_args()


def parse_extra(extra: list[str]) -> dict:
    """
    解析额外的训练参数。

    支持以下类型自动转换：
    - "true"/"false" → bool
    - 整数 → int
    - 浮点数 → float
    - 其他 → str（原样保留）

    Parameters:
        extra: ["hsv_v=0.6", "degrees=2.0"] 格式的参数列表

    Returns:
        参数字典 {"hsv_v": 0.6, "degrees": 2.0}
    """
    parsed: dict[str, object] = {}
    for item in extra:
        if "=" not in item:
            raise ValueError(f"Invalid --extra item {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        lowered = value.lower()
        if lowered in {"true", "false"}:
            parsed[key] = lowered == "true"
            continue
        try:
            parsed[key] = int(value)
            continue
        except ValueError:
            pass
        try:
            parsed[key] = float(value)
            continue
        except ValueError:
            pass
        parsed[key] = value
    return parsed


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data yaml not found: {data_path}")

    from ultralytics import YOLO

    # 加载预训练模型
    model = YOLO(args.model)

    # 训练参数配置
    train_args = {
        "data": str(data_path),     # 数据集配置
        "epochs": args.epochs,       # 训练轮数
        "imgsz": args.imgsz,         # 输入图像大小
        "batch": args.batch,         # 批次大小
        "workers": args.workers,     # 数据加载线程数
        "project": args.project,     # 输出项目目录
        "name": args.name,           # 实验名称
        "patience": args.patience,   # 早停耐心值
        "seed": args.seed,           # 随机种子
        "resume": args.resume,       # 恢复训练
        "cache": args.cache,         # 缓存
        "close_mosaic": args.close_mosaic,  # 最后N轮关闭马赛克
        "cos_lr": args.cos_lr,       # 余弦学习率
        "optimizer": args.optimizer, # 优化器

        # 以下是为车道线定制的数据增强参数：
        # HSV 增强：轻微改变色相、饱和度和明度
        # 车道线颜色相对稳定，所以色相变化很小
        "hsv_h": 0.015,              # HSV-H 增强幅度（很小，防止颜色偏移太大）
        "hsv_s": 0.5,                # HSV-S 增强幅度（中等）
        "hsv_v": 0.45,               # HSV-V 增强幅度（中等，模拟不同光照）
        # 几何增强
        "degrees": 2.0,              # 旋转角度（小，车道线通常是直的）
        "translate": 0.08,           # 平移（小偏移）
        "scale": 0.5,                # 缩放（中等范围）
        "fliplr": 0.5,               # 水平翻转概率（0.5 = 一半概率翻转）
    }

    if args.device is not None:
        train_args["device"] = args.device
    # 合并额外参数
    train_args.update(parse_extra(args.extra))

    # 开始训练
    model.train(**train_args)


if __name__ == "__main__":
    main()
```

### 关键概念讲解

1. **YOLO 分割模型**：YOLO（You Only Look Once）是实时目标检测的经典框架。分割版本（-seg）在检测基础上增加了实例分割能力，能输出每个目标的像素级掩码。对于车道线检测，分割优于检测，因为车道线的形状很重要。

2. **数据增强（Data Augmentation）**：训练时对图像进行随机变换，可以增加数据的多样性，提高模型泛化能力。针对车道线的增强策略需要特别设计：
   - 旋转角度小：车道线总是接近垂直，大角度旋转会产生不符合实际的样本
   - HSV 变化小：颜色是区分白线和黄线的关键，过大变化会混淆类别

3. **余弦退火（Cosine Annealing）学习率**：学习率按照余弦函数从大到小变化。早期学习率大（快速收敛），后期学习率小（精细调整）。`cos_lr` 启用此策略。

4. **早停（Early Stopping）**：`patience=40` 表示如果验证集指标连续 40 个 epoch 没有改善，就停止训练。这可以防止过拟合并节省计算资源。

5. **马赛克增强（Mosaic Augmentation）**：YOLO 的经典增强方法，将 4 张图像拼接为 1 张，增加小目标的检测能力。`close_mosaic=15` 表示在最后 15 个 epoch 关闭马赛克，因为马赛克图像会改变物体分布，在训练后期可能干扰收敛。

---

## 10. predict_yolo_lane.py

### 功能概述

这是项目中最核心的推理脚本。它加载训练好的 YOLO 分割模型，对图像进行推理，然后对每个检测到的车道线进行颜色分类（使用 HSV 或机器学习方法），输出结构化的 JSON 结果。

### 完整源码与注释

```python
# ============================================================
# predict_yolo_lane.py — YOLO 车道线推理与颜色分类
# ============================================================
# 这是项目中最复杂的文件，串联了多个子模块：
# 1. YOLO 分割模型检测 → 输出掩码和边界框
# 2. 颜色分类器（HSV 或 ML）→ 判断白/黄
# 3. 几何工具 → 从掩码拟合直线、计算角度
# 4. 可视化 → 可选保存标注图像
# 5. CSV 输出 → 可选保存统计表格
#
# 五种颜色分类模式（--class-mode）：
# - "model": 使用 YOLO 自身的类预测（需要训练了两类模型）
# - "hsv": 使用固定/自适应 HSV 阈值（默认）
# - "ml": 使用训练好的逻辑回归模型
# - "auto": 根据 YOLO 的输出自动选择模式
# - "hsv-refine": 先用 YOLO 预测，再用 HSV 修正
#
# 交叉文件依赖：
# - geometry.py: LineInstance、fit_line_from_points 等
# - color_classifier.py: 各种颜色分类方法
# - classes.py: 类别常量和规范化函数
# - 输出被 evaluate_lane_metrics.py 和 apply_count_constraints.py 使用

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .classes import EVAL_CLASSES, LANE_LINE, UNKNOWN, class_id_to_name, is_eval_class, normalize_class_name
from .color_classifier import adaptive_classify_lane_color, classify_lane_color, learned_classify_lane_color
from .geometry import LineInstance, fit_line_from_points, line_from_bbox_xyxy, points_from_mask


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO lane inference and classify white/yellow lanes.")
    parser.add_argument("--weights", required=True, help="YOLO weights path.")
    parser.add_argument("--source", required=True, help="Image, directory, or glob pattern.")
    parser.add_argument("--out", default="predictions.json", help="Output JSON path.")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--class-mode", choices=("auto", "model", "hsv", "ml"), default="hsv")
    parser.add_argument("--color-model", default="color_classifier.pkl")
    parser.add_argument("--color-scaler", default="color_scaler.pkl")
    parser.add_argument("--hsv-refine", action="store_true")
    parser.add_argument("--keep-unknown", action="store_true")
    parser.add_argument("--save-vis", default=None)
    parser.add_argument("--counts-out", default=None)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--roi-fraction", type=float, default=0.50)
    parser.add_argument("--adaptive-hsv", action="store_true")
    return parser.parse_args()

def image_files_from_source(source: str) -> list[Path] | None:
    """
    从输入路径获取所有图像文件列表。

    支持三种输入：
    1. 单个文件 → 返回包含该文件的列表
    2. 目录 → 递归查找所有图像文件
    3. 其他 → 返回 None（尝试交给 YOLO 的 glob 匹配）

    Parameters:
        source: 文件路径或目录路径

    Returns:
        图像文件列表，或 None
    """
    path = Path(source)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    return None


def mask_for_box(shape: tuple[int, int], xyxy: np.ndarray) -> np.ndarray:
    """
    从边界框生成一个矩形掩码。

    当分割掩码不可用时（比如检测模型而不是分割模型），
    用边界框区域作为掩码的近似。

    这是一个"降级方案"——矩形掩码没有精确形状，
    但至少能提供一个粗略的区域用于颜色分析。
    """
    h, w = shape
    x1, y1, x2, y2 = xyxy.astype(int)
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    mask = np.zeros((h, w), dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1: y2 + 1, x1: x2 + 1] = True
    return mask


def decide_class(
    image_bgr: np.ndarray,
    region_mask: np.ndarray,
    model_class: str,
    class_mode: str,
    hsv_refine: bool,
    adaptive_hsv: bool = False,
    color_model: str = "color_classifier.pkl",
    color_scaler: str = "color_scaler.pkl",
    bbox: np.ndarray | None = None,
) -> tuple[str, dict]:
    """
    根据配置的颜色分类模式决定车道线的类别。

    这是颜色分类的分发中心，支持多种模式：

    1. model 模式：直接使用 YOLO 自己的分类结果
       - 如果 YOLO 输出了 "white_lane" 就认为是白色
       - 不需要额外的颜色分类步骤

    2. hsv 模式：使用 HSV 阈值判断
       - 固定阈值或自适应阈值
       - 不依赖训练数据

    3. ml 模式：使用训练好的逻辑回归模型
       - 需要先训练 color_classifier.pkl
       - 理论上最准确

    4. auto 模式：智能选择
       - 如果 YOLO 已经分出了白/黄，就用 model 模式
       - 否则用 hsv 模式

    5. hsv_refine：用 HSV 修正 YOLO 的分类（不管 YOLO 的输出是什么）

    Parameters:
        image_bgr: BGR 图像
        region_mask: 车道线区域掩码
        model_class: YOLO 模型输出的类别
        class_mode: 颜色分类模式
        hsv_refine: 是否用 HSV 修正
        adaptive_hsv: 是否使用自适应 HSV 阈值

    Returns:
        (类别名, 颜色信息字典)
    """
    normalized_model_class = normalize_class_name(model_class)

    # mode="model" 且 YOLO 已经分出白/黄 → 直接用 YOLO 的结果
    if class_mode == "model" and normalized_model_class in EVAL_CLASSES:
        return normalized_model_class, {}
    if class_mode == "model":
        return normalized_model_class, {}

    # mode="ml" → 使用机器学习分类器
    if class_mode == "ml":
        decision = learned_classify_lane_color(image_bgr, region_mask, bbox,
                                                model_path=color_model, scaler_path=color_scaler)
        return decision.cls, {
            "color_score": decision.score,
            "white_fraction": decision.white_fraction,
            "yellow_fraction": decision.yellow_fraction,
        }

    # mode="hsv" 或 hsv_refine 或 auto（且 YOLO 没分出白/黄）
    need_hsv = class_mode == "hsv" or hsv_refine or (
        class_mode == "auto" and normalized_model_class not in EVAL_CLASSES
    )
    if need_hsv:
        if adaptive_hsv:
            decision = adaptive_classify_lane_color(image_bgr, region_mask)
        else:
            decision = classify_lane_color(image_bgr, region_mask)
        return decision.cls, {
            "color_score": decision.score,
            "white_fraction": decision.white_fraction,
            "yellow_fraction": decision.yellow_fraction,
        }

    # 默认：返回原始类别
    return normalized_model_class, {}


def draw_instance(image: np.ndarray, inst: LineInstance) -> None:
    """
    在图像上绘制检测结果（用于可视化）。

    颜色编码：
    - 白色线: 灰色（(240, 240, 240)）
    - 黄色线: 亮黄色（(0, 220, 255)）
    - 未知: 灰色（(160, 160, 160)）

    绘制内容：
    - 边界框（矩形）
    - 线段（粗线）
    - 标签文字（类别+置信度+角度）
    """
    import cv2

    color = (240, 240, 240) if inst.cls == "white_lane" else (0, 220, 255)
    if inst.cls == UNKNOWN:
        color = (160, 160, 160)

    p0 = tuple(int(round(v)) for v in inst.endpoints[0])
    p1 = tuple(int(round(v)) for v in inst.endpoints[1])
    x1, y1, x2, y2 = [int(round(v)) for v in inst.bbox]

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)     # 边界框
    cv2.line(image, p0, p1, color, 2)                       # 线段
    label = f"{inst.cls} {inst.conf:.2f} {inst.angle_deg:.1f}"
    cv2.putText(image, label, (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def write_counts_csv(images: dict[str, dict], path: Path) -> None:
    """
    将每张图片的车道线数量统计写入 CSV 文件。

    CSV 格式与 Excel 统计表相同：
    文件名, 车道线总数, 白线数, 黄线数

    encoding="utf-8-sig"：添加 BOM（Byte Order Mark），
    确保中文在 Excel 中能正常显示。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.writer(fp)
        writer.writerow(["文件名", "车道线数", "白线数", "黄线数"])
        for filename in sorted(images):
            instances = images[filename].get("instances", [])
            white = sum(1 for inst in instances if inst.get("class") == "white_lane")
            yellow = sum(1 for inst in instances if inst.get("class") == "yellow_lane")
            writer.writerow([filename, white + yellow, white, yellow])


def predict_one_result(result, model_names: dict[int, str], args: argparse.Namespace) -> tuple[str, dict]:
    """
    处理单个 YOLO 推理结果。

    这是推理流程中"逐结果处理"的核心函数。
    对于每个检测到的目标：
    1. 提取分割掩码
    2. 从掩码中拟合直线
    3. 判断颜色（调用 decide_class）
    4. 应用 ROI 过滤（只保留道路区域的目标）
    5. 构建 LineInstance 对象

    Parameters:
        result: YOLO 推理结果
        model_names: 类别 ID → 名称的映射
        args: 命令行参数

    Returns:
        (文件名, 结果字典)
    """
    import cv2

    image_path = Path(result.path)
    image_bgr = result.orig_img
    height, width = image_bgr.shape[:2]

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return image_path.name, {"width": width, "height": height, "instances": []}

    xyxy = boxes.xyxy.cpu().numpy()  # 边界框坐标
    confs = boxes.conf.cpu().numpy()  # 置信度
    class_ids = boxes.cls.cpu().numpy().astype(int)  # 类别 ID

    # 处理分割掩码
    masks = None
    if result.masks is not None and result.masks.data is not None:
        masks = result.masks.data.cpu().numpy()
        if masks.shape[1:3] != (height, width):
            resized = []
            for mask in masks:
                resized.append(cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST))
            masks = np.asarray(resized)

    instances: list[LineInstance] = []
    vis = image_bgr.copy() if args.save_vis else None

    for idx in range(len(xyxy)):
        # 获取分割掩码（如果可用），否则用边界框生成矩形掩码
        region_mask = masks[idx] > 0.5 if masks is not None else mask_for_box((height, width), xyxy[idx])

        # 从掩码中提取像素点，拟合直线
        points = points_from_mask(region_mask)
        fitted = fit_line_from_points(points)

        if fitted is None:
            # 拟合失败（点太少），用边界框生成直线
            angle, endpoints, bbox = line_from_bbox_xyxy(xyxy[idx])
        else:
            angle, endpoints, bbox = fitted

        # 确定类别
        source_class = class_id_to_name(int(class_ids[idx]), model_names)
        cls, color_info = decide_class(
            image_bgr, region_mask, source_class,
            args.class_mode, args.hsv_refine, args.adaptive_hsv,
            args.color_model, args.color_scaler, xyxy[idx],
        )

        # 过滤不需要保留的检测
        if not args.keep_unknown and (cls == UNKNOWN or (not is_eval_class(cls) and cls != LANE_LINE)):
            continue
        if cls == LANE_LINE and not args.keep_unknown:
            continue

        # ROI 过滤：只保留道路区域内的检测
        if args.roi_fraction > 0 and args.roi_fraction < 1:
            bbox_center_y = (bbox[1] + bbox[3]) / 2.0  # 边界框中心 y 坐标
            roi_top = height * (1.0 - args.roi_fraction)  # ROI 区域的顶部边界
            if bbox_center_y < roi_top:
                continue  # 中心在 ROI 上方 → 不在道路区域

        # 构建 LineInstance
        inst = LineInstance(
            cls=cls,
            conf=float(confs[idx]),
            angle_deg=angle,
            endpoints=endpoints,
            bbox=bbox,
            source_class=source_class,
            color_score=color_info.get("color_score"),
            white_fraction=color_info.get("white_fraction"),
            yellow_fraction=color_info.get("yellow_fraction"),
        )
        instances.append(inst)

        if vis is not None:
            draw_instance(vis, inst)

    # 保存可视化图像
    if vis is not None:
        save_dir = Path(args.save_vis)
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / image_path.name), vis)

    return image_path.name, {
        "width": width,
        "height": height,
        "instances": [inst.to_dict() for inst in instances],
    }


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    # 加载模型
    model = YOLO(args.weights)

    # 获取类别名称映射（不同 YOLO 版本格式可能不同）
    if isinstance(model.names, dict):
        model_names = {int(k): str(v) for k, v in model.names.items()}
    else:
        model_names = {i: str(v) for i, v in enumerate(model.names)}

    # YOLO 推理参数
    predict_kwargs = {
        "source": args.source,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "stream": True,          # 流式模式（逐张处理，不一次加载所有图像）
        "max_det": args.max_det, # 最大检测数
        "verbose": False,        # 不打印详细信息
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    # 如果源是目录或文件，获取文件总数供进度条使用
    files = image_files_from_source(args.source)
    total = len(files) if files is not None else None

    images: dict[str, dict] = {}

    # 推理并处理每一张图像
    t_start = time.perf_counter()
    for result in tqdm(model.predict(**predict_kwargs), total=total, desc="Predicting"):
        key, payload = predict_one_result(result, model_names, args)
        images[key] = payload
    t_total = time.perf_counter() - t_start

    n_images = len(images)
    avg_ms = (t_total / n_images * 1000) if n_images > 0 else 0.0

    # 构建输出
    output = {
        "meta": {
            "weights": args.weights,
            "source": args.source,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "class_mode": args.class_mode,
            "hsv_refine": args.hsv_refine,
            "roi_fraction": args.roi_fraction,
            "model_names": model_names,
            "timing": {
                "total_seconds": round(t_total, 3),
                "num_images": n_images,
                "avg_ms_per_image": round(avg_ms, 1),
            },
        },
        "images": images,
    }

    # 保存
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved predictions to {out_path}")
    print(f"Timing: {n_images} images in {t_total:.1f}s, avg {avg_ms:.1f} ms/image")

    if args.counts_out:
        counts_path = Path(args.counts_out)
        write_counts_csv(images, counts_path)
        print(f"Saved count CSV to {counts_path}")


if __name__ == "__main__":
    main()
```

### 关键概念讲解

1. **YOLO 推理流程**：模型前向传播 → NMS（非极大值抑制）去除重复检测 → 输出边界框、置信度、类别 ID、分割掩码。`stream=True` 模式逐张处理图像，适合处理大量图像。

2. **颜色分类模式**：项目的核心设计之一是"分阶段"的颜色分类——先用 YOLO 检测车道线（不分颜色），再用颜色分类器判断白/黄。这种"检测+分类"的解耦设计比端到端的两类检测更灵活。

3. **ROI（感兴趣区域）限制**：`roi_fraction=0.50` 意味着只保留图像下半部分的检测。这是合理的先验知识：车道线不会出现在天空或建筑物区域。

4. **JSON 输出结构**：输出包含 meta（模型信息、参数、计时）和 images（每张图片的检测结果）。这种结构化的输出便于后续的评估和约束处理。

5. **tqdm 进度条**：显示推理进度、速度、剩余时间，对处理大量图像很有用。

---

## 11. evaluate_lane_metrics.py

### 功能概述

评估车道线检测和颜色分类的性能。支持两种评估模式：基于位置匹配的精确评估（需要 YOLO 格式的 GT 标签）和基于数量统计的粗略评估（只用 Excel 中的 count 标注）。输出 precision、recall、F1 等指标。

### 完整源码与注释

```python
# ============================================================
# evaluate_lane_metrics.py — 车道线检测评估指标
# ============================================================
# 评估目标：白线和黄线检测的准确率、召回率和 F1 分数。
#
# 两种评估模式：
#
# 模式1：位置匹配评估（需要 GT 标签文件）
#   - 对每一张图片，将预测结果与 GT 标签进行匹配
#   - 匹配条件：相同类别 + 角度差 < 阈值 + 位置接近
#   - 可以评估角度准确性（15度规则）
#   - 需要 YOLO 格式的分割/检测标签
#
# 模式2：数量统计评估（只用 Excel 中的 GT counts）
#   - 比较预测的数量和 GT 数量
#   - 不能评估角度和位置准确性
#   - 但只需要 Excel 统计表，不需要精确的 GT 标签
#
# 关键评估指标：
# - Precision（精确率）：正确检测数 / 总检测数
#   "模型说这是白线，模型有多大概率是对的？"
# - Recall（召回率）：正确检测数 / GT 总数
#   "所有的白线里，模型找到了多少？"
# - F1：Precision 和 Recall 的调和平均
#   "综合来看，模型表现如何？"
#
# 交叉文件依赖：
# - classes.py: 类别常量和 is_eval_class
# - geometry.py: LineInstance, 几何匹配工具
# - xlsx_counts.py: 读取 GT count

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

from .classes import EVAL_CLASSES, names_from_yaml, normalize_class_name
from .geometry import (
    LineInstance,
    angle_diff_deg,
    bbox_iou_xyxy,
    center_distance,
    find_image_by_stem,
    read_yolo_label_file,
)
from .xlsx_counts import read_count_json, read_count_xlsx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate white/yellow lane detections.")
    parser.add_argument("--pred", required=True, help="predictions.json from src.predict_yolo_lane")
    parser.add_argument("--data", default=None, help="Optional YOLO data yaml.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--gt-label-dir", default=None)
    parser.add_argument("--gt-counts", default=None)
    parser.add_argument("--gt-xlsx", default=None)
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--angle-thr", type=float, default=15.0)
    parser.add_argument("--conf-thr", type=float, default=0.25)
    parser.add_argument("--max-center-dist-ratio", type=float, default=0.08)
    parser.add_argument("--min-bbox-iou", type=float, default=0.01)
    parser.add_argument("--ignore-distance", action="store_true")
    parser.add_argument("--out", default="metrics.json")
    return parser.parse_args()

def resolve_from_data_yaml(data_yaml: Path, split: str) -> tuple[Path | None, Path | None, dict[int, str]]:
    """
    从 YOLO 的 data.yaml 中解析图像目录、标签目录和类别名称。

    YOLO data.yaml 格式示例：
    ```
    path: /path/to/dataset
    train: images/train
    val: images/val
    names:
      0: white_lane
      1: yellow_lane
    ```

    我们通过将 images 替换为 labels 来推断标签目录位置。
    这是 YOLO 数据集的惯例：标签目录与图像目录结构相同，
    只是将 "images" 替换为 "labels"。

    Parameters:
        data_yaml: data.yaml 文件路径
        split: 分集名称（train/val/test）

    Returns:
        (image_dir, label_dir, names) 三个值的元组
    """
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = names_from_yaml(data.get("names"))
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    split_value = data.get(split)
    image_dir = None
    label_dir = None
    if split_value is not None:
        image_dir = Path(split_value)
        if not image_dir.is_absolute():
            image_dir = (root / image_dir).resolve()
        # 通过替换 "images" 为 "labels" 推断标签目录
        parts = list(image_dir.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            label_dir = Path(*parts)
    return image_dir, label_dir, names


def load_predictions(pred_path: Path, conf_thr: float) -> dict[str, list[LineInstance]]:
    """
    加载预测结果，按文件名主名组织，过滤掉低置信度和非评估类别的实例。

    注意：这里将所有 key 统一为文件名主名（stem），
    因为 GT 标签也是按 stem 组织的。这样可以确保 key 的一致性。
    """
    raw = json.loads(pred_path.read_text(encoding="utf-8"))
    if "images" not in raw:
        raise ValueError(f"{pred_path} must contain an 'images' object.")

    by_stem: dict[str, list[LineInstance]] = {}
    for key, payload in raw["images"].items():
        stem = Path(key).stem  # 统一为文件名主名
        instances = []
        for item in payload.get("instances", []):
            inst = LineInstance.from_dict(item)
            inst.cls = normalize_class_name(inst.cls)
            # 只保留置信度 >= 阈值且属于评估类别的实例
            if inst.conf >= conf_thr and inst.cls in EVAL_CLASSES:
                instances.append(inst)
        by_stem[stem] = instances
    return by_stem


def load_ground_truth(
    label_dir: Path,
    image_dir: Path,
    names: dict[int, str],
) -> tuple[dict[str, list[LineInstance]], dict[str, tuple[int, int]]]:
    """
    加载真实标注（GT）标签文件。

    对每个标签文件：
    1. 找到对应的图像文件
    2. 读取图像尺寸（YOLO 坐标需要图像尺寸才能反归一化）
    3. 读取并解析 YOLO 标签
    4. 返回按文件名主名组织的字典和图像尺寸字典

    Parameters:
        label_dir: 标签目录
        image_dir: 图像目录
        names: 类别 ID 到名称的映射

    Returns:
        (gt_by_stem, sizes) — GT 实例字典和图像尺寸字典
    """
    if not label_dir.exists():
        raise FileNotFoundError(f"GT label dir not found: {label_dir}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")

    import cv2

    gt_by_stem: dict[str, list[LineInstance]] = {}
    sizes: dict[str, tuple[int, int]] = {}

    for label_path in sorted(label_dir.rglob("*.txt")):
        # 找到对应的图像文件
        image_path = find_image_by_stem(image_dir, label_path.stem)
        if image_path is None:
            raise FileNotFoundError(f"No image found for label stem {label_path.stem!r} under {image_dir}")

        # 读取图像获取尺寸
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        height, width = image.shape[:2]

        # 读取 YOLO 标签
        gt_by_stem[label_path.stem] = read_yolo_label_file(label_path, width, height, names)
        sizes[label_path.stem] = (width, height)

    return gt_by_stem, sizes


def can_match(
    pred: LineInstance,
    gt: LineInstance,
    image_diag: float,  # 图像对角线长度
    args: argparse.Namespace,
) -> tuple[bool, float, dict[str, float]]:
    """
    判断一个预测实例是否能与一个 GT 实例匹配。

    匹配条件（同时满足）：
    1. 类别相同（白线配白线，黄线配黄线）
    2. 角度差 <= angle_thr（默认 15 度）
    3. 位置接近：中心距离 <= max_center_dist_ratio * 图像对角线
       或 IoU >= min_bbox_iou

    匹配分数（用于选择最佳匹配）：
    score = angle_diff + 3.0 * norm_dist - iou
    分数越小表示匹配越好：
    - 角度差越小越好
    - 距离越小越好
    - IoU 越大越好（减号）

    Parameters:
        pred: 预测实例
        gt: GT 实例
        image_diag: 图像对角线长度（像素）
        args: 命令行参数

    Returns:
        (是否匹配, 匹配分数, 详细信息的字典)
    """
    angle_diff = angle_diff_deg(pred.angle_deg, gt.angle_deg)
    iou = bbox_iou_xyxy(pred.bbox, gt.bbox)
    dist = center_distance(pred.bbox, gt.bbox)
    max_dist = args.max_center_dist_ratio * image_diag

    angle_ok = angle_diff <= args.angle_thr
    distance_ok = args.ignore_distance or dist <= max_dist or iou >= args.min_bbox_iou

    ok = pred.cls == gt.cls and angle_ok and distance_ok

    # 归一化距离（用于评分）
    norm_dist = dist / max(max_dist, 1e-6)
    score = angle_diff + 3.0 * norm_dist - iou

    details = {"angle_diff": angle_diff, "bbox_iou": iou, "center_dist": dist}
    return ok, score, details


def match_image(
    preds: list[LineInstance],
    gts: list[LineInstance],
    image_size: tuple[int, int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """
    对单张图片进行预测与 GT 的匹配。

    匹配算法（贪心匹配）：
    1. 按置信度从高到低遍历预测
    2. 对每个预测，找到最佳匹配的 GT（已匹配的不再使用）
    3. 记录匹配结果

    为什么用贪心匹配而不是匈牙利算法？
    - 车道线数量通常很少（<10），贪心匹配已经足够好
    - 实现简单，不容易出错
    - 匈牙利算法（最优匹配）在数量很少时优势不明显

    Parameters:
        preds: 预测实例列表
        gts: GT 实例列表
        image_size: 图像尺寸 (width, height)
        args: 命令行参数

    Returns:
        匹配结果列表
    """
    width, height = image_size
    image_diag = math.hypot(width, height)
    matches: list[dict[str, Any]] = []
    used_gt: set[int] = set()  # 已匹配的 GT 索引

    # 按置信度降序排列预测
    for pred_idx, pred in sorted(enumerate(preds), key=lambda item: item[1].conf, reverse=True):
        best: tuple[float, int, dict[str, float]] | None = None
        for gt_idx, gt in enumerate(gts):
            if gt_idx in used_gt:
                continue  # 这个 GT 已经被匹配过了
            ok, score, details = can_match(pred, gt, image_diag, args)
            if not ok:
                continue
            if best is None or score < best[0]:  # 分数越小越好
                best = (score, gt_idx, details)

        if best is not None:
            _, gt_idx, details = best
            used_gt.add(gt_idx)
            matches.append({
                "pred_index": pred_idx,
                "gt_index": gt_idx,
                "class": pred.cls,
                **details,
            })
    return matches


def safe_div(num: float, den: float) -> float:
    """
    安全的除法，避免除以零。

    Parameters:
        num: 分子
        den: 分母

    Returns:
        商，如果分母为零则返回 0.0
    """
    return 0.0 if den == 0 else num / den


def print_table(metrics: dict[str, dict[str, float]]) -> None:
    """
    打印格式化的评估指标表格。

    表头：类别 | 检测数 | 正确数 | GT数 | Precision | Recall | F1
    """
    header = f"{'class':<14}{'detected':>10}{'correct':>10}{'gt':>10}{'precision':>12}{'recall':>10}{'f1':>10}"
    print(header)
    print("-" * len(header))
    for cls, row in metrics.items():
        print(
            f"{cls:<14}"
            f"{int(row['detected']):>10}"
            f"{int(row['correct']):>10}"
            f"{int(row['gt']):>10}"
            f"{row['precision']:>12.4f}"
            f"{row['recall']:>10.4f}"
            f"{row['f1']:>10.4f}"
        )


def empty_counts() -> dict[str, int]:
    """返回只有 EVAL_CLASSES 的零计数字典。"""
    return {cls: 0 for cls in EVAL_CLASSES}


def load_gt_counts(args: argparse.Namespace) -> dict[str, dict[str, int]]:
    """
    加载 GT 计数标注（来自 JSON 或 Excel）。

    Returns:
        {文件名主名: {"white_lane": 数量, "yellow_lane": 数量, "lane_line": 数量}}
    """
    if args.gt_counts:
        raw = read_count_json(Path(args.gt_counts))
    elif args.gt_xlsx:
        raw = read_count_xlsx(Path(args.gt_xlsx))
    else:
        raise ValueError("Count-only evaluation needs --gt-counts or --gt-xlsx.")
    return {Path(filename).stem: counts for filename, counts in raw.items()}


def evaluate_count_only(args: argparse.Namespace, pred_path: Path) -> None:
    """
    基于数量统计的评估模式。

    不检查位置和角度，只比较各类别的数量。
    适用于只有 Excel 统计表、没有 YOLO 标签数据的场景。

    评价逻辑：
    - correct = min(pred_count, gt_count)
      "预测了 3 条白线，GT 说 2 条 → 最多 2 条是对的"
    - precision = correct / detected
    - recall = correct / gt

    缺陷：
    - 不能检测"误判"（比如把黄线检测成白线但数量刚好对上）
    - 不能验证 15 度角规则
    """
    preds_by_stem = load_predictions(pred_path, args.conf_thr)
    gt_counts_by_stem = load_gt_counts(args)

    counts = {cls: {"detected": 0, "correct": 0, "gt": 0} for cls in EVAL_CLASSES}
    image_reports: dict[str, Any] = {}

    all_stems = sorted(set(gt_counts_by_stem) | set(preds_by_stem))
    for stem in all_stems:
        preds = preds_by_stem.get(stem, [])
        # 统计预测的各类别数量
        pred_counts = empty_counts()
        for pred in preds:
            if pred.cls in pred_counts:
                pred_counts[pred.cls] += 1

        # GT 数量
        gt_raw = gt_counts_by_stem.get(stem, {})
        gt_counts = {cls: int(gt_raw.get(cls, 0)) for cls in EVAL_CLASSES}

        # 正确数 = min(预测数, GT数)
        correct_counts = {
            cls: min(pred_counts[cls], gt_counts[cls])
            for cls in EVAL_CLASSES
        }

        for cls in EVAL_CLASSES:
            counts[cls]["detected"] += pred_counts[cls]
            counts[cls]["gt"] += gt_counts[cls]
            counts[cls]["correct"] += correct_counts[cls]

        image_reports[stem] = {
            "pred_counts": pred_counts,
            "gt_counts": gt_counts,
            "correct_counts": correct_counts,
        }

    # 计算指标
    metrics: dict[str, dict[str, float]] = {}
    total = {"detected": 0, "correct": 0, "gt": 0}
    for cls, row in counts.items():
        precision = safe_div(row["correct"], row["detected"])
        recall = safe_div(row["correct"], row["gt"])
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        metrics[cls] = {**row, "precision": precision, "recall": recall, "f1": f1}
        for key in total:
            total[key] += row[key]

    precision = safe_div(total["correct"], total["detected"])
    recall = safe_div(total["correct"], total["gt"])
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    metrics["overall"] = {**total, "precision": precision, "recall": recall, "f1": f1}

    output = {
        "settings": {
            "mode": "count_only",
            "conf_thr": args.conf_thr,
            "note": "Count-only GT cannot verify the 15-degree angle rule.",
        },
        "metrics": metrics,
        "images": image_reports,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WARNING: count-only GT cannot verify the 15-degree angle rule.")
    print_table(metrics)
    print(f"Saved metrics to {out_path}")


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred)

    # 如果使用了 count-only 相关参数，进入数量评估模式
    if args.count_only or args.gt_counts or args.gt_xlsx:
        evaluate_count_only(args, pred_path)
        return

    # --- 位置匹配评估模式 ---
    image_dir = Path(args.image_dir).resolve() if args.image_dir else None
    label_dir = Path(args.gt_label_dir).resolve() if args.gt_label_dir else None
    names: dict[int, str] = {}

    # 如果提供了 data.yaml，从中解析目录和名称
    if args.data:
        data_image_dir, data_label_dir, names = resolve_from_data_yaml(Path(args.data), args.split)
        image_dir = image_dir or data_image_dir
        label_dir = label_dir or data_label_dir

    if image_dir is None or label_dir is None:
        raise ValueError("Please provide --data, or both --image-dir and --gt-label-dir.")

    # 加载预测和 GT
    preds_by_stem = load_predictions(pred_path, args.conf_thr)
    gt_by_stem, sizes = load_ground_truth(label_dir, image_dir, names)

    # 初始化计数
    counts = {cls: {"detected": 0, "correct": 0, "gt": 0} for cls in EVAL_CLASSES}
    image_reports: dict[str, Any] = {}

    # 遍历所有图片
    all_stems = sorted(set(gt_by_stem) | set(preds_by_stem))
    for stem in all_stems:
        preds = preds_by_stem.get(stem, [])
        gts = gt_by_stem.get(stem, [])
        size = sizes.get(stem)
        if size is None:
            # 只有预测没有 GT 的图片：检测数计入，但不能判断是否正确
            size = (1, 1)

        # 统计各类别检测数
        for cls in EVAL_CLASSES:
            counts[cls]["detected"] += sum(1 for pred in preds if pred.cls == cls)
            counts[cls]["gt"] += sum(1 for gt in gts if gt.cls == cls)

        # 匹配预测和 GT
        matches = match_image(preds, gts, size, args)
        for match in matches:
            counts[match["class"]]["correct"] += 1

        image_reports[stem] = {
            "pred_count": len(preds),
            "gt_count": len(gts),
            "matches": matches,
        }

    # 计算指标
    metrics: dict[str, dict[str, float]] = {}
    total = {"detected": 0, "correct": 0, "gt": 0}
    for cls, row in counts.items():
        precision = safe_div(row["correct"], row["detected"])
        recall = safe_div(row["correct"], row["gt"])
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        metrics[cls] = {**row, "precision": precision, "recall": recall, "f1": f1}
        for key in total:
            total[key] += row[key]

    precision = safe_div(total["correct"], total["detected"])
    recall = safe_div(total["correct"], total["gt"])
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    metrics["overall"] = {**total, "precision": precision, "recall": recall, "f1": f1}

    # 保存结果
    output = {
        "settings": {
            "angle_thr": args.angle_thr,
            "conf_thr": args.conf_thr,
            "max_center_dist_ratio": args.max_center_dist_ratio,
            "min_bbox_iou": args.min_bbox_iou,
            "ignore_distance": args.ignore_distance,
        },
        "metrics": metrics,
        "images": image_reports,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print_table(metrics)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
```

### 关键概念讲解

1. **Precision（精确率） vs Recall（召回率）**：
   - 精确率 = 正确预测数 / 总预测数：模型说这是白线，可信度有多高？
   - 召回率 = 正确预测数 / GT 总数：所有白线中，模型找到了多少？
   - 两者通常需要权衡：过于激进的检测（什么都说是线）会有高召回但低精确率

2. **F1 分数**：Precision 和 Recall 的调和平均。公式：F1 = 2 * (P * R) / (P + R)。相比算术平均，调和平均对低值更敏感——如果 P 或 R 任一很低，F1 也会很低。

3. **贪心匹配（Greedy Matching）**：先匹配置信度最高的预测，然后是次高的，依此类推。虽然不一定能找到全局最优匹配，但在车道线数量少、清晰度高的情况下已经足够好。

4. **数量评估的局限性**：count_only 模式只能评估数量准确性，不能评估位置和角度。例如：模型预测了 2 条白线，GT 也说有 2 条，看起来完美，但模型可能把 2 条黄线错认为了白线。这就是为什么只有在缺乏精确标签时才使用 count_only 模式。

---

## 12. apply_count_constraints.py

### 功能概述

利用 GT 计数标注来约束 YOLO 的预测结果。核心思路：对于每张图片，GT 告诉我们有多少条白线、多少条黄线。我们从 YOLO 的候选检测中挑选"最可能"的 K 条，确保各类别数量与 GT 一致。

### 完整源码与注释

```python
# ============================================================
# apply_count_constraints.py — 用量级标注约束预测结果
# ============================================================
# 这是项目中"弱监督"思想的最后一步。
#
# 问题：YOLO 的预测可能包含很多候选，有些是真线，有些是误检。
# 但我们知道准确的数量（GT count）。
#
# 解决方案：从候选中选择最可能的前 K 条，使各类别数量匹配 GT。
# 这类似于"用人头数来估算人数"——虽然不知道每个具体位置，
# 但总数对了，整体统计就是可信的。
#
# 约束流程：
# 1. 对每个候选检测，计算"白色得分"和"黄色得分"
# 2. 按类别顺序（可选 rare-first / yellow-first / white-first）
# 3. 对每个类别，选取得分最高的 N 条（N = GT count）
# 4. 标记选取的检测及其分配的颜色
# 5. 输出约束后的预测
#
# 为什么这个文件在 pipeline 末尾？
# 因为在训练了颜色分类器之后，我们可以用约束后的结果来"修正"颜色分类，
# 输出更准确的统计报表。
#
# 但要注意：约束后的结果使用了 GT 信息，不能作为"纯模型性能"的报告依据。
#
# 交叉文件依赖：
# - 输入来自 predict_yolo_lane.py 的输出
# - 调用 xlsx_counts.py 读取 GT counts
# - 输出被 train_color_classifier.py 用于训练

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .classes import WHITE, YELLOW, normalize_class_name
from .xlsx_counts import read_count_json, read_count_xlsx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use white/yellow count annotations to constrain prediction results."
    )
    parser.add_argument("--pred", required=True, help="Input predictions.json.")
    parser.add_argument("--out", default="predictions_count_constrained.json")
    parser.add_argument("--counts-out", default="prediction_counts_constrained.csv")
    parser.add_argument("--gt-xlsx", default=None, help="Count spreadsheet.")
    parser.add_argument("--gt-counts", default=None, help="Count JSON.")
    parser.add_argument("--min-conf", type=float, default=0.0)
    parser.add_argument("--keep-unmatched", action="store_true",
                        help="Keep predictions for images that have no count GT.")
    parser.add_argument("--class-order", choices=("rare-first", "yellow-first", "white-first"),
                        default="rare-first",
                        help="Order used when assigning classes to candidates.")
    return parser.parse_args()


def load_gt_counts(args: argparse.Namespace) -> dict[str, dict[str, int]]:
    """
    加载 GT 计数。支持 JSON 和 Excel 两种格式。
    统一按文件名主名（stem）索引。
    """
    if args.gt_counts:
        raw = read_count_json(Path(args.gt_counts))
    elif args.gt_xlsx:
        raw = read_count_xlsx(Path(args.gt_xlsx))
    else:
        raise ValueError("Please provide --gt-xlsx or --gt-counts.")
    return {Path(filename).stem: counts for filename, counts in raw.items()}


def safe_float(value: Any, default: float = 0.0) -> float:
    """安全地转换为浮点数。"""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def color_scores(instance: dict[str, Any]) -> dict[str, float]:
    """
    计算一个候选检测的白色得分和黄色得分。

    得分综合考虑了三个因素：
    1. conf（模型置信度）：模型对这个检测的可信度
    2. white_fraction / yellow_fraction（颜色分析）：HSV/ML 分析的颜色像素比例
    3. 原始类别加分：如果 YOLO 原本就预测为某类，给予 0.15 的加分

    白色得分公式：
    white_score = conf * max(0, white_fraction + 0.15(YOLO加分) - 0.05*yellow_fraction)

    为什么减去一部分对方分数？这是一个"竞争机制"——
    如果某个区域的黄色像素也不少，白色得分会适当降低。
    类似的想法：两个类别互相竞争。

    Parameters:
        instance: 预测实例字典

    Returns:
        {"white_lane": 白色得分, "yellow_lane": 黄色得分, "candidate": 综合候选得分}
    """
    conf = safe_float(instance.get("conf"), 1.0)
    white_fraction = safe_float(instance.get("white_fraction"), 0.0)
    yellow_fraction = safe_float(instance.get("yellow_fraction"), 0.0)
    cls = normalize_class_name(instance.get("class", ""))

    # 如果没有颜色分析数据（比如 class_mode="model"），
    # 直接用 YOLO 的原始类别作为打分依据
    if white_fraction == 0.0 and yellow_fraction == 0.0:
        white_fraction = 1.0 if cls == WHITE else 0.0
        yellow_fraction = 1.0 if cls == YELLOW else 0.0

    # 类别加分：YOLO 的原预测类别有 0.15 的加分
    white_bonus = 0.15 if cls == WHITE else 0.0
    yellow_bonus = 0.15 if cls == YELLOW else 0.0

    # 分数计算（加入竞争机制）
    white_score = conf * max(0.0, white_fraction + white_bonus - 0.05 * yellow_fraction)
    yellow_score = conf * max(0.0, yellow_fraction + yellow_bonus - 0.05 * white_fraction)

    return {
        WHITE: white_score,
        YELLOW: yellow_score,
        "candidate": conf * (max(white_fraction, yellow_fraction) + 0.05),
    }


def ordered_classes(target_counts: dict[str, int], order: str) -> list[str]:
    """
    决定类别选择的顺序。

    为什么顺序重要？因为如果我们先选黄色，再选白色，
    和先选白色再选黄色，结果可能不同。
    如果某个候选在两类上得分都很高，先选哪一类就决定了它的归属。

    三种顺序：
    - rare-first: 稀有类别优先（靠数量排序，少的先选）
    - yellow-first: 黄色优先
    - white-first: 白色优先

    Parameters:
        target_counts: 各类别的 GT 数量
        order: 排序策略

    Returns:
        类别列表（按选择顺序）
    """
    if order == "yellow-first":
        return [YELLOW, WHITE]
    if order == "white-first":
        return [WHITE, YELLOW]
    # rare-first: 按数量升序（少的优先）+ 类别名次排序
    return sorted([WHITE, YELLOW], key=lambda cls: (target_counts.get(cls, 0), cls))

def constrain_instances(
    instances: list[dict[str, Any]],    # 原始预测实例列表
    target_counts: dict[str, int],      # GT 数量目标
    *,
    min_conf: float,                    # 最低置信度
    class_order: str,                   # 类别选择顺序
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    对单张图片的预测实例进行计数约束。

    这是整个文件的核心算法。算法流程：

    1. 收集所有置信度 >= min_conf 的候选检测
    2. 对每个候选计算白/黄/综合得分
    3. 按类别顺序遍历（如 rare-first）
    4. 对每个类别，从"尚未选取"的候选中选择得分最高的 N 条
    5. N = GT 中该类别的数量
    6. 为选中的实例打上 count_constraint 标记

    示例：
    - 候选检测：A(白0.8), B(黄0.6), C(白0.3), D(黄0.2)
    - GT counts：白=1, 黄=1
    - 按 rare-first（假设白和黄数量都是1，按字母 white < yellow）
    - 先选白色：A（得分0.8）
    - 再选黄色：从{未选中的B/C/D}中选 B（得分0.6）
    - 结果：[A(white), B(yellow)]

    Parameters:
        instances: 预测实例列表
        target_counts: GT 数量 {"white_lane": N, "yellow_lane": N}
        min_conf: 最低置信度阈值
        class_order: 类别排序策略

    Returns:
        (selected_instances, report) — 约束后的实例列表和统计报告
    """
    # 构建候选列表：(索引, 实例, 得分字典)
    candidates = [
        (idx, inst, color_scores(inst))
        for idx, inst in enumerate(instances)
        if safe_float(inst.get("conf"), 1.0) >= min_conf
    ]

    selected_indices: set[int] = set()  # 已选取的索引
    selected: list[dict[str, Any]] = []  # 选取的实例列表
    assignments: dict[int, str] = {}     # 索引 → 类别映射

    # 按类别顺序遍历
    for cls in ordered_classes(target_counts, class_order):
        need = max(0, int(target_counts.get(cls, 0)))  # 需要多少个
        if need == 0:
            continue  # 不需要这个类别的线

        # 从"尚未选取"的候选中选择
        available = [
            (scores[cls], scores["candidate"], idx, inst)
            for idx, inst, scores in candidates
            if idx not in selected_indices
        ]
        # 排序：先按类别得分，再按综合候选得分，再按置信度
        available.sort(reverse=True, key=lambda item: (item[0], item[1], safe_float(item[3].get("conf"), 0.0)))

        # 选取前 need 个
        for _, _, idx, inst in available[:need]:
            selected_indices.add(idx)
            assignments[idx] = cls
            # 深拷贝实例，修改 class 为分配的颜色
            constrained = deepcopy(inst)
            constrained["class"] = cls
            constrained["count_constraint"] = {
                "target_class": cls,           # 分配的目标类别
                "source_class": normalize_class_name(inst.get("class", "")),  # 原始类别
                "white_score": color_scores(inst)[WHITE],   # 白色得分
                "yellow_score": color_scores(inst)[YELLOW], # 黄色得分
            }
            selected.append(constrained)

    # 按置信度降序排列
    selected.sort(key=lambda inst: safe_float(inst.get("conf"), 0.0), reverse=True)

    # 统计报告
    pred_counts = {
        WHITE: sum(1 for inst in selected if normalize_class_name(inst.get("class", "")) == WHITE),
        YELLOW: sum(1 for inst in selected if normalize_class_name(inst.get("class", "")) == YELLOW),
    }
    report = {
        "candidate_count": len(candidates),
        "kept_count": len(selected),
        "dropped_count": max(0, len(candidates) - len(selected)),
        "target_counts": {
            WHITE: int(target_counts.get(WHITE, 0)),
            YELLOW: int(target_counts.get(YELLOW, 0)),
        },
        "pred_counts": pred_counts,
        "shortage": {
            WHITE: max(0, int(target_counts.get(WHITE, 0)) - pred_counts[WHITE]),
            YELLOW: max(0, int(target_counts.get(YELLOW, 0)) - pred_counts[YELLOW]),
        },
        "assignments": assignments,
    }
    return selected, report


def write_counts_csv(images: dict[str, dict[str, Any]], path: Path) -> None:
    """
    将约束后的预测结果写入 CSV 统计表。

    格式与 predict_yolo_lane.py 的 write_counts_csv 相同。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.writer(fp)
        writer.writerow(["文件名", "车道线数", "白线数", "黄线数"])
        for filename in sorted(images, key=lambda name: (Path(name).stem.zfill(8), name)):
            instances = images[filename].get("instances", [])
            white = sum(1 for inst in instances if normalize_class_name(inst.get("class", "")) == WHITE)
            yellow = sum(1 for inst in instances if normalize_class_name(inst.get("class", "")) == YELLOW)
            writer.writerow([filename, white + yellow, white, yellow])


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred)
    predictions = json.loads(pred_path.read_text(encoding="utf-8"))
    gt_counts = load_gt_counts(args)

    # 深拷贝原始预测，在此基础上进行约束
    output = deepcopy(predictions)
    output.setdefault("meta", {})
    output["meta"]["count_constraints"] = {
        "enabled": True,
        "gt_xlsx": args.gt_xlsx,
        "gt_counts": args.gt_counts,
        "min_conf": args.min_conf,
        "class_order": args.class_order,
        # 重要提示：约束后的结果使用了 GT 信息，不能作为模型性能报告
        "note": "This uses count-level GT as a constraint; do not report it as unconstrained model performance.",
    }

    reports: dict[str, Any] = {}
    constrained_images: dict[str, dict[str, Any]] = {}
    matched = 0

    # 对每张图片应用约束
    for filename, payload in predictions.get("images", {}).items():
        stem = Path(filename).stem
        target = gt_counts.get(stem)  # 查找 GT 数量

        if target is None:
            # 没有 GT 数量的图片：如果设置 keep_unmatched 则保留原样
            if args.keep_unmatched:
                constrained_images[filename] = payload
            reports[stem] = {"matched_gt": False}
            continue

        matched += 1
        instances = payload.get("instances", [])
        selected, report = constrain_instances(
            instances,
            target,
            min_conf=args.min_conf,
            class_order=args.class_order,
        )
        new_payload = deepcopy(payload)
        new_payload["instances"] = selected
        constrained_images[filename] = new_payload
        reports[stem] = {"matched_gt": True, **report}

    output["images"] = constrained_images
    output["constraint_report"] = reports

    # 保存结果
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_counts_csv(constrained_images, Path(args.counts_out))

    print(f"Loaded predictions: {len(predictions.get('images', {}))} images")
    print(f"Matched count GT: {matched}/{len(predictions.get('images', {}))} images")
    print(f"Saved constrained predictions to {out_path}")
    print(f"Saved constrained counts to {args.counts_out}")
    print("NOTE: constrained results use count-level GT and should be described as weakly supervised/post-processed.")


if __name__ == "__main__":
    main()
```

### 关键概念讲解

1. **计数约束（Count Constraint）**：利用弱监督信息（数量）来优化强监督输出（检测结果）。这在实际工程项目中非常实用，因为"数数"比"画框"容易得多——标注者可以很快地说出"这张图有3条白线、2条黄线"，但精确标注每条线需要大量时间。

2. **得分竞争机制**：`color_scores` 函数中，白色得分减去了一部分黄色分数（`- 0.05 * yellow_fraction`）。这模拟了两个类别的竞争——如果一个区域既有白色特征又有黄色特征，它更可能是哪一种？这种"互斥"设计可以避免同一区域被同时判定为白和黄。

3. **类别顺序的影响**：`rare-first` 策略先分配"更稀有"的类别。如果黄线较少，优先分配黄色就能确保稀缺类别的要求先被满足。这是一个有趣的博弈论问题——先手优势。

4. **深拷贝（deepcopy）**：在修改实例时使用 `deepcopy` 确保原始预测数据不被修改。这是函数式编程的良好实践——输入不变，输出是新对象。

5. **关于约束结果的描述**：代码明确在 meta 中注明了 "do not report it as unconstrained model performance"。这是一个重要的学术诚信提醒——使用了额外信息（GT counts）的方法应该被描述为"后处理"或"弱监督"，而不是"纯模型性能"。

---

## 项目总结

### 完整 Pipeline 流程

```
1. prepare_local_dataset.py
   原始 ZIP + Excel → 解压、划分、生成 data.yaml + gt_counts.json

2. generate_pseudo_labels.py
   无标签图像 → Canny+Hough → YOLO 分割伪标签

3. train_yolo.py (第一次)
   伪标签 → 训练 YOLO 分割模型 → best.pt

4. predict_yolo_lane.py (第一次)
   best.pt → 推理 → predictions_raw.json

5. apply_count_constraints.py (第一次)
   predictions_raw.json + gt_counts → predictions_constrained.json

6. train_color_classifier.py
   predictions_raw + predictions_constrained → 训练 ML 颜色分类器 → color_classifier.pkl

7. predict_yolo_lane.py (第二次，使用 ML 分类器)
   best.pt + color_classifier.pkl → 推理 → predictions_final.json

8. evaluate_lane_metrics.py
   predictions_final.json + GT → metrics.json + 指标报告

9. refine_labels_from_gt.py (可选)
   从约束预测中生成高质量 YOLO 标签，用于迭代训练
```

### 文件依赖关系图

```
classes.py  ←── geometry.py ←── xlsx_counts.py
    ↑               ↑               ↑
    │               │               │
    └── color_classifier.py ────────┤
            ↑                       │
            │                       │
    train_color_classifier.py       │
                                    │
prepare_local_dataset.py ───────────┤
                                    │
generate_pseudo_labels.py           │
                                    │
train_yolo.py                       │
                                    │
predict_yolo_lane.py ───────────────┤
                                    │
evaluate_lane_metrics.py ───────────┤
                                    │
apply_count_constraints.py ─────────┘
                                    │
refine_labels_from_gt.py ───────────┘
```

### 项目中使用的核心 ML/CV 概念

| 概念 | 出现文件 | 用途 |
|------|----------|------|
| Canny 边缘检测 | generate_pseudo_labels.py | 检测图像边缘 |
| Hough 变换 | generate_pseudo_labels.py | 从边缘检测直线 |
| HSV 颜色空间 | color_classifier.py | 基于颜色的车道线分类 |
| SVD 直线拟合 | geometry.py | 从掩码点拟合车道线 |
| IoU 交并比 | geometry.py, evaluate_lane_metrics.py | 边界框匹配与评估 |
| YOLO 分割模型 | train_yolo.py, predict_yolo_lane.py | 车道线检测与分割 |
| 逻辑回归 | train_color_classifier.py | 有监督颜色分类 |
| 交叉验证 | train_color_classifier.py | 模型评估 |
| 特征标准化 | train_color_classifier.py | 特征预处理 |
| Precision/Recall/F1 | evaluate_lane_metrics.py | 评估指标 |
| 弱监督学习 | apply_count_constraints.py, refine_labels_from_gt.py | 用弱标注生成强标签 |
| 贪心匹配 | evaluate_lane_metrics.py | 预测与 GT 匹配 |
| 自训练 | train_color_classifier.py | 用模型输出训练自身 |
| 数据增强 | train_yolo.py | 提高泛化能力 |
| 早停 | train_yolo.py | 防止过拟合 |
| 形态学操作 | color_classifier.py | 膨胀运算计算对比度 |
| Douglas-Peucker | refine_labels_from_gt.py | 多边形简化 |
