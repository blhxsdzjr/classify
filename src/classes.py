from __future__ import annotations

from typing import Any, Mapping


WHITE = "white_lane"
YELLOW = "yellow_lane"
LANE_LINE = "lane_line"
ROAD = "road_surface"
UNKNOWN = "unknown"
EVAL_CLASSES = (WHITE, YELLOW)


_WHITE_NAMES = {
    "white",
    "white_lane",
    "white_line",
    "lane_white",
    "line_white",
    "bai",
    "bai_line",
}

_YELLOW_NAMES = {
    "yellow",
    "yellow_lane",
    "yellow_line",
    "lane_yellow",
    "line_yellow",
    "huang",
    "huang_line",
}

_LANE_NAMES = {
    "lane",
    "lane_line",
    "line",
    "road_line",
    "marking",
    "lane_marking",
}

_ROAD_NAMES = {
    "road",
    "road_surface",
    "surface",
    "pavement",
}


def normalize_class_name(name: Any) -> str:
    text = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if text in _WHITE_NAMES:
        return WHITE
    if text in _YELLOW_NAMES:
        return YELLOW
    if text in _LANE_NAMES:
        return LANE_LINE
    if text in _ROAD_NAMES:
        return ROAD
    return text or UNKNOWN


def names_from_yaml(raw_names: Any) -> dict[int, str]:
    if raw_names is None:
        return {}
    if isinstance(raw_names, Mapping):
        return {int(k): str(v) for k, v in raw_names.items()}
    if isinstance(raw_names, (list, tuple)):
        return {idx: str(name) for idx, name in enumerate(raw_names)}
    raise TypeError(f"Unsupported names format: {type(raw_names)!r}")


def class_id_to_name(class_id: int, names: Mapping[int, str] | None) -> str:
    if names and class_id in names:
        return normalize_class_name(names[class_id])
    return str(class_id)


def is_eval_class(name: str) -> bool:
    return normalize_class_name(name) in EVAL_CLASSES
