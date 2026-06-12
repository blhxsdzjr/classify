from __future__ import annotations

import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .classes import WHITE, YELLOW


NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values = []
    for item in root.findall("a:si", NS):
        values.append("".join(node.text or "" for node in item.findall(".//a:t", NS)))
    return values


def _cell_text(cell: ET.Element, shared: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value = cell.find("a:v", NS)
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//a:t", NS)).strip()
    if value is None or value.text is None:
        return ""
    text = value.text.strip()
    if cell_type == "s" and text:
        return shared[int(text)].strip()
    return text


def _column_name(ref: str) -> str:
    return "".join(ch for ch in ref if ch.isalpha())


def _to_int(value: Any) -> int:
    text = str(value).strip()
    if not text:
        return 0
    return int(float(text))


def read_count_xlsx(path: Path) -> dict[str, dict[str, int]]:
    """Read the course count template: 文件名 / 车道线数 / 白线数 / 黄线数."""
    with zipfile.ZipFile(path) as zf:
        shared = _shared_strings(zf)
        sheet_names = [
            name for name in zf.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        ]
        if not sheet_names:
            raise ValueError(f"No worksheet found in {path}")
        root = ET.fromstring(zf.read(sheet_names[0]))

    header_by_col: dict[str, str] = {}
    rows: dict[str, dict[str, int]] = {}

    for row in root.findall(".//a:row", NS):
        row_values: dict[str, str] = {}
        for cell in row.findall("a:c", NS):
            ref = cell.attrib.get("r", "")
            col = _column_name(ref)
            if col:
                row_values[col] = _cell_text(cell, shared)
        if not row_values:
            continue

        if not header_by_col and any(value == "文件名" for value in row_values.values()):
            header_by_col = {col: value for col, value in row_values.items()}
            continue

        if not header_by_col:
            continue

        values_by_header = {
            header_by_col[col]: value
            for col, value in row_values.items()
            if col in header_by_col
        }
        filename = values_by_header.get("文件名", "").strip()
        if not filename:
            continue
        white = _to_int(values_by_header.get("白线数", 0))
        yellow = _to_int(values_by_header.get("黄线数", 0))
        total = _to_int(values_by_header.get("车道线数", white + yellow))
        rows[filename] = {
            "lane_line": total,
            WHITE: white,
            YELLOW: yellow,
        }
    return rows


def write_count_json(counts: dict[str, dict[str, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")


def read_count_json(path: Path) -> dict[str, dict[str, int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(filename): {
            "lane_line": _to_int(values.get("lane_line", 0)),
            WHITE: _to_int(values.get(WHITE, values.get("white", 0))),
            YELLOW: _to_int(values.get(YELLOW, values.get("yellow", 0))),
        }
        for filename, values in raw.items()
    }
