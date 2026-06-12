from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .classes import names_from_yaml, normalize_class_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy YOLO labels and merge white/yellow lane classes into one lane_line class."
    )
    parser.add_argument("--src-label-dir", required=True)
    parser.add_argument("--dst-label-dir", required=True)
    parser.add_argument("--data", default="configs/colm_lane.yaml", help="Original data yaml with class names.")
    parser.add_argument("--keep-road", action="store_true", help="Keep road_surface as class 1 in output labels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src_label_dir)
    dst = Path(args.dst_label_dir)
    data = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    names = names_from_yaml(data.get("names"))
    dst.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    for label_path in src.rglob("*.txt"):
        rel = label_path.relative_to(src)
        out_path = dst / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_lines = []
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            parts = stripped.split(maxsplit=1)
            class_id = int(float(parts[0]))
            class_name = normalize_class_name(names.get(class_id, str(class_id)))
            rest = parts[1] if len(parts) > 1 else ""
            if class_name in {"white_lane", "yellow_lane", "lane_line"}:
                out_lines.append(f"0 {rest}".rstrip())
                converted += 1
            elif args.keep_road and class_name == "road_surface":
                out_lines.append(f"1 {rest}".rstrip())
                converted += 1
            else:
                skipped += 1
        out_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

    print(f"Converted labels: {converted}")
    print(f"Skipped labels: {skipped}")
    print(f"Saved to: {dst}")


if __name__ == "__main__":
    main()
