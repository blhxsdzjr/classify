from __future__ import annotations

import argparse
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO lane-line detector/segmenter.")
    parser.add_argument("--data", default="datasets/local_colm/data.yaml", help="YOLO data yaml.")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="Pretrained YOLO model, e.g. yolov8n-seg.pt.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="CUDA device such as 0, 0,1 or cpu.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/segment")
    parser.add_argument("--name", default="colm_lane")
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache", action="store_true", help="Cache images if server RAM/disk allows it.")
    parser.add_argument("--close-mosaic", type=int, default=15)
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Ultralytics train args, e.g. hsv_v=0.6 degrees=2.0",
    )
    return parser.parse_args()


def parse_extra(extra: list[str]) -> dict:
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

    model = YOLO(args.model)
    train_args = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
        "seed": args.seed,
        "resume": args.resume,
        "cache": args.cache,
        "close_mosaic": args.close_mosaic,
        "cos_lr": args.cos_lr,
        "optimizer": args.optimizer,
        # Useful for lane markings under shadow and glare.
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.45,
        "degrees": 2.0,
        "translate": 0.08,
        "scale": 0.5,
        "fliplr": 0.5,
    }
    if args.device is not None:
        train_args["device"] = args.device
    train_args.update(parse_extra(args.extra))
    model.train(**train_args)


if __name__ == "__main__":
    main()
