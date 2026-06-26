from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from pathlib import Path

from .xlsx_counts import read_count_xlsx, write_count_json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare image folders and count labels for the non-YOLO pipeline.")
    parser.add_argument("--root", default=".", help="Project root containing the example/test zips and count xlsx.")
    parser.add_argument("--example-zip", default=None, help="Training/example zip. Auto-detected when omitted.")
    parser.add_argument("--test-zip", default=None, help="Test zip. Auto-detected when omitted.")
    parser.add_argument("--gt-xlsx", default=None, help="Count spreadsheet. Auto-detected when omitted.")
    parser.add_argument("--out", default="datasets/local_colm")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def auto_find(root: Path, kind: str) -> Path | None:
    if kind == "example_zip":
        candidates = [p for p in root.glob("*.zip") if "example" in p.name.lower()]
    elif kind == "test_zip":
        candidates = [p for p in root.glob("*.zip") if "test" in p.name.lower()]
    elif kind == "gt_xlsx":
        candidates = list(root.glob("*.xlsx"))
    else:
        raise ValueError(kind)
    return sorted(candidates, key=lambda p: p.name)[0] if candidates else None


def zip_members(path: Path, extensions: set[str]) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(path) as zf:
        members = [
            info
            for info in zf.infolist()
            if not info.is_dir() and Path(info.filename).suffix.lower() in extensions
        ]
    return sorted(members, key=lambda info: info.filename)


def extract_member(zf: zipfile.ZipFile, member: zipfile.ZipInfo, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, dst.open("wb") as out:
        shutil.copyfileobj(src, out)


def copy_zip_images(
    zip_path: Path,
    out_dir: Path,
    split: str,
    image_members: list[zipfile.ZipInfo],
    *,
    clear: bool,
) -> list[str]:
    image_dir = out_dir / "images" / split
    if clear and image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in image_members:
            filename = Path(member.filename).name
            extract_member(zf, member, image_dir / filename)
            copied.append(filename)
    return copied


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()

    example_zip = Path(args.example_zip).resolve() if args.example_zip else auto_find(root, "example_zip")
    test_zip = Path(args.test_zip).resolve() if args.test_zip else auto_find(root, "test_zip")
    gt_xlsx = Path(args.gt_xlsx).resolve() if args.gt_xlsx else auto_find(root, "gt_xlsx")
    if example_zip is None or not example_zip.exists():
        raise FileNotFoundError("Could not find example zip. Pass --example-zip explicitly.")
    if test_zip is None or not test_zip.exists():
        raise FileNotFoundError("Could not find test zip. Pass --test-zip explicitly.")

    example_images = zip_members(example_zip, IMAGE_EXTENSIONS)
    test_images = zip_members(test_zip, IMAGE_EXTENSIONS)
    if not example_images:
        raise ValueError(f"No images found in {example_zip}")
    if not test_images:
        raise ValueError(f"No images found in {test_zip}")

    rng = random.Random(args.seed)
    shuffled = example_images[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * args.val_ratio))) if len(shuffled) > 1 else 0
    val_images = sorted(shuffled[:val_count], key=lambda info: info.filename)
    train_images = sorted(shuffled[val_count:], key=lambda info: info.filename)

    train_files = copy_zip_images(example_zip, out_dir, "train", train_images, clear=args.overwrite)
    val_files = copy_zip_images(example_zip, out_dir, "val", val_images, clear=args.overwrite)
    test_files = copy_zip_images(test_zip, out_dir, "test", test_images, clear=args.overwrite)

    gt_json = None
    if gt_xlsx is not None and gt_xlsx.exists():
        counts = read_count_xlsx(gt_xlsx)
        gt_json = out_dir / "gt_counts.json"
        write_count_json(counts, gt_json)

    print(f"Prepared dataset: {out_dir}")
    print(f"Train images: {len(train_files)}")
    print(f"Val images: {len(val_files)}")
    print(f"Test images: {len(test_files)}")
    if gt_json is not None:
        print(f"GT count json: {gt_json}")
    print("Non-YOLO pipeline ready: use src.train_color_model and src.run_classical_pipeline.")


if __name__ == "__main__":
    main()
