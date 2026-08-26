"""Ingest your own labeled footage into a WiderFaceDataset-compatible layout.

Input layout (--source):
    my_footage/
      images/frame_0001.jpg, frame_0002.jpg, ...
      labels/frame_0001.txt, frame_0002.txt, ...

Each label file has one face per line: "x1 y1 x2 y2" (pixel coordinates,
top-left/bottom-right corners). An image with zero faces still needs an
empty (0-byte) label file, not a missing one - a missing file is treated
as a labeling mistake and raises an error rather than being silently
assumed to mean "no faces".

Output: data/widerface/<name>/ with the exact same wider_face_split/*_bbx_gt.txt
+ WIDER_<split>/images/<event>/*.jpg structure the real WIDER FACE dataset
uses, so it loads directly via WiderFaceDataset(root=..., split=...) - and
can be combined with the real dataset for joint training via
torch.utils.data.ConcatDataset([wider_ds, my_footage_ds]).

Usage:
    .venv\\Scripts\\python.exe scripts\\import_my_footage.py --source my_footage --split train
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVENT_NAME = "0--MyFootage"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def convert(source: Path, dest_root: Path, split: str) -> None:
    images_src = source / "images"
    labels_src = source / "labels"
    if not images_src.is_dir() or not labels_src.is_dir():
        raise FileNotFoundError(f"expected both {images_src} and {labels_src} to exist")

    image_paths = sorted(p for ext in IMAGE_EXTENSIONS for p in images_src.glob(f"*{ext}"))
    if not image_paths:
        raise FileNotFoundError(f"no {IMAGE_EXTENSIONS} images found in {images_src}")

    dest_images = dest_root / f"WIDER_{split}" / "images" / EVENT_NAME
    dest_images.mkdir(parents=True, exist_ok=True)
    split_dir = dest_root / "wider_face_split"
    split_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    num_images = num_boxes = 0

    for img_path in image_paths:
        label_path = labels_src / f"{img_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(
                f"missing label file for {img_path.name}: expected {label_path} "
                "(use an empty file for images with zero faces, not a missing one)"
            )

        shutil.copy2(img_path, dest_images / img_path.name)

        boxes: list[tuple[float, float, float, float]] = []
        text = label_path.read_text().strip()
        if text:
            for line_num, line in enumerate(text.splitlines(), start=1):
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError(f"{label_path}:{line_num}: expected 'x1 y1 x2 y2', got {line!r}")
                x1, y1, x2, y2 = (float(p) for p in parts)
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    raise ValueError(f"{label_path}:{line_num}: degenerate box (w={w}, h={h}): {line!r}")
                boxes.append((x1, y1, w, h))

        lines.append(f"{EVENT_NAME}/{img_path.name}")
        lines.append(str(len(boxes)))
        if boxes:
            for x, y, w, h in boxes:
                # x y w h blur expression illumination occlusion pose invalid - the
                # attribute columns are dummy zeros; only x/y/w/h are real here.
                lines.append(f"{int(x)} {int(y)} {int(w)} {int(h)} 0 0 0 0 0 0")
        else:
            lines.append("0 0 0 0 0 0 0 0 0 0")  # the official format's own zero-face-line quirk

        num_images += 1
        num_boxes += len(boxes)

    annotation_path = split_dir / f"wider_face_{split}_bbx_gt.txt"
    annotation_path.write_text("\n".join(lines) + "\n")

    print(f"Imported {num_images} images, {num_boxes} faces -> {dest_root}")
    print(f"Load it with: WiderFaceDataset(root={str(dest_root)!r}, split={split!r}, ...)")
    print(
        "Combine with the real dataset for joint training via:\n"
        "  torch.utils.data.ConcatDataset([wider_face_dataset, my_footage_dataset])"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="folder containing images/ and labels/ subfolders")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--name", default="MyFootage", help="output dataset folder name under data/widerface/")
    args = parser.parse_args()

    dest_root = REPO_ROOT / "data" / "widerface" / args.name
    convert(Path(args.source), dest_root, args.split)


if __name__ == "__main__":
    main()
