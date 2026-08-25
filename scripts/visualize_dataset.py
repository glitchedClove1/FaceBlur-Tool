"""Dump N training images with ground-truth boxes drawn, so parsing and the
augmentation pipeline can be eyeballed before spending any GPU time.

Usage:
    .venv\\Scripts\\python.exe scripts\\visualize_dataset.py
    .venv\\Scripts\\python.exe scripts\\visualize_dataset.py --split val --num 20 --raw
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.datasets import WiderFaceDataset
from data.transforms import build_train_visualization_transform

REPO_ROOT = Path(__file__).resolve().parent.parent
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2


def draw_boxes(image, boxes) -> None:
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), BOX_COLOR, BOX_THICKNESS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--out", default=str(REPO_ROOT / "outputs" / "dataset_viz"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--raw", action="store_true",
        help="Draw boxes on the untransformed original image instead of the augmented one "
             "(useful to isolate parsing bugs from augmentation bugs).",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = None if args.raw else build_train_visualization_transform(cfg)
    dataset = WiderFaceDataset(
        root=REPO_ROOT / cfg["data"]["root"],
        split=args.split,
        transform=transform,
        min_face_size=cfg["data"]["min_face_size"],
        drop_invalid=cfg["data"]["drop_invalid"],
        drop_empty_images=cfg["data"]["drop_empty_images"],
    )
    print(f"Loaded {len(dataset)} images from split={args.split!r}")

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), k=min(args.num, len(dataset)))

    for rank, index in enumerate(indices):
        item = dataset[index]
        image = item["image"]
        boxes = item["boxes"].numpy()

        # transform() output is HxWxC uint8 (no ToTensorV2 in the vis pipeline);
        # --raw uses the dataset's own untransformed BGR ndarray directly.
        image = image.copy()
        draw_boxes(image, boxes)

        out_path = out_dir / f"{rank:02d}_{Path(item['image_path']).stem}.jpg"
        cv2.imwrite(str(out_path), image)

    print(f"Wrote {len(indices)} annotated images to {out_dir}")


if __name__ == "__main__":
    main()
