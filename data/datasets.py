"""WIDER FACE dataset parser and torch Dataset.

Annotation format (wider_face_{train,val}_bbx_gt.txt), confirmed against a
real fixture of the official files:

    <image relative path, e.g. 0--Parade/0_Parade_marchingband_1_100.jpg>
    <N: number of faces>
    x y w h blur expression illumination occlusion pose invalid   (repeated N times)

When N == 0 the file still contains exactly one dummy box line
"0 0 0 0 0 0 0 0 0 0" - a real quirk of the released annotation files, not a
bug in this parser. It must be consumed and discarded like any other box
line, or parsing desyncs on every following image.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable, Optional, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

_INVALID_COL = 9  # index of the "invalid" flag within a 10-value box line


@dataclasses.dataclass
class FaceSample:
    image_path: Path
    boxes: np.ndarray    # [N, 4] float32, xyxy, pixel coords in the original image
    invalid: np.ndarray  # [N] bool, annotator-flagged invalid faces


def parse_bbx_gt(annotation_path: Path, images_root: Path) -> list[FaceSample]:
    lines = annotation_path.read_text().splitlines()
    samples: list[FaceSample] = []
    i = 0
    while i < len(lines):
        rel_path = lines[i].strip()
        i += 1
        num_boxes = int(lines[i].strip())
        i += 1

        num_lines_to_read = max(num_boxes, 1)  # the N==0 dummy-line quirk
        boxes, invalid = [], []
        for _ in range(num_lines_to_read):
            values = [int(v) for v in lines[i].split()]
            i += 1
            if num_boxes > 0:
                x, y, w, h = values[0:4]
                boxes.append([x, y, x + w, y + h])
                invalid.append(bool(values[_INVALID_COL]))

        samples.append(
            FaceSample(
                image_path=images_root / rel_path,
                boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                invalid=np.asarray(invalid, dtype=bool).reshape(-1),
            )
        )
    return samples


class WiderFaceDataset(Dataset):
    """torch Dataset over a WIDER FACE train/val split.

    __getitem__ returns a dict:
        image:  ndarray (pre-transform, HxWx3 BGR uint8) or transform output
        boxes:  [N, 4] float32 tensor, xyxy
        labels: [N] int64 tensor (always 1 - single "face" class)
        image_path: str
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        transform: Optional[Callable] = None,
        min_face_size: float = 0.0,
        drop_invalid: bool = True,
        drop_empty_images: bool = False,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        root = Path(root)
        annotation_path = root / "wider_face_split" / f"wider_face_{split}_bbx_gt.txt"
        images_root = root / f"WIDER_{split}" / "images"
        if not annotation_path.exists():
            raise FileNotFoundError(
                f"{annotation_path} not found. Run scripts/download_widerface.py first."
            )

        self.split = split
        self.transform = transform

        raw_samples = parse_bbx_gt(annotation_path, images_root)
        self.samples: list[FaceSample] = []
        for sample in raw_samples:
            boxes, invalid = sample.boxes, sample.invalid
            keep = np.ones(len(boxes), dtype=bool)
            if len(boxes):
                w = boxes[:, 2] - boxes[:, 0]
                h = boxes[:, 3] - boxes[:, 1]
                keep &= (w > 0) & (h > 0)  # always drop zero/negative-area boxes
                if min_face_size > 0:
                    keep &= (w >= min_face_size) & (h >= min_face_size)
                if drop_invalid:
                    keep &= ~invalid

            filtered = dataclasses.replace(sample, boxes=boxes[keep], invalid=invalid[keep])
            if drop_empty_images and len(filtered.boxes) == 0:
                continue
            self.samples.append(filtered)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = cv2.imread(str(sample.image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {sample.image_path}")

        boxes = sample.boxes.copy()
        labels = np.ones(len(boxes), dtype=np.int64)

        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed["image"]
            boxes = np.asarray(transformed["bboxes"], dtype=np.float32).reshape(-1, 4)
            labels = np.asarray(transformed["labels"], dtype=np.int64)

        return {
            "image": image,
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_path": str(sample.image_path),
        }


if __name__ == "__main__":
    # Smoke test: parse both splits (no transform) and report basic stats.
    # Requires the real dataset - run scripts/download_widerface.py first.
    root = Path(__file__).resolve().parent.parent / "data" / "widerface"

    for split in ("train", "val"):
        try:
            ds = WiderFaceDataset(root, split=split, min_face_size=4.0, drop_invalid=True)
        except FileNotFoundError as exc:
            print(f"[{split}] SKIPPED: {exc}")
            continue

        num_boxes = sum(len(s.boxes) for s in ds.samples)
        num_empty = sum(1 for s in ds.samples if len(s.boxes) == 0)
        print(f"[{split}] {len(ds)} images, {num_boxes} faces after filtering, {num_empty} images with 0 faces")

        item = ds[0]
        print(f"[{split}] sample 0: image shape {item['image'].shape}, "
              f"{item['boxes'].shape[0]} boxes, path={item['image_path']}")
