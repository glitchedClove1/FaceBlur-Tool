"""Augmentation pipelines for training and evaluation.

Every toggle and probability comes from config (never hardcoded mid-function)
so experiments can be run by editing YAML, not code. Both pipelines operate
on (image, bboxes) jointly via albumentations so boxes stay correct through
crops/flips/resizes.
"""

from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

# Neutral gray padding, matching the common YOLO-family convention - chosen
# so letterbox borders don't look like "black edges" the network could
# learn to associate with frame boundaries.
_PAD_VALUE = (114, 114, 114)

# Magnitude constants for the noise/blur augmentations. These are the "how
# strong" knobs; config controls "on/off" and "how often" (probability) per
# the project brief, but a fixed mild magnitude is enough here (RGB is only
# barely disturbed with these values, by design).
_GAUSSIAN_BLUR_MIN = 3
_GAUSS_NOISE_STD_RANGE = (0.02, 0.08)


def _bbox_params() -> A.BboxParams:
    return A.BboxParams(
        format="pascal_voc",
        label_fields=["labels"],
        min_visibility=0.2,  # drop boxes that a crop reduced to <20% of their original visible area
        clip=True,           # clamp box coords into the image after crop/pad, instead of erroring
        filter_invalid_bboxes=True,
    )


def _train_augment_steps(cfg: dict) -> list[A.BasicTransform]:
    """The augmentation steps shared by build_train_transform and the
    dataset-visualization script, which needs the augmented image as
    viewable uint8 (i.e. without Normalize/ToTensorV2 applied)."""
    aug = cfg["augmentation"]
    size = cfg["data"]["image_size"]

    steps: list[A.BasicTransform] = []

    if aug["horizontal_flip"]["enabled"]:
        steps.append(A.HorizontalFlip(p=aug["horizontal_flip"]["prob"]))

    if aug["random_crop"]["enabled"]:
        # The "zoom" augmentation: crops a random bbox-safe region then
        # resizes it to (size, size), which is what lets the network see
        # small faces at a larger effective scale during training.
        steps.append(
            A.RandomSizedBBoxSafeCrop(
                height=size,
                width=size,
                erosion_rate=aug["random_crop"]["erosion_rate"],
                p=aug["random_crop"]["prob"],
            )
        )

    # Deterministic letterbox resize to the fixed training resolution. Runs
    # regardless of whether the crop above fired - when it did, the image is
    # already size x size and this becomes a cheap no-op; when it didn't
    # (or the source image is smaller/differently shaped), this is what
    # guarantees every sample reaches the exact same resolution.
    steps.append(A.LongestMaxSize(max_size=size))
    steps.append(
        A.PadIfNeeded(
            min_height=size,
            min_width=size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=_PAD_VALUE,
            position="random",  # extra augmentation: face position jitters within the padded frame
        )
    )

    if aug["color_jitter"]["enabled"]:
        cj = aug["color_jitter"]
        steps.append(
            A.ColorJitter(
                brightness=cj["brightness"],
                contrast=cj["contrast"],
                saturation=cj["saturation"],
                hue=cj["hue"],
                p=cj["prob"],
            )
        )

    blur_noise_options = []
    if aug["blur"]["enabled"]:
        blur_noise_options.append(
            A.GaussianBlur(blur_limit=(_GAUSSIAN_BLUR_MIN, aug["blur"]["blur_limit"]), p=1.0)
        )
    if aug["noise"]["enabled"]:
        blur_noise_options.append(A.GaussNoise(std_range=_GAUSS_NOISE_STD_RANGE, p=1.0))
    if blur_noise_options:
        combined_p = max(aug["blur"]["prob"], aug["noise"]["prob"])
        steps.append(A.OneOf(blur_noise_options, p=combined_p))

    return steps


def build_train_transform(cfg: dict) -> A.Compose:
    norm = cfg["augmentation"]["normalize"]
    steps = _train_augment_steps(cfg) + [
        A.Normalize(mean=norm["mean"], std=norm["std"]),
        ToTensorV2(),
    ]
    return A.Compose(steps, bbox_params=_bbox_params())


def build_train_visualization_transform(cfg: dict) -> A.Compose:
    """Same augmentation as build_train_transform, but stops before
    Normalize/ToTensorV2 so the output stays a viewable uint8 image."""
    return A.Compose(_train_augment_steps(cfg), bbox_params=_bbox_params())


def build_eval_transform(cfg: dict) -> A.Compose:
    """Resize + normalize only - no randomness, so eval numbers are stable."""
    size = cfg["data"]["image_size"]
    norm = cfg["augmentation"]["normalize"]

    steps = [
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(
            min_height=size,
            min_width=size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=_PAD_VALUE,
            position="center",  # deterministic - engine/inference.py must replicate this exact padding to decode boxes back to original coordinates
        ),
        A.Normalize(mean=norm["mean"], std=norm["std"]),
        ToTensorV2(),
    ]
    return A.Compose(steps, bbox_params=_bbox_params())


if __name__ == "__main__":
    # Smoke test: build both pipelines on a synthetic image + boxes and
    # confirm shapes come out as expected, without needing real data.
    import numpy as np
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "default.yaml").read_text())

    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    boxes = np.array([[100, 100, 200, 220], [400, 300, 430, 340]], dtype=np.float32)
    labels = np.array([1, 1], dtype=np.int64)

    for name, builder in [("train", build_train_transform), ("eval", build_eval_transform)]:
        transform = builder(cfg)
        out = transform(image=image, bboxes=boxes, labels=labels)
        size = cfg["data"]["image_size"]
        assert out["image"].shape == (3, size, size), out["image"].shape
        print(f"[{name}] image shape={tuple(out['image'].shape)}, boxes={out['bboxes']}")

    print("transforms.py smoke test passed")
