"""Box post-processing shared by training visualization, evaluation, and
both inference apps: decode raw model outputs into (boxes, scores), clip to
the image, and run NMS. Face blurring lives here too since it's another
"operate on a decoded box" utility, used by both the video and webcam apps.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torchvision.ops import nms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.anchors import cxcywh_to_xyxy, decode


def clip_boxes_to_image(boxes: Tensor, width: int, height: int) -> Tensor:
    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
    return boxes


def postprocess(
    cls_logits: Tensor,
    bbox_deltas: Tensor,
    anchors_cxcywh: Tensor,
    variances: tuple[float, float],
    image_width: int,
    image_height: int,
    conf_thresh: float,
    nms_thresh: float,
    max_detections: int = 200,
) -> tuple[Tensor, Tensor]:
    """Single image: raw model outputs -> final (boxes[K,4] xyxy, scores[K]).

    cls_logits: [N], bbox_deltas: [N, 4], anchors_cxcywh: [N, 4]
    """
    scores = torch.sigmoid(cls_logits)
    keep_mask = scores > conf_thresh
    if not keep_mask.any():
        return torch.zeros((0, 4), device=cls_logits.device), torch.zeros((0,), device=cls_logits.device)

    scores = scores[keep_mask]
    boxes = decode(bbox_deltas[keep_mask], anchors_cxcywh[keep_mask], variances)
    boxes = clip_boxes_to_image(boxes, image_width, image_height)

    keep_idx = nms(boxes, scores, nms_thresh)[:max_detections]
    return boxes[keep_idx], scores[keep_idx]


def postprocess_batch(
    cls_logits: Tensor,
    bbox_deltas: Tensor,
    anchors_cxcywh: Tensor,
    variances: tuple[float, float],
    image_width: int,
    image_height: int,
    conf_thresh: float,
    nms_thresh: float,
    max_detections: int = 200,
) -> list[tuple[Tensor, Tensor]]:
    """Batched convenience wrapper - loops per-image (NMS/threshold counts are
    small enough that this isn't a bottleneck outside training's hot loop)."""
    return [
        postprocess(
            cls_logits[i], bbox_deltas[i], anchors_cxcywh, variances,
            image_width, image_height, conf_thresh, nms_thresh, max_detections,
        )
        for i in range(cls_logits.shape[0])
    ]


def draw_detections(image_bgr: np.ndarray, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Draw green boxes + confidence scores (the "off"/no-blur display mode),
    shared by the Gradio demo and both CLI apps. Returns a copy - the input
    frame is left untouched."""
    out = image_bgr.copy()
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(out, f"{score:.2f}", (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
    return out


def _pad_and_clip_box(box: np.ndarray, pad_frac: float, width: int, height: int) -> tuple[int, int, int, int]:
    """Expand a box by pad_frac of its own size (so hairline/chin/ears near
    the tight face box get covered too), then clip to the frame - boxes near
    the image border must not index outside the array."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = w * pad_frac, h * pad_frac
    x1 = int(max(0, x1 - pad_x))
    y1 = int(max(0, y1 - pad_y))
    x2 = int(min(width, x2 + pad_x))
    y2 = int(min(height, y2 + pad_y))
    return x1, y1, x2, y2


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def blur_faces(image_bgr: np.ndarray, boxes: np.ndarray, mode: str, pad_frac: float = 0.15) -> np.ndarray:
    """Blur every box region in-place-equivalent (returns a modified copy).

    mode: "gaussian" (kernel size scales with box size, so small and large
    faces are equally obscured - a fixed kernel would under-blur a large
    face and over-blur a tiny one) or "pixelate" (downscale-then-upscale
    mosaic). boxes: [N, 4] xyxy, any dtype.
    """
    if mode not in ("gaussian", "pixelate"):
        raise ValueError(f"mode must be 'gaussian' or 'pixelate', got {mode!r}")

    out = image_bgr.copy()
    height, width = out.shape[:2]

    for box in np.asarray(boxes):
        x1, y1, x2, y2 = _pad_and_clip_box(box, pad_frac, width, height)
        if x2 <= x1 or y2 <= y1:
            continue  # degenerate after clipping (box entirely outside the frame)

        region = out[y1:y2, x1:x2]
        w, h = x2 - x1, y2 - y1

        if mode == "gaussian":
            # Kernel proportional to box size, capped so huge faces don't
            # produce a huge (slow) kernel; floor of 3 keeps tiny faces blurred too.
            k = _odd(int(np.clip(min(w, h) * 0.5, 3, 99)))
            blurred = cv2.GaussianBlur(region, (k, k), 0)
        else:
            # Downscale to a coarse grid, then upscale with nearest-neighbor
            # for hard mosaic blocks - block count scales with box size so
            # small faces still get a handful of visible blocks.
            blocks = max(4, min(w, h) // 12)
            small = cv2.resize(region, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
            blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        out[y1:y2, x1:x2] = blurred

    return out


if __name__ == "__main__":
    from data.anchors import generate_anchors

    anchors = generate_anchors(384, [8, 16, 32], [[16, 32], [64, 128], [256, 512]])
    n = anchors.shape[0]

    torch.manual_seed(0)
    cls_logits = torch.randn(n) - 3.0  # mostly background, a few plausible positives
    bbox_deltas = torch.randn(n, 4) * 0.1

    boxes, scores = postprocess(
        cls_logits, bbox_deltas, anchors, variances=(0.1, 0.2),
        image_width=384, image_height=384, conf_thresh=0.5, nms_thresh=0.4,
    )
    print(f"kept {boxes.shape[0]} boxes after threshold+NMS")
    assert boxes.shape[1] == 4
    assert (boxes[:, 0] <= boxes[:, 2]).all() and (boxes[:, 1] <= boxes[:, 3]).all()
    assert (boxes >= 0).all() and (boxes <= 384).all()

    # Blur smoke test, including a box that hangs off the frame edge.
    dummy_frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    edge_boxes = np.array([[10, 10, 60, 60], [-20, 150, 40, 220]], dtype=np.float32)
    for mode in ("gaussian", "pixelate"):
        blurred_frame = blur_faces(dummy_frame, edge_boxes, mode=mode)
        assert blurred_frame.shape == dummy_frame.shape
    print("boxes.py smoke test passed")
