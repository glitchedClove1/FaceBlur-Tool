"""Box post-processing shared by training visualization, evaluation, and
both inference apps: decode raw model outputs into (boxes, scores), clip to
the image, and run NMS. Face blurring lives here too (added in the video/
webcam app phase) since it's another "operate on a decoded box" utility.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    print("boxes.py smoke test passed")
