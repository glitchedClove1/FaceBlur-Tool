"""Anchor generation, box encode/decode, and anchor<->GT matching.

Anchors are square (aspect ratio 1:1) since faces are near-square; letting
box regression handle the rest is simpler than adding aspect-ratio variants
and is what RetinaFace does for the same reason. min_sizes per stride are
RetinaFace's WIDER FACE config (data.image_size scale), chosen because it's
tuned for exactly this dataset's face-scale distribution rather than a
generic COCO-style anchor set.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchvision.ops import box_iou


def generate_anchors(image_size: int, strides: list[int], min_sizes: list[list[int]]) -> Tensor:
    """Returns [N, 4] anchors in (cx, cy, w, h) pixel coordinates at image_size scale."""
    if image_size % max(strides) != 0:
        raise ValueError(f"image_size={image_size} must be divisible by the largest stride={max(strides)}")

    anchors = []
    for stride, sizes in zip(strides, min_sizes):
        feat_size = image_size // stride
        # centers of each stride x stride cell, in input-pixel coordinates
        centers = (torch.arange(feat_size, dtype=torch.float32) + 0.5) * stride
        cy, cx = torch.meshgrid(centers, centers, indexing="ij")
        cx = cx.reshape(-1)  # [feat_size * feat_size]
        cy = cy.reshape(-1)

        for size in sizes:
            w = torch.full_like(cx, float(size))
            h = torch.full_like(cx, float(size))
            anchors.append(torch.stack([cx, cy, w, h], dim=1))

    return torch.cat(anchors, dim=0)


def anchors_per_level(image_size: int, strides: list[int], min_sizes: list[list[int]]) -> list[int]:
    """Number of anchors contributed by each level - matches the concatenation order
    of generate_anchors, so model heads can split/reshape per level consistently."""
    counts = []
    for stride, sizes in zip(strides, min_sizes):
        feat_size = image_size // stride
        counts.append(feat_size * feat_size * len(sizes))
    return counts


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def encode(matched_gt_xyxy: Tensor, anchors_cxcywh: Tensor, variances: tuple[float, float]) -> Tensor:
    """GT boxes (xyxy) matched 1:1 to anchors -> regression targets, SSD-style."""
    gt = xyxy_to_cxcywh(matched_gt_xyxy)
    a_cx, a_cy, a_w, a_h = anchors_cxcywh.unbind(-1)
    g_cx, g_cy, g_w, g_h = gt.unbind(-1)

    tx = (g_cx - a_cx) / a_w / variances[0]
    ty = (g_cy - a_cy) / a_h / variances[0]
    tw = torch.log(g_w / a_w) / variances[1]
    th = torch.log(g_h / a_h) / variances[1]
    return torch.stack([tx, ty, tw, th], dim=-1)


def decode(deltas: Tensor, anchors_cxcywh: Tensor, variances: tuple[float, float]) -> Tensor:
    """Regression outputs -> boxes (xyxy), inverse of encode()."""
    a_cx, a_cy, a_w, a_h = anchors_cxcywh.unbind(-1)
    tx, ty, tw, th = deltas.unbind(-1)

    g_cx = tx * variances[0] * a_w + a_cx
    g_cy = ty * variances[0] * a_h + a_cy
    g_w = torch.exp(tw * variances[1]) * a_w
    g_h = torch.exp(th * variances[1]) * a_h
    return cxcywh_to_xyxy(torch.stack([g_cx, g_cy, g_w, g_h], dim=-1))


def match(anchors_xyxy: Tensor, gt_boxes_xyxy: Tensor, pos_iou_threshold: float, neg_iou_threshold: float) -> tuple[Tensor, Tensor]:
    """Assigns each anchor a label and a matched GT index for one image.

    Returns:
        labels: [N] long tensor, 1 = positive, 0 = negative, -1 = ignore
        matched_gt_idx: [N] long tensor, index into gt_boxes_xyxy (0 where labels != 1)
    """
    num_anchors = anchors_xyxy.shape[0]
    device = anchors_xyxy.device

    if gt_boxes_xyxy.numel() == 0:
        # No faces in this image (kept as a hard-negative background sample) - every anchor is negative.
        labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        matched_gt_idx = torch.zeros(num_anchors, dtype=torch.long, device=device)
        return labels, matched_gt_idx

    iou = box_iou(anchors_xyxy, gt_boxes_xyxy)  # [N, M]
    max_iou_per_anchor, matched_gt_idx = iou.max(dim=1)

    labels = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    labels[max_iou_per_anchor < neg_iou_threshold] = 0
    labels[max_iou_per_anchor >= pos_iou_threshold] = 1

    # Guarantee every GT has at least one positive anchor (its best-IoU match),
    # even if that IoU falls under pos_iou_threshold - otherwise small/unusual
    # faces with no "good enough" anchor would never get a training signal.
    best_anchor_per_gt = iou.argmax(dim=0)  # [M]
    labels[best_anchor_per_gt] = 1
    matched_gt_idx[best_anchor_per_gt] = torch.arange(gt_boxes_xyxy.shape[0], device=device)

    return labels, matched_gt_idx


if __name__ == "__main__":
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "default.yaml").read_text())
    image_size = cfg["data"]["image_size"]
    strides = cfg["anchors"]["strides"]
    min_sizes = cfg["anchors"]["min_sizes"]
    variances = tuple(cfg["anchors"]["variances"])

    anchors = generate_anchors(image_size, strides, min_sizes)
    counts = anchors_per_level(image_size, strides, min_sizes)
    print(f"image_size={image_size}, strides={strides}, min_sizes={min_sizes}")
    print(f"anchors per level: {counts}, total: {anchors.shape[0]}")
    assert anchors.shape[0] == sum(counts)
    assert anchors.shape == (sum(counts), 4)

    # Encode/decode round-trip: a decoded-then-encoded box should recover
    # the original box (up to floating point error).
    anchors_xyxy = cxcywh_to_xyxy(anchors)
    fake_gt = anchors_xyxy[:5].clone()
    fake_gt[:, 2:] += 3.0  # nudge so it's not a degenerate zero-offset case
    deltas = encode(fake_gt, anchors[:5], variances)
    recovered = decode(deltas, anchors[:5], variances)
    max_error = (recovered - fake_gt).abs().max().item()
    print(f"encode/decode round-trip max error: {max_error:.6f}")
    assert max_error < 1e-3

    # Matching smoke test: a couple of synthetic GT boxes placed near known anchors.
    gt_boxes = torch.tensor([[100.0, 100.0, 132.0, 132.0], [200.0, 200.0, 264.0, 264.0]])
    labels, matched_idx = match(anchors_xyxy, gt_boxes, cfg["anchors"]["pos_iou_threshold"], cfg["anchors"]["neg_iou_threshold"])
    print(f"positives: {(labels == 1).sum().item()}, negatives: {(labels == 0).sum().item()}, ignored: {(labels == -1).sum().item()}")
    assert (labels == 1).sum() >= 2  # at least one positive per GT box

    # Empty-GT (background-only image) must not crash and must yield all negatives.
    labels_empty, _ = match(anchors_xyxy, torch.zeros((0, 4)), 0.5, 0.4)
    assert (labels_empty == 0).all()

    print("anchors.py smoke test passed")
