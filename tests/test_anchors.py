"""Unit tests for anchor generation, box encode/decode, and matching."""

import torch

from data.anchors import cxcywh_to_xyxy, decode, encode, generate_anchors, match

STRIDES = [8, 16, 32]
MIN_SIZES = [[16, 32], [64, 128], [256, 512]]
IMAGE_SIZE = 384
VARIANCES = (0.1, 0.2)


def test_generate_anchors_count_and_shape():
    anchors = generate_anchors(IMAGE_SIZE, STRIDES, MIN_SIZES)
    # (48*48 + 24*24 + 12*12) positions * 2 sizes per level
    expected = (48 * 48 + 24 * 24 + 12 * 12) * 2
    assert anchors.shape == (expected, 4)


def test_generate_anchors_rejects_non_divisible_size():
    try:
        generate_anchors(385, STRIDES, MIN_SIZES)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_encode_decode_round_trip():
    anchors = generate_anchors(IMAGE_SIZE, STRIDES, MIN_SIZES)
    boxes_xyxy = cxcywh_to_xyxy(anchors[:10]).clone()
    boxes_xyxy[:, 2:] += 5.0  # perturb so targets aren't a trivial zero-delta case

    deltas = encode(boxes_xyxy, anchors[:10], VARIANCES)
    recovered = decode(deltas, anchors[:10], VARIANCES)

    assert torch.allclose(recovered, boxes_xyxy, atol=1e-3)


def test_match_assigns_positive_to_overlapping_gt():
    anchors = generate_anchors(IMAGE_SIZE, STRIDES, MIN_SIZES)
    anchors_xyxy = cxcywh_to_xyxy(anchors)

    # Exactly overlaps a known P3-level anchor (size 16 centered at (100,100)-ish grid point).
    gt_boxes = torch.tensor([[92.0, 92.0, 124.0, 124.0]])
    labels, matched_idx = match(anchors_xyxy, gt_boxes, pos_iou_threshold=0.5, neg_iou_threshold=0.4)

    assert (labels == 1).sum() >= 1
    positive_indices = (labels == 1).nonzero(as_tuple=True)[0]
    assert (matched_idx[positive_indices] == 0).all()


def test_match_every_gt_gets_at_least_one_positive():
    anchors = generate_anchors(IMAGE_SIZE, STRIDES, MIN_SIZES)
    anchors_xyxy = cxcywh_to_xyxy(anchors)

    # A deliberately awkward/tiny box unlikely to hit the 0.5 IoU threshold
    # anywhere - the "best anchor per GT" rule must still force a positive.
    gt_boxes = torch.tensor([[10.0, 10.0, 15.0, 15.0]])
    labels, matched_idx = match(anchors_xyxy, gt_boxes, pos_iou_threshold=0.5, neg_iou_threshold=0.4)

    assert (labels == 1).sum() >= 1


def test_match_empty_gt_is_all_negative():
    anchors = generate_anchors(IMAGE_SIZE, STRIDES, MIN_SIZES)
    anchors_xyxy = cxcywh_to_xyxy(anchors)

    labels, _ = match(anchors_xyxy, torch.zeros((0, 4)), pos_iou_threshold=0.5, neg_iou_threshold=0.4)

    assert (labels == 0).all()


def test_match_labels_are_valid_categories():
    anchors = generate_anchors(IMAGE_SIZE, STRIDES, MIN_SIZES)
    anchors_xyxy = cxcywh_to_xyxy(anchors)
    gt_boxes = torch.tensor([[50.0, 50.0, 90.0, 90.0], [300.0, 300.0, 350.0, 350.0]])

    labels, _ = match(anchors_xyxy, gt_boxes, pos_iou_threshold=0.5, neg_iou_threshold=0.4)

    assert set(labels.unique().tolist()) <= {-1, 0, 1}


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
