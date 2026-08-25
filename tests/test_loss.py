"""Loss unit tests: shape/gradient sanity, and the required proof that the
loss decreases when predictions are optimized toward known matches."""

import torch

from data.anchors import generate_anchors
from losses.loss import DetectionLoss

CFG = {
    "anchors": {
        "strides": [8, 16, 32],
        "min_sizes": [[16, 32], [64, 128], [256, 512]],
        "variances": [0.1, 0.2],
        "pos_iou_threshold": 0.5,
        "neg_iou_threshold": 0.4,
    },
    "loss": {"focal_alpha": 0.25, "focal_gamma": 2.0, "bbox_loss_weight": 1.0},
}
IMAGE_SIZE = 384


def test_loss_produces_finite_scalar_with_gradients():
    anchors = generate_anchors(IMAGE_SIZE, CFG["anchors"]["strides"], CFG["anchors"]["min_sizes"])
    criterion = DetectionLoss(CFG)

    cls_logits = torch.randn(2, anchors.shape[0], requires_grad=True)
    bbox_deltas = torch.randn(2, anchors.shape[0], 4, requires_grad=True)
    gt_boxes_list = [torch.tensor([[100.0, 100.0, 140.0, 140.0]]), torch.zeros((0, 4))]

    out = criterion(cls_logits, bbox_deltas, anchors, gt_boxes_list)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert cls_logits.grad is not None
    assert bbox_deltas.grad is not None
    assert torch.isfinite(cls_logits.grad).all()
    assert torch.isfinite(bbox_deltas.grad).all()


def test_loss_decreases_on_synthetic_data():
    """Directly optimize raw logits/deltas (no network) toward one known GT
    box and confirm the loss trends down - proves data -> anchors -> loss ->
    gradients is wired correctly end to end."""
    torch.manual_seed(0)
    anchors = generate_anchors(IMAGE_SIZE, CFG["anchors"]["strides"], CFG["anchors"]["min_sizes"])
    criterion = DetectionLoss(CFG)

    cls_logits = torch.zeros(1, anchors.shape[0], requires_grad=True)
    bbox_deltas = torch.zeros(1, anchors.shape[0], 4, requires_grad=True)
    gt_boxes_list = [torch.tensor([[150.0, 150.0, 190.0, 190.0]])]

    optimizer = torch.optim.SGD([cls_logits, bbox_deltas], lr=0.5)

    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        out = criterion(cls_logits, bbox_deltas, anchors, gt_boxes_list)
        out["loss"].backward()
        optimizer.step()
        losses.append(out["loss"].item())

    # Losses aren't required to be perfectly monotonic step-to-step (SGD
    # noise), but the trend must be clearly downward.
    assert losses[-1] < losses[0] * 0.5, f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert losses[-1] < min(losses[:5])


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
