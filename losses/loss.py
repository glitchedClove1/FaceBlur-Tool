"""Detection loss: sigmoid focal loss for classification + Smooth L1 for
box regression, matched against anchors per-sample in the batch.

Focal loss (not hard-negative mining) is the imbalance strategy here: WIDER
FACE images can carry thousands of background anchors per handful of faces,
and focal loss down-weights confidently-correct negatives via a smooth
per-anchor (1-p)^gamma term instead of discarding most negatives outright
via a fixed neg:pos sampling ratio.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import sigmoid_focal_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.anchors import cxcywh_to_xyxy, encode, match


class DetectionLoss(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        anchors_cfg = cfg["anchors"]
        loss_cfg = cfg["loss"]

        self.pos_iou_threshold = anchors_cfg["pos_iou_threshold"]
        self.neg_iou_threshold = anchors_cfg["neg_iou_threshold"]
        self.variances = tuple(anchors_cfg["variances"])
        self.focal_alpha = loss_cfg["focal_alpha"]
        self.focal_gamma = loss_cfg["focal_gamma"]
        self.bbox_loss_weight = loss_cfg["bbox_loss_weight"]

    def forward(
        self,
        cls_logits: Tensor,
        bbox_deltas: Tensor,
        anchors_cxcywh: Tensor,
        gt_boxes_list: list[Tensor],
    ) -> dict[str, Tensor]:
        """
        cls_logits:  [B, N]     raw logits, one per anchor
        bbox_deltas: [B, N, 4]  raw regression outputs, one per anchor
        anchors_cxcywh: [N, 4]  shared anchor grid (model.anchors)
        gt_boxes_list: length-B list of [Mi, 4] xyxy GT boxes (Mi may be 0)
        """
        device = cls_logits.device
        anchors_xyxy = cxcywh_to_xyxy(anchors_cxcywh)

        batch_labels = []
        batch_reg_targets = []
        for gt_boxes in gt_boxes_list:
            gt_boxes = gt_boxes.to(device)
            labels, matched_idx = match(anchors_xyxy, gt_boxes, self.pos_iou_threshold, self.neg_iou_threshold)

            if gt_boxes.numel() > 0:
                reg_targets = encode(gt_boxes[matched_idx], anchors_cxcywh, self.variances)
            else:
                reg_targets = torch.zeros_like(anchors_cxcywh)

            batch_labels.append(labels)
            batch_reg_targets.append(reg_targets)

        labels = torch.stack(batch_labels)          # [B, N], values in {-1, 0, 1}
        reg_targets = torch.stack(batch_reg_targets)  # [B, N, 4]

        valid_mask = labels != -1   # ignored anchors (between the two IoU thresholds) skip the loss entirely
        pos_mask = labels == 1
        num_pos = pos_mask.sum().clamp(min=1)  # normalizer; clamp avoids div-by-zero on an all-negative batch

        cls_targets = pos_mask.float()
        cls_loss_per_anchor = sigmoid_focal_loss(
            cls_logits, cls_targets, alpha=self.focal_alpha, gamma=self.focal_gamma, reduction="none"
        )
        cls_loss = cls_loss_per_anchor[valid_mask].sum() / num_pos

        if pos_mask.any():
            bbox_loss = F.smooth_l1_loss(bbox_deltas[pos_mask], reg_targets[pos_mask], reduction="sum") / num_pos
        else:
            bbox_loss = bbox_deltas.sum() * 0.0

        total_loss = cls_loss + self.bbox_loss_weight * bbox_loss

        return {
            "loss": total_loss,
            "cls_loss": cls_loss.detach(),
            "bbox_loss": bbox_loss.detach(),
            "num_pos": num_pos.detach(),
        }


if __name__ == "__main__":
    import yaml

    from data.anchors import generate_anchors

    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "default.yaml").read_text())
    anchors_cfg = cfg["anchors"]

    anchors = generate_anchors(cfg["data"]["image_size"], anchors_cfg["strides"], anchors_cfg["min_sizes"])
    num_anchors = anchors.shape[0]
    batch_size = 2

    criterion = DetectionLoss(cfg)
    cls_logits = torch.randn(batch_size, num_anchors, requires_grad=True)
    bbox_deltas = torch.randn(batch_size, num_anchors, 4, requires_grad=True)
    gt_boxes_list = [
        torch.tensor([[100.0, 100.0, 140.0, 140.0]]),
        torch.zeros((0, 4)),  # background-only image
    ]

    out = criterion(cls_logits, bbox_deltas, anchors, gt_boxes_list)
    print({k: v.item() for k, v in out.items()})
    out["loss"].backward()
    assert cls_logits.grad is not None and bbox_deltas.grad is not None
    print("loss.py smoke test passed")
