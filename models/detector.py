"""Assembles backbone + FPN neck + shared head into the full single-shot
face detector, and owns the anchor grid + output reshaping that ties the
raw per-level conv outputs to data/anchors.py's flat anchor ordering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.anchors import generate_anchors
from models.backbone import Backbone
from models.head import DetectionHead
from models.neck import FPNNeck

_LEVEL_NAMES = ["P3", "P4", "P5"]


class FaceDetector(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        anchors_cfg = cfg["anchors"]
        image_size = cfg["data"]["image_size"]

        min_sizes = anchors_cfg["min_sizes"]
        anchors_per_level = {len(sizes) for sizes in min_sizes}
        if len(anchors_per_level) != 1:
            raise ValueError(
                f"All pyramid levels must have the same anchor count for the shared head, got min_sizes={min_sizes}"
            )
        num_anchors = anchors_per_level.pop()

        self.backbone = Backbone(
            stem_channels=model_cfg["backbone"]["stem_channels"],
            stage_channels=model_cfg["backbone"]["stage_channels"],
        )
        self.neck = FPNNeck(self.backbone.out_channels, fpn_channels=model_cfg["neck"]["fpn_channels"])
        self.head = DetectionHead(
            in_channels=self.neck.out_channels,
            tower_channels=model_cfg["head"]["channels"],
            num_convs=model_cfg["head"]["num_convs"],
            num_anchors=num_anchors,
        )

        anchors_cxcywh = generate_anchors(image_size, anchors_cfg["strides"], min_sizes)
        self.register_buffer("anchors", anchors_cxcywh, persistent=False)
        self.num_anchors_per_loc = num_anchors

        self._init_weights()

    def _init_weights(self) -> None:
        # Kaiming/He init (fan_out, relu) for every conv weight - the right
        # default for a ReLU network trained from random init. Biases are
        # left alone: BatchNorm already defaults to (weight=1, bias=0), and
        # head.cls_out's bias was deliberately set to the focal-loss prior
        # trick in DetectionHead.__init__ - overwriting it here would undo that.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def _reshape_level(self, cls_logits: Tensor, bbox_deltas: Tensor) -> tuple[Tensor, Tensor]:
        """[B, A*C, H, W] -> [B, A*H*W, C], ordered (anchor-size-major, then
        row-major spatial) to match data.anchors.generate_anchors()."""
        b, _, h, w = cls_logits.shape
        a = self.num_anchors_per_loc

        cls_logits = cls_logits.view(b, a, 1, h, w).permute(0, 1, 3, 4, 2).reshape(b, a * h * w, 1)
        bbox_deltas = bbox_deltas.view(b, a, 4, h, w).permute(0, 1, 3, 4, 2).reshape(b, a * h * w, 4)
        return cls_logits, bbox_deltas

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        features = self.backbone(images)
        pyramid = self.neck(features)

        cls_list, reg_list = [], []
        for level in _LEVEL_NAMES:
            cls_logits, bbox_deltas = self.head(pyramid[level])
            cls_logits, bbox_deltas = self._reshape_level(cls_logits, bbox_deltas)
            cls_list.append(cls_logits)
            reg_list.append(bbox_deltas)

        return {
            "cls_logits": torch.cat(cls_list, dim=1).squeeze(-1),  # [B, total_anchors]
            "bbox_deltas": torch.cat(reg_list, dim=1),             # [B, total_anchors, 4]
        }


if __name__ == "__main__":
    import yaml

    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "default.yaml").read_text())
    model = FaceDetector(cfg)

    image_size = cfg["data"]["image_size"]
    dummy = torch.randn(2, 3, image_size, image_size)
    out = model(dummy)

    print(f"input: {tuple(dummy.shape)}")
    print(f"cls_logits: {tuple(out['cls_logits'].shape)}")
    print(f"bbox_deltas: {tuple(out['bbox_deltas'].shape)}")
    print(f"anchors: {tuple(model.anchors.shape)}")

    total_anchors = model.anchors.shape[0]
    assert out["cls_logits"].shape == (2, total_anchors)
    assert out["bbox_deltas"].shape == (2, total_anchors, 4)

    print()
    print("Parameter count by module:")
    total_params = 0
    for name, module in [("backbone", model.backbone), ("neck", model.neck), ("head", model.head)]:
        n = sum(p.numel() for p in module.parameters())
        total_params += n
        print(f"  {name:10s}: {n:,}")
    print(f"  {'total':10s}: {total_params:,}")

    print()
    print("detector.py smoke test passed")
