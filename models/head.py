"""Classification + box-regression heads, shared across pyramid levels
(same weights applied to P3/P4/P5) - halves the head parameter count
versus per-level heads and is standard practice (RetinaNet) since the
feature semantics are already unified by the FPN's shared channel width.

Raw conv outputs are left as (B, num_anchors * C, H, W); models/detector.py
owns reshaping them into the (anchor-size-major, then row-major spatial)
order that matches data/anchors.py's anchor layout.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, tower_channels: int, num_convs: int, num_anchors: int) -> None:
        super().__init__()
        self.num_anchors = num_anchors

        def make_tower() -> nn.Sequential:
            layers = []
            ch = in_channels
            for _ in range(num_convs):
                layers += [nn.Conv2d(ch, tower_channels, 3, padding=1), nn.ReLU(inplace=True)]
                ch = tower_channels
            return nn.Sequential(*layers)

        self.cls_tower = make_tower()
        self.reg_tower = make_tower()

        # 1 logit per anchor (face vs. background - single foreground class,
        # so plain sigmoid works and we don't need a softmax slot for "background").
        self.cls_out = nn.Conv2d(tower_channels, num_anchors * 1, kernel_size=3, padding=1)
        self.reg_out = nn.Conv2d(tower_channels, num_anchors * 4, kernel_size=3, padding=1)

        # The output convs get their own small-std init instead of the
        # network-wide Kaiming init (see FaceDetector._init_weights, which
        # deliberately skips these two modules): Kaiming's variance is
        # tuned for hidden ReLU layers, and applied to a final prediction
        # layer it produces logits with std in the tens-to-hundreds, which
        # completely swamps the prior-bias trick below. A small std keeps
        # each anchor's prediction close to the bias at init, which is the
        # point of the trick.
        nn.init.normal_(self.cls_out.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.reg_out.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.reg_out.bias, 0.0)

        # Bias the classification output toward predicting background at
        # init (standard RetinaNet focal-loss trick: prior_prob ~0.01) so
        # the huge negative:positive anchor imbalance doesn't produce a
        # large loss (and unstable early gradients) before training starts.
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_out.bias, bias_value.item())

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        cls_logits = self.cls_out(self.cls_tower(x))  # [B, A*1, H, W]
        bbox_deltas = self.reg_out(self.reg_tower(x))  # [B, A*4, H, W]
        return cls_logits, bbox_deltas


if __name__ == "__main__":
    head = DetectionHead(in_channels=128, tower_channels=128, num_convs=2, num_anchors=2)
    dummy = torch.randn(2, 128, 48, 48)
    cls_logits, bbox_deltas = head(dummy)
    print(f"cls_logits: {tuple(cls_logits.shape)}")
    print(f"bbox_deltas: {tuple(bbox_deltas.shape)}")
    assert cls_logits.shape == (2, 2, 48, 48)
    assert bbox_deltas.shape == (2, 8, 48, 48)

    num_params = sum(p.numel() for p in head.parameters())
    print(f"head params: {num_params:,}")
    print("head.py smoke test passed")
