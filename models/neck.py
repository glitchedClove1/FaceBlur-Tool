"""Small top-down feature pyramid: unifies C3/C4/C5 to a common channel
width and fuses coarse (semantic) features down into finer (high-resolution)
ones, so the small-face-detecting P3 level still gets deep-layer context."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FPNNeck(nn.Module):
    def __init__(self, in_channels: dict[str, int], fpn_channels: int = 128) -> None:
        super().__init__()
        self.lateral_c3 = nn.Conv2d(in_channels["C3"], fpn_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(in_channels["C4"], fpn_channels, kernel_size=1)
        self.lateral_c5 = nn.Conv2d(in_channels["C5"], fpn_channels, kernel_size=1)

        # Post-fusion smoothing conv, one per output level - removes the
        # aliasing that plain nearest-neighbor upsample-and-add introduces.
        self.smooth_p3 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth_p4 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth_p5 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)

        self.out_channels = fpn_channels

    def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        c3, c4, c5 = features["C3"], features["C4"], features["C5"]

        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")

        return {
            "P3": self.smooth_p3(p3),
            "P4": self.smooth_p4(p4),
            "P5": self.smooth_p5(p5),
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models.backbone import Backbone

    backbone = Backbone()
    neck = FPNNeck(backbone.out_channels, fpn_channels=128)

    dummy = torch.randn(2, 3, 384, 384)
    features = backbone(dummy)
    pyramid = neck(features)

    for name, feat in pyramid.items():
        print(f"{name}: {tuple(feat.shape)}")
    assert pyramid["P3"].shape == (2, 128, 48, 48)
    assert pyramid["P4"].shape == (2, 128, 24, 24)
    assert pyramid["P5"].shape == (2, 128, 12, 12)

    num_params = sum(p.numel() for p in neck.parameters())
    print(f"neck params: {num_params:,}")
    print("neck.py smoke test passed")
