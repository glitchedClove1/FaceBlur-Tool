"""From-scratch CNN backbone: stem + 4 residual stages, exposing C3/C4/C5
(strides 8/16/32) for the FPN neck. No pretrained weights anywhere - every
parameter is randomly initialized (models/detector.py owns the init call).

Kept narrow (default stage widths [32, 64, 128, 256]) since this trains on
a 4GB GPU with no ImageNet head start to lean on for feature quality.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Two 3x3 convs with a skip connection; a 1x1 projection shortcut
    handles the cases where stride or channel count changes."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x) if self.shortcut is not None else x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class Stage(nn.Module):
    """One downsampling residual block followed by `depth - 1` same-resolution ones."""

    def __init__(self, in_ch: int, out_ch: int, depth: int = 2) -> None:
        super().__init__()
        blocks = [ResidualBlock(in_ch, out_ch, stride=2)]
        blocks += [ResidualBlock(out_ch, out_ch, stride=1) for _ in range(depth - 1)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class Backbone(nn.Module):
    """Input (B,3,H,W) -> {"C3": stride 8, "C4": stride 16, "C5": stride 32}."""

    def __init__(self, stem_channels: int = 16, stage_channels: list[int] | None = None, stage_depth: int = 2) -> None:
        super().__init__()
        stage_channels = stage_channels or [32, 64, 128, 256]
        if len(stage_channels) != 4:
            raise ValueError(f"stage_channels must have 4 entries (stride4/8/16/32), got {stage_channels}")

        self.stem = ConvBNAct(3, stem_channels, kernel_size=3, stride=2)  # stride 2
        self.stage1 = Stage(stem_channels, stage_channels[0], depth=stage_depth)  # stride 4 (internal only)
        self.stage2 = Stage(stage_channels[0], stage_channels[1], depth=stage_depth)  # stride 8 -> C3
        self.stage3 = Stage(stage_channels[1], stage_channels[2], depth=stage_depth)  # stride 16 -> C4
        self.stage4 = Stage(stage_channels[2], stage_channels[3], depth=stage_depth)  # stride 32 -> C5

        self.out_channels = {"C3": stage_channels[1], "C4": stage_channels[2], "C5": stage_channels[3]}

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return {"C3": c3, "C4": c4, "C5": c5}


if __name__ == "__main__":
    backbone = Backbone()
    dummy = torch.randn(2, 3, 384, 384)
    out = backbone(dummy)
    for name, feat in out.items():
        print(f"{name}: {tuple(feat.shape)}")
    assert out["C3"].shape == (2, 64, 48, 48)
    assert out["C4"].shape == (2, 128, 24, 24)
    assert out["C5"].shape == (2, 256, 12, 12)
    num_params = sum(p.numel() for p in backbone.parameters())
    print(f"backbone params: {num_params:,}")
    print("backbone.py smoke test passed")
