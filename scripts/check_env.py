"""Print environment info and recommend training settings for the detected GPU.

Run this first, always: every other script in this repo assumes CUDA is
available and reachable through `torch.cuda`. If it isn't, they still run
(on CPU) but training will be far too slow for real use.
"""

import platform
import sys

import torch

# (min_vram_gb, resolution, batch_size) — ordered smallest VRAM first.
# Resolution/batch are tuned for our single-shot detector (a handful of
# conv-BN-ReLU stages + an FPN-style neck), not a heavy backbone like
# ResNet-50, so even small GPUs can fit a usable batch at 320-416px.
_VRAM_PROFILES = [
    (3.0, 320, 8),
    (5.0, 384, 16),
    (7.0, 416, 24),
    (9.0, 512, 32),
    (11.0, 640, 40),
    (float("inf"), 640, 64),
]


def recommend(vram_gb: float) -> tuple[int, int]:
    for min_gb, resolution, batch_size in _VRAM_PROFILES:
        if vram_gb <= min_gb:
            return resolution, batch_size
    return _VRAM_PROFILES[-1][1], _VRAM_PROFILES[-1][2]


def main() -> None:
    print("=" * 60)
    print("Environment check")
    print("=" * 60)
    print(f"Python:          {platform.python_version()} ({sys.executable})")
    print(f"PyTorch:         {torch.__version__}")
    print(f"Platform:        {platform.platform()}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available:  {cuda_available}")

    if not cuda_available:
        print()
        print("WARNING: CUDA is not available. Training/inference will fall "
              "back to CPU, which will be very slow for this project. "
              "Check your NVIDIA driver and that a CUDA-enabled build of "
              "PyTorch is installed (see requirements.txt).")
        print("=" * 60)
        return

    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    vram_gb = props.total_memory / (1024 ** 3)

    print(f"PyTorch built for CUDA: {torch.version.cuda}")
    print(f"cuDNN version:   {torch.backends.cudnn.version()}")
    print(f"GPU:             {props.name}")
    print(f"VRAM:            {vram_gb:.1f} GB")
    print(f"Compute capability: {props.major}.{props.minor}")

    resolution, batch_size = recommend(vram_gb)
    print()
    print("-" * 60)
    print("Recommended starting config (override in configs/*.yaml):")
    print(f"  input resolution: {resolution}x{resolution}")
    print(f"  batch size:       {batch_size}")
    print("  mixed precision:  enabled (env.amp: true)")
    print("-" * 60)

    if vram_gb < 6.0:
        print(
            "Note: <6GB VRAM detected. Training will use a small input "
            "resolution and batch size, and gradient checkpointing / "
            "smaller backbone widths may be needed later if you hit "
            "out-of-memory errors."
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
