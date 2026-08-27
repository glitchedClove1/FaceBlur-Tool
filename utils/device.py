"""Device selection shared by every entry point (training, evaluation, both
CLI apps, and the Gradio demo) - one place to change if device-selection
semantics ever need a new branch (e.g. mps)."""

from __future__ import annotations

import torch

from utils.log_setup import get_logger

logger = get_logger()


def get_device(cfg: dict) -> torch.device:
    requested = cfg["env"]["device"]
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("env.device is 'cuda' but CUDA is not available - falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)
