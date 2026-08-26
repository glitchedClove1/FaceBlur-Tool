"""Shared single-frame detection used by the evaluator and both apps
(video file, webcam): preprocess (letterbox + normalize) -> forward ->
decode -> NMS -> rescale boxes back to the original frame's coordinates.

Preprocessing is done directly with cv2/numpy rather than through the
albumentations pipeline used for training/eval-loss - mathematically
equivalent to data/transforms.py's build_eval_transform (LongestMaxSize +
PadIfNeeded, centered), but without Compose's per-call overhead, which
matters for webcam real-time use.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.boxes import postprocess


def letterbox(image_bgr: np.ndarray, size: int, pad_value: int = 114) -> tuple[np.ndarray, float, int, int]:
    """Resize (aspect-preserving) so the long side == size, then center-pad
    to size x size. Returns (padded_image, scale, pad_left, pad_top) - the
    three values needed to map predictions back to original coordinates.
    """
    h, w = image_bgr.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w, pad_h = size - new_w, size - new_h
    pad_left, pad_top = pad_w // 2, pad_h // 2
    pad_right, pad_bottom = pad_w - pad_left, pad_h - pad_top

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value),
    )
    return padded, scale, pad_left, pad_top


def preprocess(image_bgr: np.ndarray, size: int, mean: list[float], std: list[float]) -> tuple[Tensor, float, int, int]:
    padded, scale, pad_left, pad_top = letterbox(image_bgr, size)
    img = padded.astype(np.float32) / 255.0
    img = (img - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1, 3, size, size]
    return tensor, scale, pad_left, pad_top


def rescale_boxes(boxes_xyxy: Tensor, scale: float, pad_left: int, pad_top: int) -> Tensor:
    """Inverse of the letterbox transform: model-space boxes -> original-frame pixel coordinates."""
    boxes = boxes_xyxy.clone()
    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top
    boxes /= scale
    return boxes


@torch.no_grad()
def detect(
    image_bgr: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    conf_thresh: float,
    nms_thresh: float,
    image_size: int,
    mean: list[float],
    std: list[float],
    variances: tuple[float, float],
    max_detections: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Single BGR frame -> (boxes[K,4] xyxy in original-frame pixels, scores[K]), both numpy."""
    h, w = image_bgr.shape[:2]
    tensor, scale, pad_left, pad_top = preprocess(image_bgr, image_size, mean, std)
    tensor = tensor.to(device)

    out = model(tensor)
    boxes, scores = postprocess(
        out["cls_logits"][0], out["bbox_deltas"][0], model.anchors, variances,
        image_width=image_size, image_height=image_size,
        conf_thresh=conf_thresh, nms_thresh=nms_thresh, max_detections=max_detections,
    )
    if boxes.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    boxes = rescale_boxes(boxes, scale, pad_left, pad_top)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
    return boxes.cpu().numpy(), scores.cpu().numpy()


def load_model(checkpoint_path: str, cfg: dict, device: torch.device):
    """Build a FaceDetector from cfg and load trained weights - shared by
    evaluate.py and both apps so there's one way to go from a .pth file to
    a ready-to-use model."""
    from models.detector import FaceDetector

    model = FaceDetector(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


if __name__ == "__main__":
    import yaml

    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "default.yaml").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(str(Path(__file__).resolve().parent.parent / "checkpoints" / "best.pth"), cfg, device)

    # Synthetic frame at an arbitrary non-square resolution, to prove the
    # letterbox round-trip lands predictions back in the right coordinate space.
    dummy = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    norm = cfg["augmentation"]["normalize"]
    boxes, scores = detect(
        dummy, model, device,
        conf_thresh=0.5, nms_thresh=cfg["inference"]["nms_thresh"],
        image_size=cfg["data"]["image_size"], mean=norm["mean"], std=norm["std"],
        variances=tuple(cfg["anchors"]["variances"]),
    )
    print(f"input frame: {dummy.shape}, detections: {boxes.shape[0]}")
    if boxes.shape[0] > 0:
        assert (boxes[:, 0] >= 0).all() and (boxes[:, 2] <= 1280).all()
        assert (boxes[:, 1] >= 0).all() and (boxes[:, 3] <= 720).all()
    print("inference.py smoke test passed")
