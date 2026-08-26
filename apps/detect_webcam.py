"""Run face detection (and optional blurring) live on a webcam feed.

Usage:
    .venv\\Scripts\\python.exe -m apps.detect_webcam
    .venv\\Scripts\\python.exe -m apps.detect_webcam --blur pixelate
    .venv\\Scripts\\python.exe -m apps.detect_webcam --camera 1 --blur gaussian

Press 'q' in the preview window to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.inference import detect, load_model
from utils.boxes import blur_faces

REPO_ROOT = Path(__file__).resolve().parent.parent


def draw_detections(frame, boxes, scores) -> None:
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(frame, f"{score:.2f}", (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--weights", default=str(REPO_ROOT / "checkpoints" / "best.pth"))
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--conf", type=float, default=None, help="overrides configs/default.yaml inference.conf_thresh")
    parser.add_argument("--blur", choices=["off", "gaussian", "pixelate"], default="off")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(cfg["env"]["device"] if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and cfg["env"]["device"] == "cuda":
        print("WARNING: CUDA not available, running on CPU (will be slower).")

    model = load_model(args.weights, cfg, device)

    norm = cfg["augmentation"]["normalize"]
    variances = tuple(cfg["anchors"]["variances"])
    image_size = cfg["data"]["image_size"]
    conf_thresh = args.conf if args.conf is not None else cfg["inference"]["conf_thresh"]
    nms_thresh = cfg["inference"]["nms_thresh"]

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    print(f"Camera {args.camera} opened. Blur: {args.blur}. Press 'q' to quit.")

    fps_smoothed = 0.0
    prev_t = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed, stopping.")
                break

            boxes, scores = detect(
                frame, model, device, conf_thresh=conf_thresh, nms_thresh=nms_thresh,
                image_size=image_size, mean=norm["mean"], std=norm["std"], variances=variances,
            )

            if args.blur != "off":
                frame = blur_faces(frame, boxes, mode=args.blur)
            else:
                draw_detections(frame, boxes, scores)

            now = time.time()
            instant_fps = 1.0 / max(now - prev_t, 1e-6)
            fps_smoothed = instant_fps if fps_smoothed == 0.0 else 0.9 * fps_smoothed + 0.1 * instant_fps
            prev_t = now
            cv2.putText(frame, f"{fps_smoothed:.1f} fps", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

            cv2.imshow("detect_webcam - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
