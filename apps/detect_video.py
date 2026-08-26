"""Run face detection (and optional blurring) on a video file.

Usage:
    .venv\\Scripts\\python.exe -m apps.detect_video --input in.mp4 --output out.mp4
    .venv\\Scripts\\python.exe -m apps.detect_video --input in.mp4 --output out.mp4 --blur gaussian
    .venv\\Scripts\\python.exe -m apps.detect_video --input in.mp4 --output out.mp4 --blur pixelate --show
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
    parser.add_argument("--input", required=True, help="path to the input video file")
    parser.add_argument("--output", required=True, help="path to write the annotated/blurred output video")
    parser.add_argument("--weights", default=str(REPO_ROOT / "checkpoints" / "best.pth"))
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--conf", type=float, default=None, help="overrides configs/default.yaml inference.conf_thresh")
    parser.add_argument("--blur", choices=["off", "gaussian", "pixelate"], default="off")
    parser.add_argument("--show", action="store_true", help="also preview frames in a window while processing")
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

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"Input:  {args.input} ({width}x{height} @ {fps:.1f}fps, {total_frames} frames)")
    print(f"Output: {args.output}")
    print(f"Blur:   {args.blur}")

    frame_idx = 0
    t_start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        boxes, scores = detect(
            frame, model, device, conf_thresh=conf_thresh, nms_thresh=nms_thresh,
            image_size=image_size, mean=norm["mean"], std=norm["std"], variances=variances,
        )

        if args.blur != "off":
            frame = blur_faces(frame, boxes, mode=args.blur)
        else:
            draw_detections(frame, boxes, scores)

        writer.write(frame)
        if args.show:
            cv2.imshow("detect_video", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped early by user (q).")
                break

        if frame_idx % 30 == 0 or frame_idx == total_frames:
            elapsed = time.time() - t_start
            print(f"frame {frame_idx}/{total_frames}  ({frame_idx / max(elapsed, 1e-6):.1f} fps processing)")

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"Done: {frame_idx} frames in {elapsed:.1f}s ({frame_idx / max(elapsed, 1e-6):.1f} fps average)")


if __name__ == "__main__":
    main()
