"""Run face detection (and optional blurring) on a video file.

Usage:
    .venv\\Scripts\\python.exe -m apps.detect_video --input in.mp4 --output out.mp4
    .venv\\Scripts\\python.exe -m apps.detect_video --input in.mp4 --output out.mp4 --blur gaussian
    .venv\\Scripts\\python.exe -m apps.detect_video --input in.mp4 --output out.mp4 --blur pixelate --show
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.inference import build_detect_kwargs, detect, load_model
from utils.boxes import blur_faces, draw_detections
from utils.device import get_device
from utils.video import FfmpegVideoWriter, VideoEncodeError

REPO_ROOT = Path(__file__).resolve().parent.parent


class _Mp4vFallbackWriter:
    """Used when ffmpeg isn't on PATH. Plays fine in desktop players (VLC,
    Windows Media Player) but the mp4v codec isn't decodable by most web
    browsers - install ffmpeg for browser-playable output."""

    def __init__(self, out_path: str, width: int, height: int, fps: float):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    def write(self, frame) -> None:
        self._writer.write(frame)

    def __enter__(self) -> "_Mp4vFallbackWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._writer.release()


def _open_writer(out_path: str, width: int, height: int, fps: float):
    if shutil.which("ffmpeg") is None:
        print(
            "WARNING: ffmpeg not found on PATH - writing with OpenCV's mp4v codec "
            "instead. Output will play in desktop players but not directly in most "
            "web browsers. Install ffmpeg for browser-playable output."
        )
        return _Mp4vFallbackWriter(out_path, width, height, fps)
    return FfmpegVideoWriter(out_path, width, height, fps)


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
    device = get_device(cfg)
    model = load_model(args.weights, cfg, device)
    detect_kwargs = build_detect_kwargs(cfg)
    conf_thresh = args.conf if args.conf is not None else cfg["inference"]["conf_thresh"]

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Could not open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {args.input} ({width}x{height} @ {fps:.1f}fps, {total_frames} frames)")
    print(f"Output: {args.output}")
    print(f"Blur:   {args.blur}")

    frame_idx = 0
    t_start = time.time()
    try:
        with _open_writer(args.output, width, height, fps) as writer:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1

                boxes, scores = detect(frame, model, device, conf_thresh=conf_thresh, **detect_kwargs)

                if args.blur != "off":
                    frame = blur_faces(frame, boxes, mode=args.blur)
                else:
                    frame = draw_detections(frame, boxes, scores)

                writer.write(frame)
                if args.show:
                    cv2.imshow("detect_video", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Stopped early by user (q).")
                        break

                if frame_idx % 30 == 0 or frame_idx == total_frames:
                    elapsed = time.time() - t_start
                    print(f"frame {frame_idx}/{total_frames}  ({frame_idx / max(elapsed, 1e-6):.1f} fps processing)")
    except VideoEncodeError as e:
        raise SystemExit(f"error: video encoding failed ({e})") from e
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"Done: {frame_idx} frames in {elapsed:.1f}s ({frame_idx / max(elapsed, 1e-6):.1f} fps average)")


if __name__ == "__main__":
    main()
