"""Gradio demo: upload a video (or use your webcam) and get faces blurred.

This is the deployment entry point for Hugging Face Spaces - a thin UI
wrapper around the same engine/inference.py detect() and utils/boxes.py
blur_faces() used by apps/detect_video.py and apps/detect_webcam.py. No
detection logic lives here; this file only wires those into a web UI.

Local run:
    .venv\\Scripts\\python.exe app.py
"""

from __future__ import annotations

import os

# Must be set before torch/numpy/cv2 are imported - their native thread
# pools are sized at import/init time. On a free host (Render's 0.1 CPU
# instance, for example) extra threads buy no real parallelism and just
# add memory overhead, which matters when the whole budget is 512MB.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import yaml

from engine.inference import detect, load_model
from utils.boxes import blur_faces

torch.set_num_threads(1)
cv2.setNumThreads(1)

REPO_ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = load_model(str(REPO_ROOT / "checkpoints" / "best.pth"), CFG, DEVICE)

_NORM = CFG["augmentation"]["normalize"]
_VARIANCES = tuple(CFG["anchors"]["variances"])
_IMAGE_SIZE = CFG["data"]["image_size"]
_NMS_THRESH = CFG["inference"]["nms_thresh"]

BLUR_CHOICES = ["gaussian", "pixelate", "off (show boxes only)"]


def _run_detection(frame_bgr: np.ndarray, conf_thresh: float):
    return detect(
        frame_bgr, MODEL, DEVICE, conf_thresh=conf_thresh, nms_thresh=_NMS_THRESH,
        image_size=_IMAGE_SIZE, mean=_NORM["mean"], std=_NORM["std"], variances=_VARIANCES,
    )


def _draw_boxes(frame_bgr: np.ndarray, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    out = frame_bgr.copy()
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(out, f"{score:.2f}", (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
    return out


def process_frame(frame_rgb: np.ndarray | None, blur_choice: str, conf_thresh: float) -> np.ndarray | None:
    """Webcam callback: one RGB frame in, one RGB frame out."""
    if frame_rgb is None:
        return None
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    boxes, scores = _run_detection(frame_bgr, conf_thresh)

    if blur_choice.startswith("off"):
        frame_bgr = _draw_boxes(frame_bgr, boxes, scores)
    else:
        frame_bgr = blur_faces(frame_bgr, boxes, mode=blur_choice)

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def process_video(video_path: str | None, blur_choice: str, conf_thresh: float, progress=gr.Progress()):
    """Upload-a-video callback: processes every frame, returns a path to
    the output file (Gradio's Video output reads a plain filesystem path).
    Not real-time - the point of this mode is correctness on your own
    footage, not live speed, so a slow CPU host is fine here."""
    if video_path is None:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        boxes, scores = _run_detection(frame, conf_thresh)
        if blur_choice.startswith("off"):
            frame = _draw_boxes(frame, boxes, scores)
        else:
            frame = blur_faces(frame, boxes, mode=blur_choice)

        writer.write(frame)
        if total_frames:
            progress(frame_idx / total_frames, desc=f"Processing frame {frame_idx}/{total_frames}")

    cap.release()
    writer.release()
    return out_path


with gr.Blocks(title="Face Blur Tool") as demo:
    gr.Markdown(
        "# Face Blur Tool\n"
        "A from-scratch, custom-trained face detector (no pretrained weights, "
        "no third-party face-detection library) used to automatically blur "
        "faces in your own video, or live from your webcam."
    )

    with gr.Tab("Upload a video"):
        gr.Markdown("Upload a video - every frame is processed, faces blurred, and the result is downloadable below.")
        with gr.Row():
            video_in = gr.Video(label="Input video", sources=["upload"])
            video_out = gr.Video(label="Output video")
        with gr.Row():
            video_blur = gr.Radio(BLUR_CHOICES, value="gaussian", label="Blur mode")
            video_conf = gr.Slider(0.1, 0.9, value=CFG["inference"]["conf_thresh"], label="Confidence threshold")
        video_btn = gr.Button("Process video", variant="primary")
        video_btn.click(process_video, [video_in, video_blur, video_conf], video_out)

    with gr.Tab("Live webcam"):
        gr.Markdown("Live per-frame detection from your webcam. Free CPU hosting may lag - upload mode is more reliable for a smooth result.")
        with gr.Row():
            cam_blur = gr.Radio(BLUR_CHOICES, value="gaussian", label="Blur mode")
            cam_conf = gr.Slider(0.1, 0.9, value=CFG["inference"]["conf_thresh"], label="Confidence threshold")
        cam_in = gr.Image(sources=["webcam"], streaming=True, label="Webcam")
        cam_out = gr.Image(label="Blurred output")
        cam_in.stream(process_frame, [cam_in, cam_blur, cam_conf], cam_out)


if __name__ == "__main__":
    # Render (and most PaaS hosts) assign a port dynamically via $PORT and
    # require binding to 0.0.0.0, not Gradio's local-only default of
    # 127.0.0.1:7860. Falls back to the normal local default when PORT
    # isn't set, so `python app.py` on your own machine is unaffected.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
