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

from engine.inference import build_detect_kwargs, detect, load_model
from utils.boxes import blur_faces, draw_detections
from utils.device import get_device
from utils.video import FfmpegNotFoundError, FfmpegVideoWriter, VideoEncodeError

torch.set_num_threads(1)
cv2.setNumThreads(1)

REPO_ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text())
DEVICE = get_device(CFG)
MODEL = load_model(str(REPO_ROOT / "checkpoints" / "best.pth"), CFG, DEVICE)
DETECT_KWARGS = build_detect_kwargs(CFG)

BLUR_CHOICES = ["gaussian", "pixelate", "off (show boxes only)"]


def _gradio_temp_dir() -> str:
    """Same directory Gradio itself uses for uploads/cache (GRADIO_TEMP_DIR,
    defaulting to <system temp>/gradio) - not re-derived from a private
    import, just the documented env var Gradio's own resolution follows."""
    d = Path(os.environ.get("GRADIO_TEMP_DIR") or (Path(tempfile.gettempdir()) / "gradio"))
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _run_detection(frame_bgr: np.ndarray, conf_thresh: float):
    return detect(frame_bgr, MODEL, DEVICE, conf_thresh=conf_thresh, **DETECT_KWARGS)


def process_frame(frame_rgb: np.ndarray | None, blur_choice: str, conf_thresh: float) -> np.ndarray | None:
    """Webcam callback: one RGB frame in, one RGB frame out."""
    if frame_rgb is None:
        return None
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    boxes, scores = _run_detection(frame_bgr, conf_thresh)

    if blur_choice.startswith("off"):
        frame_bgr = draw_detections(frame_bgr, boxes, scores)
    else:
        frame_bgr = blur_faces(frame_bgr, boxes, mode=blur_choice)

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _process_one_video(
    video_path: str, blur_choice: str, conf_thresh: float,
    start_sec: float = 0.0, end_sec: float | None = None,
    progress_cb=None,
) -> str:
    """Process a single video file end-to-end, returning the output path.
    Shared by the single-video and batch tabs. progress_cb(frame_idx,
    total_frames), if given, is called after every frame - callers map that
    to their own progress scale (a fraction of one video for the single-video
    tab, a fraction of the whole batch for the batch tab).

    start_sec/end_sec trim the range processed (end_sec=None means "to the
    end"). Seeking is frame-index-based (CAP_PROP_POS_FRAMES), not
    timestamp-based (CAP_PROP_POS_MSEC) - more reliably accurate across
    container/codec combinations for landing on the exact requested frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise gr.Error(f"Could not open video: {Path(video_path).name}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    start_frame = max(0, round((start_sec or 0.0) * fps))
    end_frame = round(end_sec * fps) if end_sec is not None and end_sec > 0 else video_frame_count
    if video_frame_count is not None and end_frame is not None:
        end_frame = min(end_frame, video_frame_count)
    if end_frame is not None and start_frame >= end_frame:
        cap.release()
        raise gr.Error(f"{Path(video_path).name}: start time must be before end time and within the video's length.")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    total_frames = (end_frame - start_frame) if end_frame is not None else None

    # Written inside Gradio's own cache dir (not the plain OS temp root) so
    # delete_cache (below) tracks and cleans up this exact file. A path
    # outside that directory gets copied into it by Gradio when returned -
    # the copy would get cleaned up, but this original would silently sit
    # on disk untracked and undeleted.
    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", dir=_gradio_temp_dir(), delete=False).name

    try:
        # Encodes H.264/yuv420p directly instead of cv2.VideoWriter's mp4v
        # fourcc, which isn't browser-playable and previously left Gradio
        # doing its own full second re-encode pass after we returned - on a
        # free-tier CPU, that second pass over the video (on top of our own
        # per-frame inference pass) was slow enough to look hung.
        with FfmpegVideoWriter(out_path, width, height, fps) as writer:
            frame_idx = 0
            while True:
                if end_frame is not None and start_frame + frame_idx >= end_frame:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1

                boxes, scores = _run_detection(frame, conf_thresh)
                if blur_choice.startswith("off"):
                    frame = draw_detections(frame, boxes, scores)
                else:
                    frame = blur_faces(frame, boxes, mode=blur_choice)

                writer.write(frame)
                if progress_cb is not None:
                    progress_cb(frame_idx, total_frames)
    except FfmpegNotFoundError as e:
        raise gr.Error(str(e)) from e
    except VideoEncodeError as e:
        raise gr.Error("Video encoding failed.") from e
    finally:
        cap.release()

    return out_path


def process_video(
    video_path: str | None, blur_choice: str, conf_thresh: float,
    start_sec: float, end_sec: float | None, progress=gr.Progress(),
):
    """Upload-a-video callback: processes every frame in [start_sec, end_sec)
    (end_sec=None/0 means to the end), returns a path to the output file
    (Gradio's Video output reads a plain filesystem path). Not real-time -
    the point of this mode is correctness on your own footage, not live
    speed, so a slow CPU host is fine here."""
    if video_path is None:
        return None

    def _cb(frame_idx: int, total_frames: int | None) -> None:
        if total_frames:
            progress(frame_idx / total_frames, desc=f"Processing frame {frame_idx}/{total_frames}")

    return _process_one_video(video_path, blur_choice, conf_thresh, start_sec or 0.0, end_sec, progress_cb=_cb)


def process_video_batch(
    video_paths: list[str] | None, blur_choice: str, conf_thresh: float,
    start_sec: float, end_sec: float | None, progress=gr.Progress(),
):
    """Batch-upload callback: processes each file in sequence, trimmed to
    the same [start_sec, end_sec) range. One bad/corrupt file - or one too
    short for the requested range - doesn't cost the rest of the batch:
    failures are skipped and reported in the status message rather than
    aborting the whole run."""
    if not video_paths:
        return None, "No files uploaded."

    n = len(video_paths)
    outputs: list[str] = []
    failures: list[str] = []

    for i, video_path in enumerate(video_paths):
        name = Path(video_path).name

        def _cb(frame_idx: int, total_frames: int | None, i: int = i, name: str = name) -> None:
            if total_frames:
                frac = (i + frame_idx / total_frames) / n
                progress(frac, desc=f"[{i + 1}/{n}] {name}: frame {frame_idx}/{total_frames}")

        try:
            outputs.append(_process_one_video(
                video_path, blur_choice, conf_thresh, start_sec or 0.0, end_sec, progress_cb=_cb
            ))
        except gr.Error as e:
            failures.append(f"{name}: {e}")

    status = f"Processed {len(outputs)}/{n} file(s) successfully."
    if failures:
        status += "\n\nFailed:\n" + "\n".join(f"- {f}" for f in failures)

    return outputs, status


# Switching away from the Live webcam tab doesn't unmount it - gr.Tab just
# hides content via CSS - so cam_in's streaming .stream() binding keeps
# firing and the browser's webcam capture (and its indicator light) stays
# on indefinitely otherwise. Gradio has no Python-level API to release a
# getUserMedia stream, so this reaches into the DOM directly and stops
# every live video track when the user leaves this tab.
_STOP_CAMERA_JS = """
() => {
    document.querySelectorAll('video').forEach((video) => {
        const stream = video.srcObject;
        if (stream && stream.getTracks) {
            stream.getTracks().forEach((track) => track.stop());
        }
        video.srcObject = null;
    });
}
"""


# delete_cache=(frequency, age): every 10 minutes, delete any temp file
# (upload or generated output) older than 10 minutes. Makes the privacy
# note below actually true instead of just a claim - without this, both
# uploaded and processed files would otherwise sit on disk until the whole
# server process restarts.
with gr.Blocks(title="Face Blur Tool", delete_cache=(600, 600)) as demo:
    gr.Markdown(
        "# Face Blur Tool\n"
        "A from-scratch, custom-trained face detector (no pretrained weights, "
        "no third-party face-detection library) used to automatically blur "
        "faces in your own video, or live from your webcam.\n\n"
        "**Data handling:** uploaded and processed files are kept only as "
        "temporary files on the server and are automatically deleted within "
        "10 minutes. Nothing is written to a database, logged, or shared "
        "with any third party."
    )

    with gr.Tab("Upload a video") as video_tab:
        gr.Markdown("Upload a video - every frame is processed, faces blurred, and the result is downloadable below.")
        with gr.Row():
            video_in = gr.Video(label="Input video", sources=["upload"])
            video_out = gr.Video(label="Output video")
        with gr.Row():
            video_blur = gr.Radio(BLUR_CHOICES, value="gaussian", label="Blur mode")
            video_conf = gr.Slider(0.1, 0.9, value=CFG["inference"]["conf_thresh"], label="Confidence threshold")
        with gr.Row():
            video_start = gr.Number(value=0, minimum=0, label="Start (seconds)")
            video_end = gr.Number(value=None, minimum=0, label="End (seconds, blank = to the end)")
        video_btn = gr.Button("Process video", variant="primary")
        video_btn.click(process_video, [video_in, video_blur, video_conf, video_start, video_end], video_out)

    with gr.Tab("Batch upload") as batch_tab:
        gr.Markdown("Upload multiple videos - each is processed in sequence, faces blurred, with results downloadable below.")
        with gr.Row():
            batch_in = gr.File(label="Input videos", file_count="multiple", file_types=["video"])
            batch_out = gr.File(label="Output videos", file_count="multiple")
        with gr.Row():
            batch_blur = gr.Radio(BLUR_CHOICES, value="gaussian", label="Blur mode")
            batch_conf = gr.Slider(0.1, 0.9, value=CFG["inference"]["conf_thresh"], label="Confidence threshold")
        with gr.Row():
            batch_start = gr.Number(value=0, minimum=0, label="Start (seconds)")
            batch_end = gr.Number(value=None, minimum=0, label="End (seconds, blank = to the end)")
        gr.Markdown("Start/end apply to every file in the batch. A file shorter than the requested range is skipped and reported below, not aborted.")
        batch_status = gr.Markdown()
        batch_btn = gr.Button("Process batch", variant="primary")
        batch_btn.click(
            process_video_batch, [batch_in, batch_blur, batch_conf, batch_start, batch_end], [batch_out, batch_status]
        )

    with gr.Tab("Live webcam"):
        gr.Markdown("Live per-frame detection from your webcam. Free CPU hosting may lag - upload mode is more reliable for a smooth result.")
        with gr.Row():
            cam_blur = gr.Radio(BLUR_CHOICES, value="gaussian", label="Blur mode")
            cam_conf = gr.Slider(0.1, 0.9, value=CFG["inference"]["conf_thresh"], label="Confidence threshold")
        cam_in = gr.Image(sources=["webcam"], streaming=True, label="Webcam")
        cam_out = gr.Image(label="Blurred output")
        cam_in.stream(process_frame, [cam_in, cam_blur, cam_conf], cam_out)

    video_tab.select(fn=lambda: None, outputs=[cam_in], js=_STOP_CAMERA_JS)
    batch_tab.select(fn=lambda: None, outputs=[cam_in], js=_STOP_CAMERA_JS)


if __name__ == "__main__":
    # Render (and most PaaS hosts) assign a port dynamically via $PORT and
    # require binding to 0.0.0.0, not Gradio's local-only default of
    # 127.0.0.1:7860. Falls back to the normal local default when PORT
    # isn't set, so `python app.py` on your own machine is unaffected.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
