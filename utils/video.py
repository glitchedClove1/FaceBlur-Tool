"""H.264 video encoding via a piped ffmpeg subprocess, shared by the Gradio
app and the detect_video CLI. Frames are written one at a time so encoding
happens interleaved with per-frame inference, rather than as a separate
full re-encode pass afterward - writing cv2.VideoWriter's mp4v fourcc and
letting Gradio (or a browser) re-encode it to a playable codec afterward
means doing the whole video twice, which is what made processing look
hung on Render's free CPU tier.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


class FfmpegNotFoundError(RuntimeError):
    """ffmpeg is not on PATH in this environment."""


class VideoEncodeError(RuntimeError):
    """ffmpeg was invoked but failed or was interrupted mid-stream."""


class FfmpegVideoWriter:
    """Context manager: feed BGR uint8 frames in, get a browser-playable
    H.264 mp4 out. Raises FfmpegNotFoundError immediately (before any frame
    is written) if ffmpeg isn't available. On any failure - a bad frame, or
    ffmpeg crashing/getting OOM-killed mid-stream - __exit__ kills the
    subprocess and deletes the partial output instead of leaving an orphaned
    process or a half-written file behind.
    """

    def __init__(self, out_path: str, width: int, height: int, fps: float):
        if shutil.which("ffmpeg") is None:
            raise FfmpegNotFoundError(
                "ffmpeg is not installed or not on PATH - required to encode browser-playable video."
            )
        self.out_path = out_path
        self._proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-loglevel", "error", out_path,
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame_bgr: np.ndarray) -> None:
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())
        except BrokenPipeError as e:
            raise VideoEncodeError("ffmpeg exited unexpectedly while encoding.") from e

    def __enter__(self) -> "FfmpegVideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self._proc.kill()
            self._proc.wait()
            Path(self.out_path).unlink(missing_ok=True)
            return  # don't suppress the caller's original exception

        self._proc.stdin.close()
        self._proc.wait()
        if self._proc.returncode != 0:
            Path(self.out_path).unlink(missing_ok=True)
            raise VideoEncodeError(f"ffmpeg exited with code {self._proc.returncode}.")
