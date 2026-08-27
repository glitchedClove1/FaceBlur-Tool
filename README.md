# Face Blur Tool

A face detection model built entirely from scratch — custom CNN backbone, feature pyramid neck, anchor-based detection head, and training pipeline, all implemented in this repository. No pretrained weights (not ImageNet, not a face-specific model), no third-party face detection library. Trained on [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) on a single consumer GPU (RTX 3050 Laptop, 4GB VRAM).

The project ships as three interfaces: a CLI for processing video files, a CLI for live webcam use, and a Gradio web application supporting both through a browser.

## Overview

- **Backbone / neck / head**: custom residual CNN, FPN-style feature fusion, shared classification/regression head
- **Training**: focal loss + Smooth L1, AdamW, cosine LR schedule, AMP, from-scratch weight initialization
- **Evaluation**: official WIDER FACE Easy/Medium/Hard mAP@0.5 protocol
- **Inference**: video file processing, live webcam, and a Gradio web UI, each with Gaussian or pixelate face blurring

Standard PyTorch/torchvision operators (NMS, IoU, focal loss) are used where they are not face-specific, but the network architecture and trained weights originate entirely from this project.

## Results

WIDER FACE validation set, official Easy/Medium/Hard protocol (`engine/evaluate.py`, using the same ground-truth data and algorithm as published WIDER FACE results):

| Checkpoint | Easy AP | Medium AP | Hard AP |
|---|---|---|---|
| 60 epochs | 0.642 | 0.583 | 0.355 |
| 90 epochs | 0.656 | 0.592 | 0.365 |

Hard AP is lower than Easy AP for any model on this benchmark, since it is evaluated against a stricter, larger set of required faces, including small and occluded ones. For reference, published results using larger backbones (ResNet-50 and above) with substantially longer training reach the 0.90s; the figures above reflect a 3.89M-parameter model trained from random initialization in roughly 5h45m on a 4GB laptop GPU.

**Note on extending training beyond a completed run**: the cosine LR schedule decays to zero at the configured epoch count. Resuming a run without first increasing `training.epochs` in the config will train at a near-zero learning rate and produce no improvement. Increase `training.epochs` before resuming to give the schedule room to continue.

**Overfit sanity check** (`engine/train.py --overfit-one-batch`): loss decreases from 19.56 to 0.007 over 400 steps on a fixed 8-image batch, with predicted boxes converging on every ground-truth face at 0.94–0.99 confidence. This validates the full pipeline — data loading, anchor matching, model, loss, decoding, and visualization — before committing to a full training run.

## Project structure

```
configs/default.yaml     single source of truth for every hyperparameter
data/
  datasets.py             WIDER FACE annotation parser and torch Dataset
  transforms.py           augmentation pipeline (train/eval)
  anchors.py              anchor generation, encode/decode, IoU matching
models/
  backbone.py             custom residual CNN (stem + 4 stages)
  neck.py                 FPN (top-down feature fusion)
  head.py                 shared classification/regression detection head
  detector.py             assembles the above, owns the anchor grid
losses/loss.py            sigmoid focal loss + Smooth L1
engine/
  train.py                training loop, checkpointing, --overfit-one-batch
  evaluate.py              WIDER FACE Easy/Medium/Hard mAP@0.5
  inference.py              shared detect(): preprocess, forward, decode, NMS, rescale
apps/
  detect_video.py          CLI: process a video file
  detect_webcam.py         CLI: process a live webcam feed
app.py                    Gradio web UI (video upload + live webcam)
utils/
  boxes.py                 NMS/decode postprocessing, Gaussian/pixelate blur
  log_setup.py              console logging helpers
scripts/
  check_env.py              CUDA/GPU/VRAM check and batch-size recommendation
  download_widerface.py     dataset download and verification
  visualize_dataset.py      writes annotated training images for inspection
  import_my_footage.py      ingest custom labeled footage into the Dataset interface
tests/                    pytest unit tests (anchors, dataset parsing, loss, evaluation)
```

## Setup

Requires **Python 3.12** (PyTorch does not yet provide Windows wheels for newer Python versions). [`uv`](https://github.com/astral-sh/uv) is recommended for environment setup — faster than pip and resolves the CUDA wheel index correctly.

```
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
.venv\Scripts\python.exe scripts\check_env.py
```

`check_env.py` reports detected GPU/VRAM and a recommended batch size and resolution. Defaults in `configs/default.yaml` are tuned for an RTX 3050 (4GB); adjust them for different hardware.

To run the applications without training a model, see [Download the trained weights](#download-the-trained-weights).

## Reproducing training from scratch

1. **Download WIDER FACE**
   ```
   .venv\Scripts\python.exe scripts\download_widerface.py
   ```
   Attempts an automatic download. Google Drive rate-limits large shared files, so manual-download instructions are printed if this fails — follow them, then re-run the script to verify (it checks image counts against the official totals: 12,880 train / 3,226 val).

2. **Verify parsing and augmentation**
   ```
   .venv\Scripts\python.exe scripts\visualize_dataset.py
   ```
   Writes ~20 annotated training images to `outputs/dataset_viz/` for visual inspection before training.

3. **Run the test suite**
   ```
   .venv\Scripts\python.exe -m pytest tests/ -v
   ```

4. **Overfit-one-batch check** (required before a full training run)
   ```
   .venv\Scripts\python.exe -m engine.train --overfit-one-batch
   ```
   Loss should collapse toward zero, and `outputs/overfit_check/*.jpg` should show predicted boxes converging on every face.

5. **Train**
   ```
   .venv\Scripts\python.exe -m engine.train
   ```
   Hyperparameters are read from `configs/default.yaml`. Checkpoints are written to `checkpoints/last.pth` (every epoch) and `checkpoints/best.pth` (on validation improvement). An existing `checkpoints/last.pth` is not overwritten silently — pass `--resume checkpoints/last.pth` to continue that run, or clear the checkpoint directory to start over.

   TensorBoard: `tensorboard --logdir runs`

   `training.num_workers` is set to `0` by default. On Windows with limited system RAM, multi-process DataLoader workers can crash (`DataLoader worker exited unexpectedly`), since each worker reimports the full torch/opencv/albumentations stack rather than sharing memory via copy-on-write as on Linux. Increasing this value is worth testing on systems with more RAM or on Linux; it can roughly halve epoch time when stable.

6. **Evaluate** (official WIDER FACE Easy/Medium/Hard protocol)
   ```
   .venv\Scripts\python.exe -m engine.evaluate --checkpoint checkpoints/best.pth
   ```
   Downloads the official ground-truth files on first run.

## Download the trained weights

`checkpoints/best.pth` is committed to this repository so the applications and deployment targets can run without retraining.

## Running on a video file or webcam

```
.venv\Scripts\python.exe -m apps.detect_video --input in.mp4 --output out.mp4 --blur gaussian
.venv\Scripts\python.exe -m apps.detect_webcam --blur pixelate
```

Both commands share `engine/inference.py`'s `detect()` function (letterbox preprocessing, forward pass, decoding, NMS, and rescaling to the original frame's coordinates) and `utils/boxes.py`'s blur functions.

`--blur` accepts `off` (draws boxes and confidence scores only), `gaussian` (blur kernel size scales with face size), or `pixelate` (mosaic effect). Boxes are padded before blurring to cover hairline and edges, and clipped safely at frame boundaries.

`--conf` overrides `configs/default.yaml`'s `inference.conf_thresh` (default 0.5).

## Deploying the web application

`app.py` is a Gradio interface wrapping the same detection and blurring code as the CLI applications, with tabs for single-video upload, batch upload (multiple files processed in sequence, one failure doesn't abort the rest), and live webcam use. The video tabs support trimming to a start/end range, so only the requested portion of each file is processed - useful on its own, and it also cuts processing time on a CPU-limited free-tier deployment.

### Render

1. Create a [Render](https://render.com) account (no payment method required for the free tier).
2. Create a new **Web Service** and connect this repository.
3. Configure:
   - Runtime: **Python 3**
   - Build command: `pip install -r requirements-deploy.txt`
   - Start command: `python app.py`
   - Instance type: **Free**
4. Deploy. `app.py` binds to `$PORT` and `0.0.0.0` automatically when running on Render.

The free instance provides 512MB RAM and sleeps after 15 minutes of inactivity, with a brief cold-start delay on the next request.

### Hugging Face Spaces

1. Create a new [Space](https://huggingface.co/new-space) with the Gradio SDK.
2. Push this repository's contents to the Space, or connect it to sync from GitHub.
3. Configure the Space to install from `requirements-deploy.txt` — a trimmed, CPU-only dependency set that excludes training/evaluation-only packages (albumentations, tensorboard, pytest, gdown, scipy).

Note: Hugging Face has restricted CPU Basic Space creation for some new free accounts, defaulting them to ZeroGPU instead. ZeroGPU requires an account older than 30 days with a verified email, and provides a 5-minute daily GPU quota.

Live webcam mode may be less responsive on free CPU hosting under load; the video upload path is the more reliable option for a free deployment.

### Data handling

Uploaded and processed files are kept only as temporary files on the server; nothing is written to a database, logged, or shared with any third party. `app.py` runs with Gradio's `delete_cache=(600, 600)`, which deletes any temp file (upload or generated output) older than 10 minutes — see the comment above the `gr.Blocks(...)` call for how this is wired so it actually covers files this app generates, not just what Gradio's own upload path creates by default.

## Bring your own footage

**Testing only** (no labels required): point `apps/detect_video.py` or `apps/detect_webcam.py` at any video file or webcam.

**Training or fine-tuning on custom footage**: label a set of frames with face bounding boxes.

- **Labeling tools**: [CVAT](https://www.cvat.ai/) or [Label Studio](https://labelstud.io/) for browser-based labeling; [labelImg](https://github.com/HumanSignal/labelImg) for smaller datasets.
- **Label format**: one `.txt` file per image, one box per line: `x1 y1 x2 y2` (pixel coordinates).
- **Directory layout**:
  ```
  my_footage/
    images/
      frame_0001.jpg
      frame_0002.jpg
      ...
    labels/
      frame_0001.txt
      frame_0002.txt
      ...
  ```
  Images with no faces require an empty label file, not a missing one.
- **Ingest**:
  ```
  .venv\Scripts\python.exe scripts\import_my_footage.py --source my_footage --split train
  ```
  Converts the layout above into the same structure `WiderFaceDataset` uses for WIDER FACE.

The number of labeled frames needed depends on how much the target footage differs from WIDER FACE. A few hundred frames are typically sufficient to produce a measurable change via `engine/evaluate.py`; a few thousand for a more substantial shift in accuracy on the target domain.

## Hardware notes

Developed and trained on an RTX 3050 Laptop GPU (4GB VRAM) and a 7.4GB-RAM Windows machine. `scripts/check_env.py` measures available hardware and recommends settings; `configs/default.yaml` documents the reasoning behind each hardware-driven default (batch size, `num_workers`, image resolution).

## License

MIT — see [LICENSE](LICENSE).
