# Face Blur Tool

A face detector built **from scratch** — custom CNN backbone, FPN neck,
anchor-based detection head, and training loop, all written and trained in
this repo. No pretrained weights anywhere (not even ImageNet), no
`face_recognition` / MTCNN / RetinaFace / MediaPipe. Trained on
[WIDER FACE](http://shuoyang1213.me/WIDERFACE/) on a single consumer GPU
(RTX 3050 Laptop, 4GB VRAM).

It ships as: a CLI to blur faces in a video file, a CLI for live webcam
blurring, and a Gradio web app (deployable for free on Hugging Face
Spaces) that does both through a browser.

## Why "from scratch" matters here

The backbone, neck, head, anchors, loss, and training loop are all
implemented in this repo (`models/`, `data/anchors.py`, `losses/loss.py`,
`engine/train.py`). Generic PyTorch/torchvision ops (NMS, IoU, focal loss)
are used since they're not face-specific - but the network and its weights
originate entirely from this project's own training runs, starting from
random initialization.

## Results

**WIDER FACE val, official Easy/Medium/Hard protocol** (`engine/evaluate.py`,
same ground-truth/algorithm used across published WIDER FACE results):

| Checkpoint | Easy AP | Medium AP | Hard AP |
|---|---|---|---|
| 60 epochs  | 0.642 | 0.583 | 0.355 |
| 90 epochs  | 0.656 | 0.592 | 0.365 |

*(Hard is always lower than Easy for any model - it's judged against a
stricter, larger set of required faces, including tiny/occluded ones.)*

For context: published results using much larger backbones (ResNet-50+)
with far longer training reach the 0.90s. These numbers are what a lean
3.89M-parameter, from-scratch model reaches in ~5h45m of training on a
4GB laptop GPU - a real, working detector, not a toy that only detects on
its own training batch.

The 60→90 epoch jump is a small but real lesson worth knowing if you
extend training further yourself: the gain looks modest (+0.01-0.02 AP)
because a cosine LR schedule decays to ~0 by design at whatever epoch
count you configure - so simply resuming past that point trains at
LR≈0 and does nothing. Getting real benefit from more epochs requires
bumping `training.epochs` *before* resuming, so the schedule recalculates
and gives the model real room to keep improving (see `--resume` below).

**Training sanity gate** (`engine/train.py --overfit-one-batch`): loss
19.56 → 0.007 over 400 steps on a fixed 8-image batch, predicted boxes
visually lock onto every ground-truth face at 0.94-0.99 confidence. This
is what proves the full pipeline (data → anchors → model → loss → decode
→ draw) is wired correctly before spending hours on a real run.

## Project structure

```
configs/default.yaml     single source of truth for every hyperparameter
data/
  datasets.py             WIDER FACE annotation parser + torch Dataset
  transforms.py            augmentation pipeline (train/eval)
  anchors.py                anchor generation, encode/decode, IoU matching
models/
  backbone.py               custom residual CNN (stem + 4 stages)
  neck.py                    FPN (top-down feature fusion)
  head.py                    shared cls/reg detection head
  detector.py                assembles the above + owns the anchor grid
losses/loss.py            sigmoid focal loss + Smooth L1
engine/
  train.py                   training loop, checkpointing, --overfit-one-batch
  evaluate.py                 WIDER FACE Easy/Medium/Hard mAP@0.5
  inference.py                 shared detect(): preprocess -> forward -> decode -> NMS -> rescale
apps/
  detect_video.py            CLI: blur/annotate a video file
  detect_webcam.py            CLI: blur/annotate a live webcam feed
app.py                    Gradio web UI (video upload + live webcam) - HF Spaces deployment target
utils/
  boxes.py                    NMS/decode postprocessing + gaussian/pixelate blur
  log_setup.py                 console logging helpers
scripts/
  check_env.py                 CUDA/GPU/VRAM check + batch-size recommendation
  download_widerface.py         dataset download + verification
  visualize_dataset.py           dumps annotated training images for a sanity check
  import_my_footage.py           ingest your own labeled footage into the Dataset interface
tests/                    pytest unit tests (anchors, dataset parsing, loss, evaluation)
```

## Setup

Requires **Python 3.12** specifically (PyTorch has no Windows wheels for
newer Pythons at time of writing). [`uv`](https://github.com/astral-sh/uv)
is recommended - much faster than pip and resolves the CUDA wheel index
correctly.

```
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
.venv\Scripts\python.exe scripts\check_env.py
```

`check_env.py` prints your GPU/VRAM and a recommended starting batch size
and resolution - both already set in `configs/default.yaml` for an RTX
3050 (4GB); override there if you have different hardware.

If you only want to run the apps/demo (not retrain), skip straight to
[Run on video or webcam](#run-on-a-video-file-or-webcam) - you'll need a
trained checkpoint (see [Download the trained weights](#download-the-trained-weights)
or train your own).

## Reproducing training from scratch

1. **Download WIDER FACE**:
   ```
   .venv\Scripts\python.exe scripts\download_widerface.py
   ```
   Attempts an automatic download; Google Drive rate-limits large shared
   files, so it will likely print manual-download instructions instead -
   follow them, then re-run the script to verify (it checks image counts
   against the official totals: 12,880 train / 3,226 val).

2. **Sanity-check parsing + augmentation**:
   ```
   .venv\Scripts\python.exe scripts\visualize_dataset.py
   ```
   Writes ~20 annotated training images to `outputs/dataset_viz/` - eyeball
   them before spending any GPU time.

3. **Run the unit tests**:
   ```
   .venv\Scripts\python.exe -m pytest tests/ -v
   ```

4. **Overfit-one-batch gate** (mandatory before a real run):
   ```
   .venv\Scripts\python.exe -m engine.train --overfit-one-batch
   ```
   Loss should collapse toward 0 and `outputs/overfit_check/*.jpg` should
   show predicted boxes locking onto every face. If this doesn't work,
   nothing downstream will either - fix it here first.

5. **Train**:
   ```
   .venv\Scripts\python.exe -m engine.train
   ```
   Reads every hyperparameter from `configs/default.yaml` (epochs, LR
   schedule, batch size, etc.). Checkpoints to `checkpoints/last.pth`
   (every epoch) and `checkpoints/best.pth` (on validation improvement).
   Refuses to silently overwrite an existing run - if `checkpoints/last.pth`
   already exists, pass `--resume checkpoints/last.pth` to continue it, or
   move/delete the checkpoint dir to start fresh.

   TensorBoard: `tensorboard --logdir runs`

   **A note on `training.num_workers`**: it's set to `0`, not a typo.
   On Windows with limited RAM, multi-process DataLoader workers can crash
   (`DataLoader worker exited unexpectedly`) because each worker re-imports
   the entire torch/opencv/albumentations stack from scratch (no
   copy-on-write like Linux). If you have more RAM or are on Linux, raising
   this is worth retrying - it roughly halves epoch time when it works.

6. **Evaluate** (official WIDER FACE Easy/Medium/Hard protocol):
   ```
   .venv\Scripts\python.exe -m engine.evaluate --checkpoint checkpoints/best.pth
   ```
   Downloads the official ground-truth files on first run (small, cached
   after that). Takes a few minutes on GPU.

## Download the trained weights

Trained checkpoints aren't committed to this repo (binary, ~47MB, and
reproducible from the steps above). If you just want to run the apps
without training:
- Train your own with the steps above, **or**
- Grab a released checkpoint from this repo's GitHub Releases (if
  published) and place it at `checkpoints/best.pth`.

## Run on a video file or webcam

```
.venv\Scripts\python.exe -m apps.detect_video --input in.mp4 --output out.mp4 --blur gaussian
.venv\Scripts\python.exe -m apps.detect_webcam --blur pixelate
```

Both share `engine/inference.py`'s `detect()` (letterbox preprocess →
forward → decode → NMS → rescale to your original frame's coordinates) and
`utils/boxes.py`'s blur helpers, so behavior is identical between them.

**`--blur`**: `off` (draws boxes + confidence, no blurring), `gaussian`
(kernel size scales with face size, so small and large faces are equally
obscured), or `pixelate` (mosaic effect). Boxes are padded ~15% before
blurring so hairline/chin/ears are covered too, and clipped safely at
frame edges.

**`--conf`** overrides `configs/default.yaml`'s `inference.conf_thresh`
(default 0.5) - lower it to catch more faces at the cost of more false
positives.

## Deploying the web demo (Hugging Face Spaces)

`app.py` is a Gradio UI wrapping the same detection/blur code as the CLI
apps - upload-a-video and live-webcam tabs.

1. Create a new [Hugging Face Space](https://huggingface.co/new-space),
   SDK: **Gradio**, hardware: free CPU tier is enough (the model is only
   3.89M parameters).
2. Push this repo's contents to the Space (or connect it to sync from
   GitHub).
3. Upload `checkpoints/best.pth` directly to the Space (Hugging Face
   handles large files natively - no need to fight GitHub's size limits).
4. In the Space settings, make sure it installs from
   **`requirements-deploy.txt`**, not `requirements.txt` - the deploy file
   is a trimmed, CPU-only set (no albumentations/tensorboard/pytest/gdown/
   scipy, which are training/eval-only and would slow the build for
   nothing).
5. The Space runs `app.py` automatically once weights + deps are in place.

Live webcam mode may lag on a free CPU host under heavy load - the
upload-a-video tab is the more reliable feature for a smooth result, and
was always the primary use case for a free deployment.

## Bring your own footage

**Just testing** (no labels needed): point `apps/detect_video.py` or
`apps/detect_webcam.py` at any `.mp4`/webcam - nothing else required.

**Training or fine-tuning on your own footage**: label a few hundred
frames (a few thousand is better) with face bounding boxes.

- **Recommended tool**: [CVAT](https://www.cvat.ai/) (free, browser-based,
  good bounding-box workflow) or [Label Studio](https://labelstud.io/).
  [labelImg](https://github.com/HumanSignal/labelImg) also works for a
  smaller job.
- **Format**: export as one `.txt` per image, one box per line:
  `x1 y1 x2 y2` (pixel coordinates, top-left/bottom-right corners).
- **Layout**:
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
  An image with zero faces still needs an empty (0-byte) `.txt` file, not
  a missing one.
- **Ingest it**:
  ```
  .venv\Scripts\python.exe scripts\import_my_footage.py --source my_footage --split train
  ```
  Converts your layout into the same `WiderFaceDataset`-compatible
  structure used for WIDER FACE, so it can be mixed into training without
  any other code changes.

How many labeled frames are "enough" depends on how different your
footage is from WIDER FACE (indoor office footage vs. outdoor crowds, for
example) - a few hundred frames is enough to notice a difference via
`engine/evaluate.py`; a few thousand for a meaningful shift in accuracy on
your specific footage.

## Hardware notes

Trained and tested on an RTX 3050 Laptop GPU (4GB VRAM) and a 7.4GB-RAM
Windows machine - both genuinely constrained by consumer-hardware
standards. `scripts/check_env.py` measures your actual hardware and
recommends settings; `configs/default.yaml`'s comments explain every
hardware-driven decision (batch size, `num_workers`, image resolution)
with the reasoning and measurements behind it, not just the final number.
