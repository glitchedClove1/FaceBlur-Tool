"""WIDER FACE Easy/Medium/Hard mAP@0.5 evaluation - the standard protocol
used across virtually every published WIDER FACE result, so numbers here
are directly comparable to other work.

Ground truth for the three difficulty settings comes from the official
eval toolkit's .mat files (data/widerface/eval_tools/) - these encode,
per image, which ground-truth faces are considered "in scope" for each
difficulty level (Hard keeps almost all faces including tiny/occluded
ones, Easy keeps only large clear ones). A prediction that matches a
face outside the current setting's scope is neither a true positive nor
a false positive - it's excluded from that setting's tally entirely,
which is why Hard AP is always lower than Easy AP for the exact same
model: it's judged against a stricter, larger set of required faces.

Algorithm (re-implemented from the official protocol's logic, verified
against the widely-used reference at
https://github.com/wondervictor/WiderFace-Evaluation, using its data
files but our own code):
  1. Run detect() once per validation image (shared across all 3 settings).
  2. Globally min-max normalize all detection scores across the whole
     dataset into [0, 1] (matches the official protocol - keeps the 1000
     confidence bins below meaningful regardless of this model's raw
     score distribution).
  3. Per image per setting: match predictions to "in-scope" GT boxes by
     IoU >= 0.5 (greedy, highest-IoU-first via argmax - ties are not
     re-optimized, same as the reference); accumulate a running
     (proposals, true-positives) count at 1000 confidence thresholds.
  4. Reduce to a precision/recall curve, then integrate it into AP via
     the standard PASCAL VOC precision-envelope method.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from scipy.io import loadmat
from torchvision.ops import box_iou
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.inference import build_detect_kwargs, detect, load_model
from utils.device import get_device

REPO_ROOT = Path(__file__).resolve().parent.parent
SETTINGS = ["easy", "medium", "hard"]
THRESH_NUM = 1000
IOU_THRESH = 0.5

# Official WIDER FACE eval-toolkit ground truth, mirrored (the small .mat
# files, not the dataset itself) at a well-known community repo since the
# original toolkit isn't a simple direct-download URL. Not committed to
# this repo (binary, and belongs with the rest of data/widerface/ which is
# gitignored) - fetched on demand instead, same reproducibility approach
# as scripts/download_widerface.py uses for the dataset itself.
_EVAL_TOOLS_BASE_URL = (
    "https://raw.githubusercontent.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/"
    "master/widerface_evaluate/ground_truth"
)
_EVAL_TOOLS_FILES = ["wider_face_val.mat", "wider_easy_val.mat", "wider_medium_val.mat", "wider_hard_val.mat"]


def ensure_eval_tools(gt_dir: Path) -> None:
    gt_dir.mkdir(parents=True, exist_ok=True)
    missing = [f for f in _EVAL_TOOLS_FILES if not (gt_dir / f).exists()]
    if not missing:
        return

    import urllib.request

    print(f"Fetching {len(missing)} official WIDER FACE eval ground-truth file(s)...")
    for filename in missing:
        url = f"{_EVAL_TOOLS_BASE_URL}/{filename}"
        dest = gt_dir / filename
        print(f"  {url}")
        urllib.request.urlretrieve(url, dest)


def load_ground_truth(gt_dir: Path):
    ensure_eval_tools(gt_dir)
    gt_mat = loadmat(gt_dir / "wider_face_val.mat")
    face_bbx_list = gt_mat["face_bbx_list"]
    event_list = gt_mat["event_list"]
    file_list = gt_mat["file_list"]
    keep_lists = {s: loadmat(gt_dir / f"wider_{s}_val.mat")["gt_list"] for s in SETTINGS}
    return face_bbx_list, event_list, file_list, keep_lists


def run_all_detections(model, device, cfg, images_root: Path, event_list, file_list, conf_thresh: float, max_detections: int):
    """detect() once per image, cached by (event_idx, image_idx). Returns
    the cache plus the global min/max score for normalization."""
    detect_kwargs = build_detect_kwargs(cfg)

    cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    all_scores: list[float] = []

    num_events = len(event_list)
    for ei in tqdm(range(num_events), desc="Running detection"):
        event_name = str(event_list[ei][0][0])
        img_names = file_list[ei][0]
        for ji in range(len(img_names)):
            img_name = str(img_names[ji][0][0])
            img_path = images_root / event_name / f"{img_name}.jpg"
            image = cv2.imread(str(img_path))
            if image is None:
                raise FileNotFoundError(f"Could not read {img_path}")

            boxes, scores = detect(
                image, model, device, conf_thresh=conf_thresh,
                max_detections=max_detections, **detect_kwargs,
            )
            order = np.argsort(-scores)
            boxes, scores = boxes[order], scores[order]
            cache[(ei, ji)] = (boxes, scores)
            all_scores.extend(scores.tolist())

    score_min = min(all_scores) if all_scores else 0.0
    score_max = max(all_scores) if all_scores else 1.0
    return cache, score_min, score_max


def image_eval(
    pred_boxes: np.ndarray, pred_scores_norm: np.ndarray, gt_boxes: np.ndarray, ignore: np.ndarray,
    overlaps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One image, one difficulty setting. `ignore[i] == 1` means GT box i
    counts toward this setting (kept despite the confusing official-mat
    naming); 0 means it's real but out of scope for this setting.

    `overlaps` (pred x gt IoU matrix) is the same for every setting of a
    given image - only `ignore` changes - so evaluate() computes it once per
    image and passes it in here to avoid recomputing it per setting; it's
    computed locally when omitted (e.g. by callers/tests scoring one setting
    in isolation).

    Returns (pred_recall, proposal_list): pred_recall[h] is the cumulative
    number of in-scope GT boxes found by predictions [0..h]; proposal_list[h]
    is 1 if prediction h counts as a normal proposal, -1 if it matched an
    out-of-scope GT box and is excluded from this setting's tally entirely.
    """
    n_pred, n_gt = pred_boxes.shape[0], gt_boxes.shape[0]
    recall_list = np.zeros(n_gt)
    proposal_list = np.ones(n_pred)
    pred_recall = np.zeros(n_pred)

    if n_pred == 0 or n_gt == 0:
        return pred_recall, proposal_list

    if overlaps is None:
        overlaps = box_iou(torch.from_numpy(pred_boxes).float(), torch.from_numpy(gt_boxes).float()).numpy()

    for h in range(n_pred):
        max_overlap = overlaps[h].max()
        max_idx = overlaps[h].argmax()
        if max_overlap >= IOU_THRESH:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1
        pred_recall[h] = (recall_list == 1).sum()

    return pred_recall, proposal_list


def accumulate_pr(pred_scores_norm: np.ndarray, proposal_list: np.ndarray, pred_recall: np.ndarray) -> np.ndarray:
    """Bins one image's predictions into THRESH_NUM confidence thresholds,
    recording (num proposals kept, recall achieved) at each - summed
    across images by the caller to build the dataset-wide PR curve."""
    pr_info = np.zeros((THRESH_NUM, 2))
    for t in range(THRESH_NUM):
        thresh = 1 - (t + 1) / THRESH_NUM
        keep = np.where(pred_scores_norm >= thresh)[0]
        if len(keep) == 0:
            continue
        last = keep[-1]
        kept_proposals = np.where(proposal_list[: last + 1] == 1)[0]
        pr_info[t, 0] = len(kept_proposals)
        pr_info[t, 1] = pred_recall[last]
    return pr_info


def voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(cfg: dict, checkpoint_path: str, conf_thresh: float = 0.01, max_detections: int = 750) -> dict[str, float]:
    device = get_device(cfg)
    model = load_model(checkpoint_path, cfg, device)

    gt_dir = REPO_ROOT / cfg["data"]["root"] / "eval_tools"
    images_root = REPO_ROOT / cfg["data"]["root"] / "WIDER_val" / "images"
    face_bbx_list, event_list, file_list, keep_lists = load_ground_truth(gt_dir)

    cache, score_min, score_max = run_all_detections(
        model, device, cfg, images_root, event_list, file_list, conf_thresh, max_detections
    )
    score_range = max(score_max - score_min, 1e-12)

    pr_curves = {s: np.zeros((THRESH_NUM, 2)) for s in SETTINGS}
    count_faces = {s: 0 for s in SETTINGS}

    num_events = len(event_list)
    for ei in tqdm(range(num_events), desc="Scoring"):
        img_names = file_list[ei][0]
        gt_boxes_all = face_bbx_list[ei][0]

        for ji in range(len(img_names)):
            gt_boxes_xywh = gt_boxes_all[ji][0].astype(np.float64)
            pred_boxes, pred_scores = cache[(ei, ji)]

            gt_boxes_xyxy = gt_boxes_xywh.copy()
            gt_boxes_xyxy[:, 2] += gt_boxes_xyxy[:, 0]
            gt_boxes_xyxy[:, 3] += gt_boxes_xyxy[:, 1]
            pred_scores_norm = (pred_scores - score_min) / score_range

            # The pred/gt boxes - and therefore their IoU overlaps - are the
            # same for every setting of this image; only which GT boxes are
            # "in scope" (via `ignore` below) differs. Computed once here
            # instead of 3x (once per Easy/Medium/Hard call to image_eval).
            overlaps = None
            if pred_boxes.shape[0] > 0 and gt_boxes_xyxy.shape[0] > 0:
                overlaps = box_iou(
                    torch.from_numpy(pred_boxes).float(), torch.from_numpy(gt_boxes_xyxy).float()
                ).numpy()

            for setting in SETTINGS:
                keep_index = keep_lists[setting][ei][0][ji][0].reshape(-1)
                count_faces[setting] += len(keep_index)

                if gt_boxes_xywh.shape[0] == 0 or pred_boxes.shape[0] == 0:
                    continue

                ignore = np.zeros(gt_boxes_xyxy.shape[0])
                if len(keep_index) > 0:
                    ignore[keep_index - 1] = 1  # official indices are 1-based

                pred_recall, proposal_list = image_eval(
                    pred_boxes, pred_scores_norm, gt_boxes_xyxy, ignore, overlaps=overlaps
                )
                pr_curves[setting] += accumulate_pr(pred_scores_norm, proposal_list, pred_recall)

    results = {}
    for setting in SETTINGS:
        pr_curve = pr_curves[setting]
        precision = np.divide(pr_curve[:, 1], pr_curve[:, 0], out=np.zeros(THRESH_NUM), where=pr_curve[:, 0] > 0)
        recall = pr_curve[:, 1] / max(count_faces[setting], 1)
        ap = voc_ap(recall, precision)
        results[setting] = ap
        print(f"{setting:>6} AP: {ap:.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "checkpoints" / "best.pth"))
    parser.add_argument("--conf-thresh", type=float, default=0.01)
    parser.add_argument("--max-detections", type=int, default=750)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    print(f"Evaluating {args.checkpoint} on WIDER FACE val (Easy/Medium/Hard)...")
    results = evaluate(cfg, args.checkpoint, args.conf_thresh, args.max_detections)
    print()
    print("=" * 40)
    for setting in SETTINGS:
        print(f"{setting.capitalize():>8} AP: {results[setting]:.4f}")
    print("=" * 40)
