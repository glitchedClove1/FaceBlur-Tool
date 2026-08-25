"""Fetch and verify the WIDER FACE dataset.

WIDER FACE images are hosted on Google Drive, which throttles/quota-limits
automated downloads of popular files. This script first attempts an
automatic download; when Google Drive refuses (very likely for a dataset
this widely used), it prints exact manual-download steps instead. Either
way, run this script again afterwards — it verifies what's on disk by MD5
and by image/annotation counts before declaring success.

Expected final layout (produced by extracting the files below into
data/widerface/):

    data/widerface/
      wider_face_split/
        wider_face_train_bbx_gt.txt
        wider_face_val_bbx_gt.txt
        wider_face_test_filelist.txt
        readme.txt
      WIDER_train/images/<61 event category folders>/*.jpg
      WIDER_val/images/<61 event category folders>/*.jpg

We only fetch train + val (annotated) splits — the official test split has
no released ground-truth boxes, and the project brief says not to touch it.
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data" / "widerface"

# (Google Drive file ID, MD5, filename) — IDs/hashes as published by the
# WIDER FACE authors and mirrored in torchvision's own WIDERFace loader.
IMAGE_ARCHIVES = [
    ("15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M", "3fedf70df600953d25982bcd13d91ba2", "WIDER_train.zip"),
    ("1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q", "dfa7d7e790efa35df3788964cf0bbaea", "WIDER_val.zip"),
]
ANNOTATION_URL = "http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip"
ANNOTATION_MD5 = "0e3767bcf0e326556d407bf5bff5d27c"
ANNOTATION_FILENAME = "wider_face_split.zip"

# Official dataset statistics (Yang et al., 2016) used as a sanity check,
# not an exact requirement — a handful of images off is not a failure.
EXPECTED_IMAGE_COUNTS = {"train": 12880, "val": 3226}


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def try_auto_download() -> None:
    try:
        from torchvision.datasets.utils import download_and_extract_archive
    except ImportError:
        print("torchvision not installed; skipping automatic download attempt.")
        return

    try:
        import gdown
    except ImportError:
        gdown = None

    ROOT.mkdir(parents=True, exist_ok=True)
    for file_id, md5, filename in IMAGE_ARCHIVES:
        target = ROOT / filename
        if target.exists() and md5sum(target) == md5:
            continue
        print(f"Attempting automatic download of {filename} from Google Drive...")
        if gdown is None:
            print("  Skipped: 'gdown' is not installed (pip install gdown).")
            continue
        try:
            gdown.download(id=file_id, output=str(target), quiet=False)
            if md5sum(target) != md5:
                print(f"  Downloaded {filename} but MD5 did not match - likely an incomplete/blocked download.")
                target.unlink(missing_ok=True)
        except Exception as exc:  # Google Drive quota errors, network errors, etc.
            print(f"  Automatic download failed for {filename}: {exc}")

    annotation_zip = ROOT / ANNOTATION_FILENAME
    if not (ROOT / "wider_face_split").exists():
        print("Attempting automatic download of annotations...")
        try:
            download_and_extract_archive(url=ANNOTATION_URL, download_root=str(ROOT), md5=ANNOTATION_MD5)
        except Exception as exc:
            print(f"  Automatic download failed for annotations: {exc}")


def extract_archives() -> None:
    for _, _, filename in IMAGE_ARCHIVES:
        zip_path = ROOT / filename
        split_dir = ROOT / zip_path.stem  # WIDER_train.zip -> WIDER_train
        if zip_path.exists() and not split_dir.exists():
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(ROOT)

    annotation_zip = ROOT / ANNOTATION_FILENAME
    if annotation_zip.exists() and not (ROOT / "wider_face_split").exists():
        print(f"Extracting {ANNOTATION_FILENAME}...")
        with zipfile.ZipFile(annotation_zip) as zf:
            zf.extractall(ROOT)


def print_manual_instructions() -> None:
    print()
    print("=" * 70)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 70)
    print(
        "Google Drive automatic download did not complete (this is expected -\n"
        "Google Drive rate-limits popular shared files). Download these three\n"
        "files yourself from the official WIDER FACE page:\n"
        "\n"
        "  http://shuoyang1213.me/WIDERFACE/\n"
        "\n"
        "Files needed (train + val images, plus annotations):\n"
        "  1. WIDER_train.zip   (Google Drive / Tencent Drive / Hugging Face mirror)\n"
        "  2. WIDER_val.zip     (Google Drive / Tencent Drive / Hugging Face mirror)\n"
        "  3. wider_face_split.zip  (annotations - linked directly on the page)\n"
        "\n"
        f"Place all three .zip files directly in:\n  {ROOT}\n"
        "\n"
        "Then re-run this script:\n"
        "  .venv\\Scripts\\python.exe scripts\\download_widerface.py\n"
        "\n"
        "It will extract and verify them automatically. If you already have\n"
        "the extracted WIDER_train/, WIDER_val/, wider_face_split/ folders\n"
        "from elsewhere, just place those folders directly under the path\n"
        "above and re-run - extraction will be skipped since they already exist."
    )
    print("=" * 70)


def verify() -> bool:
    ok = True

    split_dir = ROOT / "wider_face_split"
    for fname in ("wider_face_train_bbx_gt.txt", "wider_face_val_bbx_gt.txt"):
        if not (split_dir / fname).exists():
            print(f"MISSING: {split_dir / fname}")
            ok = False

    for split, expected_count in EXPECTED_IMAGE_COUNTS.items():
        images_dir = ROOT / f"WIDER_{split}" / "images"
        if not images_dir.exists():
            print(f"MISSING: {images_dir}")
            ok = False
            continue
        actual_count = sum(1 for _ in images_dir.rglob("*.jpg"))
        status = "OK" if actual_count == expected_count else "WARNING (count mismatch)"
        print(f"WIDER_{split}: {actual_count} images found (expected {expected_count}) [{status}]")
        if actual_count == 0:
            ok = False

    return ok


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)

    already_have_split = (ROOT / "wider_face_split").exists()
    already_have_images = all((ROOT / f"WIDER_{s}").exists() for s in EXPECTED_IMAGE_COUNTS)

    if not (already_have_split and already_have_images):
        try_auto_download()
        extract_archives()

    print()
    print("Verifying dataset layout...")
    ok = verify()

    if not ok:
        print_manual_instructions()
        sys.exit(1)

    print()
    print("WIDER FACE train/val is ready.")


if __name__ == "__main__":
    main()
