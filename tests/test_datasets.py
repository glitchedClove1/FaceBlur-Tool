"""Unit tests for the WIDER FACE annotation parser and Dataset filtering.

Uses a small synthetic annotation file mirroring the real format (verified
against an official fixture) so these tests don't need the actual
multi-gigabyte dataset downloaded.
"""

from pathlib import Path

from data.datasets import WiderFaceDataset, parse_bbx_gt

# name / N / boxes, mirrors the real wider_face_train_bbx_gt.txt layout,
# including the N==0 dummy-line quirk (see "4.jpg" below).
FAKE_ANNOTATIONS = """\
3.jpg
3
46 170 28 31 2 0 0 0 0 0
160 181 17 22 2 0 0 0 0 0
183 167 17 24 2 0 0 0 0 1
4.jpg
0
0 0 0 0 0 0 0 0 0 0
5.jpg
2
10 10 0 20 0 0 0 0 0 0
100 100 5 5 0 0 0 0 0 0
"""


def _write_annotations(tmp_path: Path) -> Path:
    annotation_path = tmp_path / "wider_face_train_bbx_gt.txt"
    annotation_path.write_text(FAKE_ANNOTATIONS)
    return annotation_path


def test_parse_bbx_gt_box_conversion_and_dummy_line(tmp_path):
    annotation_path = _write_annotations(tmp_path)
    samples = parse_bbx_gt(annotation_path, images_root=tmp_path / "images")

    assert len(samples) == 3

    s0 = samples[0]
    assert s0.image_path == tmp_path / "images" / "3.jpg"
    assert s0.boxes.shape == (3, 4)
    # x,y,w,h=46,170,28,31 -> xyxy = 46,170,74,201
    assert list(s0.boxes[0]) == [46.0, 170.0, 74.0, 201.0]
    assert list(s0.invalid) == [False, False, True]

    # N == 0 must consume its dummy line and produce zero boxes, not desync parsing.
    s1 = samples[1]
    assert s1.image_path == tmp_path / "images" / "4.jpg"
    assert s1.boxes.shape == (0, 4)

    # Parsing must recover correctly for the image after the N==0 entry.
    s2 = samples[2]
    assert s2.image_path == tmp_path / "images" / "5.jpg"
    assert s2.boxes.shape == (2, 4)


def test_dataset_filters_invalid_and_degenerate_boxes(tmp_path):
    split_dir = tmp_path / "wider_face_split"
    split_dir.mkdir()
    (split_dir / "wider_face_train_bbx_gt.txt").write_text(FAKE_ANNOTATIONS)
    (tmp_path / "WIDER_train" / "images").mkdir(parents=True)

    ds = WiderFaceDataset(tmp_path, split="train", drop_invalid=True, min_face_size=0.0)

    # image 3.jpg: 3 boxes, 1 flagged invalid -> 2 remain
    assert len(ds.samples[0].boxes) == 2
    # image 4.jpg: 0 boxes
    assert len(ds.samples[1].boxes) == 0
    # image 5.jpg: box[0] has w=0 (degenerate, always dropped), box[1] is valid -> 1 remains
    assert len(ds.samples[2].boxes) == 1


def test_dataset_min_face_size_filter(tmp_path):
    split_dir = tmp_path / "wider_face_split"
    split_dir.mkdir()
    (split_dir / "wider_face_train_bbx_gt.txt").write_text(FAKE_ANNOTATIONS)
    (tmp_path / "WIDER_train" / "images").mkdir(parents=True)

    # 5.jpg's remaining valid box is 5x5 - min_face_size=10 should drop it too.
    ds = WiderFaceDataset(tmp_path, split="train", drop_invalid=True, min_face_size=10.0)
    assert len(ds.samples[2].boxes) == 0


def test_dataset_drop_empty_images(tmp_path):
    split_dir = tmp_path / "wider_face_split"
    split_dir.mkdir()
    (split_dir / "wider_face_train_bbx_gt.txt").write_text(FAKE_ANNOTATIONS)
    (tmp_path / "WIDER_train" / "images").mkdir(parents=True)

    ds = WiderFaceDataset(tmp_path, split="train", drop_invalid=True, drop_empty_images=True)
    # 4.jpg (0 faces) must be dropped entirely.
    paths = [s.image_path.name for s in ds.samples]
    assert "4.jpg" not in paths


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
