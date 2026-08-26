"""Unit tests for the WIDER FACE Easy/Medium/Hard evaluation primitives."""

import numpy as np

from engine.evaluate import accumulate_pr, image_eval, voc_ap


def test_voc_ap_perfect_detector():
    # Precision 1.0 at every recall level -> AP == 1.0
    recall = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    precision = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    assert abs(voc_ap(recall, precision) - 1.0) < 1e-9


def test_voc_ap_zero_detector():
    recall = np.zeros(5)
    precision = np.zeros(5)
    assert voc_ap(recall, precision) == 0.0


def test_image_eval_true_positive_matches_in_scope_gt():
    pred_boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
    gt_boxes = np.array([[12.0, 12.0, 48.0, 48.0]])  # high IoU with the prediction
    ignore = np.array([1.0])  # in scope for this difficulty setting

    pred_recall, proposal_list = image_eval(pred_boxes, np.array([0.9]), gt_boxes, ignore)

    assert proposal_list[0] == 1  # counted as a normal proposal
    assert pred_recall[0] == 1  # the one GT box was found


def test_image_eval_out_of_scope_match_is_excluded_not_penalized():
    pred_boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
    gt_boxes = np.array([[12.0, 12.0, 48.0, 48.0]])
    ignore = np.array([0.0])  # out of scope for this difficulty setting (e.g. a tiny face on the Easy setting)

    pred_recall, proposal_list = image_eval(pred_boxes, np.array([0.9]), gt_boxes, ignore)

    # Matching an out-of-scope GT box removes the prediction from the tally
    # entirely - not a true positive, but also not penalized as a false positive.
    assert proposal_list[0] == -1
    assert pred_recall[0] == 0


def test_image_eval_no_match_counts_as_proposal_without_recall():
    pred_boxes = np.array([[500.0, 500.0, 550.0, 550.0]])  # far from any GT
    gt_boxes = np.array([[12.0, 12.0, 48.0, 48.0]])
    ignore = np.array([1.0])

    pred_recall, proposal_list = image_eval(pred_boxes, np.array([0.9]), gt_boxes, ignore)

    assert proposal_list[0] == 1  # still a normal proposal (an unmatched false positive)
    assert pred_recall[0] == 0  # but recalled nothing


def test_image_eval_handles_empty_predictions_and_gt():
    empty = np.zeros((0, 4))
    pred_recall, proposal_list = image_eval(empty, np.zeros(0), np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([1.0]))
    assert pred_recall.shape == (0,)
    assert proposal_list.shape == (0,)


def test_accumulate_pr_high_confidence_bin_reflects_top_prediction_only():
    # Two predictions: one high-confidence true positive, one low-confidence false positive.
    scores_norm = np.array([0.95, 0.05])
    proposal_list = np.array([1.0, 1.0])
    pred_recall = np.array([1.0, 1.0])  # cumulative: after pred 0, 1 GT found; still 1 after pred 1

    pr_info = accumulate_pr(scores_norm, proposal_list, pred_recall)

    # bin t has thresh = 1 - (t+1)/1000, so t=49 -> thresh=0.950: only the
    # 0.95-scored prediction clears it.
    assert pr_info[49, 0] == 1  # 1 proposal kept
    assert pr_info[49, 1] == 1  # 1 true positive recalled
    # The very strictest bin (t=0, thresh=0.999) clears neither prediction.
    assert pr_info[0, 0] == 0
    # At a very low threshold (the last bin), both predictions are visible.
    assert pr_info[-1, 0] == 2
    assert pr_info[-1, 1] == 1


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
