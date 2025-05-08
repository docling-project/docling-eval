import re
from collections import defaultdict

import editdistance
import numpy as np
from docling_core.types.doc.page import SegmentedPage, TextCell, TextCellUnit
from scipy.optimize import linear_sum_assignment

IOU_THRESHOLD = 0.5

IGNORE_CHARS_REGEX = re.compile(r'[\s.,!?;:\'"(){}\[\]<>-]')


def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) for two bounding boxes.

    Boxes are expected in {l, t, r, b} format with coordinates between 0 and 1.
    Returns 0.0 if boxes are invalid or calculation fails.
    """
    try:
        if not all(k in box1 for k in ["l", "t", "r", "b"]) or not all(
            k in box2 for k in ["l", "t", "r", "b"]
        ):
            return 0.0
        if (
            box1["l"] >= box1["r"]
            or box1["t"] >= box1["b"]
            or box2["l"] >= box2["r"]
            or box2["t"] >= box2["b"]
        ):
            return 0.0

        l_inter = max(box1["l"], box2["l"])
        t_inter = max(box1["t"], box2["t"])
        r_inter = min(box1["r"], box2["r"])
        b_inter = min(box1["b"], box2["b"])

        inter_width = max(0, r_inter - l_inter)
        inter_height = max(0, b_inter - t_inter)
        intersection_area = inter_width * inter_height

        box1_area = (box1["r"] - box1["l"]) * (box1["b"] - box1["t"])
        box2_area = (box2["r"] - box2["l"]) * (box2["b"] - box2["t"])

        union_area = box1_area + box2_area - intersection_area

        iou = intersection_area / union_area if union_area > 0 else 0.0
        iou = max(0.0, min(1.0, iou))
        return iou
    except (TypeError, KeyError) as e:
        print(f"Error calculating IoU: {e}. Boxes: {box1}, {box2}")
        return 0.0
    except Exception as e:
        print(f"Unexpected error in calculate_iou: {e}")
        return 0.0


def filter_text(text, ignore_regex=None, ignore_single_chars=False):
    """Filters text by removing specified characters and optionally ignoring if only single chars remain."""
    if text is None:
        return ""
    text_str = str(text).strip()

    if ignore_regex:
        text_str = ignore_regex.sub("", text_str)

    text_str = text_str.strip()

    if ignore_single_chars and len(text_str) <= 1:
        return ""

    return text_str


def get_text_from_element(element):
    """Safely extracts text"""
    return str(element.get("text", element.get("orig", "")))


def get_bbox_from_word(element: TextCell):
    return element.rect


def calculate_word_level_metrics(gt_text, pred_text):
    """
    Calculates word-level comparison metrics using distance.
    Simplified: Returns hits=all GT words if texts match exactly, else hits=0.
    Returns substitutions = all GT words if texts differ but pred exists, else 0.
    """
    gt_words = str(gt_text).split()
    pred_words = str(pred_text).split()
    num_gt_words = len(gt_words)

    # simplified approach based on exact block match for hits/subs
    if gt_text == pred_text and num_gt_words > 0:
        hits = num_gt_words
        substitutions = 0
    elif num_gt_words > 0 or len(pred_words) > 0:
        hits = 0
        # This is a very rough approximation. Real WER needs alignment.
        # If GT exists, and prediction is different (or empty), consider GT words as potential S or D.
        # If GT is empty, and pred exists, these are insertions (I).
        # Let's count GT words as 'not hits' when texts differ.
        # Approximation: Treat all GT words as 'substitutions' if pred is non-empty and differs.
        # Treat all GT words as 'deletions' if pred is empty.
        # Treat all Pred words as 'insertions' if GT is empty.
        # For aggregation, maybe just counting GT words involved is better?
        substitutions = num_gt_words  # Count GT words involved in non-matching pairs for simplicity here.
        # A better approach would align and count S, I, D.
    else:  # Both empty
        hits = 0
        substitutions = 0

    return hits, substitutions, num_gt_words


def evaluate_single_pair(
    gt_data: SegmentedPage, pred_data: SegmentedPage, iou_threshold=0.5
):
    """
    Calculates detection and recognition metrics for a single GT/Pred pair.
    Returns a dictionary containing raw counts and sums for aggregation.
    """
    gt_words = gt_data.word_cells
    pred_words = pred_data.word_cells

    num_gt_original = len(gt_words)
    num_pred_original = len(pred_words)

    pair_results: defaultdict[str, float] = defaultdict(float)
    pair_results["_num_gt_original"] = num_gt_original
    pair_results["_num_pred_original"] = num_pred_original
    pair_results["_num_matches"] = 0
    pair_results["_num_tp"] = 0  # true positives (matched geometry)
    pair_results["_num_fp"] = 0  # false positives (unmatched predictions)
    pair_results["_num_fn"] = 0  # false negatives (unmatched ground truth)

    # --- 1. Geometric Matching (using Optimal Bipartite Matching) ---
    matched_pairs = []
    gt_matched_indices = set()
    pred_matched_indices = set()

    if num_gt_original > 0 and num_pred_original > 0:
        iou_matrix = np.zeros((num_gt_original, num_pred_original))
        valid_gt_indices = []
        valid_pred_indices = []
        gt_map = {}
        pred_map = {}

        current_valid_gt_idx = 0
        for i, gt in enumerate(gt_words):
            gt_bbox = get_bbox_from_word(gt)
            if gt_bbox:
                valid_gt_indices.append(i)
                gt_map[current_valid_gt_idx] = i
                current_valid_gt_idx += 1

        current_valid_pred_idx = 0
        for j, pred in enumerate(pred_words):
            pred_bbox = get_bbox_from_word(pred)
            if pred_bbox:
                valid_pred_indices.append(j)
                pred_map[current_valid_pred_idx] = j
                current_valid_pred_idx += 1

        num_valid_gt = len(valid_gt_indices)
        num_valid_pred = len(valid_pred_indices)

        if num_valid_gt > 0 and num_valid_pred > 0:
            iou_matrix_valid = np.zeros((num_valid_gt, num_valid_pred))
            for i_valid, gt_orig_idx in enumerate(valid_gt_indices):
                for j_valid, pred_orig_idx in enumerate(valid_pred_indices):
                    gt_bbox = get_bbox_from_word(gt_words[gt_orig_idx])
                    pred_bbox = get_bbox_from_word(pred_words[pred_orig_idx])
                    iou_matrix_valid[i_valid, j_valid] = calculate_iou(
                        gt_bbox, pred_bbox
                    )

            cost_matrix = 1.0 - iou_matrix_valid
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r_valid, c_valid in zip(row_ind, col_ind):
                iou = iou_matrix_valid[r_valid, c_valid]
                if iou >= iou_threshold:
                    gt_orig_idx = gt_map[r_valid]
                    pred_orig_idx = pred_map[c_valid]
                    matched_pairs.append(
                        {"gt_idx": gt_orig_idx, "pred_idx": pred_orig_idx, "iou": iou}
                    )
                    gt_matched_indices.add(gt_orig_idx)
                    pred_matched_indices.add(pred_orig_idx)

    all_gt_indices = set(range(num_gt_original))
    all_pred_indices = set(range(num_pred_original))
    false_negatives_indices = all_gt_indices - gt_matched_indices
    false_positives_indices = all_pred_indices - pred_matched_indices

    num_tp = len(matched_pairs)
    num_fp = len(false_positives_indices)
    num_fn = len(false_negatives_indices)

    assert (
        num_gt_original == num_tp + num_fn
    ), f"GT count mismatch: {num_gt_original} != {num_tp} + {num_fn}"
    assert (
        num_pred_original == num_tp + num_fp
    ), f"Pred count mismatch: {num_pred_original} != {num_tp} + {num_fp}"

    pair_results["_num_tp"] = num_tp
    pair_results["_num_fp"] = num_fp
    pair_results["_num_fn"] = num_fn
    pair_results["_num_matches"] = num_tp

    sum_edit_distance_intersection_sensitive = 0.0
    sum_max_length_intersection = 0.0
    word_hits_sensitive = 0
    word_substitutions_sensitive = 0
    total_gt_words_in_tps = 0

    sum_edit_distance_intersection_ignored_chars = 0.0
    sum_max_length_intersection_ignored_chars = 0.0
    sum_edit_distance_intersection_ignored_singles = 0.0
    sum_max_length_intersection_ignored_singles = 0.0

    for match in matched_pairs:
        gt = gt_words[match["gt_idx"]]
        pred = pred_words[match["pred_idx"]]

        gt_text = get_text_from_element(gt)
        pred_text = get_text_from_element(pred)

        dist_sensitive = editdistance.eval(gt_text, pred_text)
        len_gt = len(gt_text)
        len_pred = len(pred_text)
        max_len_sensitive = max(len_gt, len_pred)
        sum_edit_distance_intersection_sensitive += dist_sensitive
        sum_max_length_intersection += max_len_sensitive

        hits, subs, gt_words_count = calculate_word_level_metrics(gt_text, pred_text)
        word_hits_sensitive += hits
        word_substitutions_sensitive += subs
        total_gt_words_in_tps += gt_words_count

        gt_text_filtered_chars = filter_text(
            gt_text, ignore_regex=IGNORE_CHARS_REGEX, ignore_single_chars=False
        )
        pred_text_filtered_chars = filter_text(
            pred_text, ignore_regex=IGNORE_CHARS_REGEX, ignore_single_chars=False
        )
        dist_ignored_chars = editdistance.eval(
            gt_text_filtered_chars, pred_text_filtered_chars
        )
        max_len_ignored_chars = max(
            len(gt_text_filtered_chars), len(pred_text_filtered_chars)
        )
        sum_edit_distance_intersection_ignored_chars += dist_ignored_chars
        sum_max_length_intersection_ignored_chars += max_len_ignored_chars

        gt_text_filtered_singles = filter_text(
            gt_text, ignore_regex=IGNORE_CHARS_REGEX, ignore_single_chars=True
        )
        pred_text_filtered_singles = filter_text(
            pred_text, ignore_regex=IGNORE_CHARS_REGEX, ignore_single_chars=True
        )
        dist_ignored_singles = editdistance.eval(
            gt_text_filtered_singles, pred_text_filtered_singles
        )
        max_len_ignored_singles = max(
            len(gt_text_filtered_singles), len(pred_text_filtered_singles)
        )
        sum_edit_distance_intersection_ignored_singles += dist_ignored_singles
        sum_max_length_intersection_ignored_singles += max_len_ignored_singles

    pair_results["_sum_edit_distance_intersection_sensitive"] = (
        sum_edit_distance_intersection_sensitive
    )
    pair_results["_sum_max_length_intersection"] = sum_max_length_intersection
    pair_results["_word_hits_sensitive"] = word_hits_sensitive
    pair_results["_word_substitutions_sensitive"] = word_substitutions_sensitive
    pair_results["_total_gt_words_in_tps"] = total_gt_words_in_tps
    pair_results["_sum_edit_distance_intersection_ignored_chars"] = (
        sum_edit_distance_intersection_ignored_chars
    )
    pair_results["_sum_max_length_intersection_ignored_chars"] = (
        sum_max_length_intersection_ignored_chars
    )
    pair_results["_sum_edit_distance_intersection_ignored_singles"] = (
        sum_edit_distance_intersection_ignored_singles
    )
    pair_results["_sum_max_length_intersection_ignored_singles"] = (
        sum_max_length_intersection_ignored_singles
    )

    text_length_false_positives = 0
    text_length_fp_ignored_chars = 0
    text_length_fp_ignored_singles = 0
    for idx in false_positives_indices:
        fp = pred_words[idx]
        fp_text = get_text_from_element(fp)
        text_length_false_positives += len(fp_text)
        text_length_fp_ignored_chars += len(
            filter_text(
                fp_text, ignore_regex=IGNORE_CHARS_REGEX, ignore_single_chars=False
            )
        )
        text_length_fp_ignored_singles += len(
            filter_text(
                fp_text, ignore_regex=IGNORE_CHARS_REGEX, ignore_single_chars=True
            )
        )

    text_length_false_negatives = 0
    text_length_fn_ignored_chars = 0
    text_length_fn_ignored_singles = 0
    num_fn_ignored_chars_becomes_empty = 0
    num_fn_ignored_singles_becomes_empty = 0
    total_words_in_gt = 0

    for idx in false_negatives_indices:
        fn = gt_words[idx]
        fn_text = get_text_from_element(fn)
        text_length_false_negatives += len(fn_text)

        fn_text_filtered_chars = filter_text(
            fn_text, ignore_regex=IGNORE_CHARS_REGEX, ignore_single_chars=False
        )
        text_length_fn_ignored_chars += len(fn_text_filtered_chars)
        if not fn_text_filtered_chars and fn_text:
            num_fn_ignored_chars_becomes_empty += 1

        fn_text_filtered_singles = filter_text(
            fn_text, ignore_regex=IGNORE_CHARS_REGEX, ignore_single_chars=True
        )
        text_length_fn_ignored_singles += len(fn_text_filtered_singles)
        if not fn_text_filtered_singles and fn_text:
            num_fn_ignored_singles_becomes_empty += 1

    for gt in gt_words:
        total_words_in_gt += len(get_text_from_element(gt).split())

    pair_results["_text_length_false_positives"] = text_length_false_positives
    pair_results["_text_length_false_negatives"] = text_length_false_negatives
    pair_results["_text_length_fp_ignored_chars"] = text_length_fp_ignored_chars
    pair_results["_text_length_fn_ignored_chars"] = text_length_fn_ignored_chars
    pair_results["_text_length_fp_ignored_singles"] = text_length_fp_ignored_singles
    pair_results["_text_length_fn_ignored_singles"] = text_length_fn_ignored_singles
    pair_results["_num_fn_ignoring_chars_becomes_empty"] = (
        num_fn_ignored_chars_becomes_empty
    )
    pair_results["_num_fn_ignoring_singles_becomes_empty"] = (
        num_fn_ignored_singles_becomes_empty
    )
    pair_results["_words_in_gt"] = total_words_in_gt

    pair_results["_FN_zero_iou"] = 0
    pair_results["_FP_low_iou"] = 0
    pair_results["_FN_ambiguous"] = 0
    pair_results["_FP_ambiguous"] = 0
    pair_results["_FP_zero_iou"] = 0
    pair_results["_model_merged"] = 0
    pair_results["_gt_merged"] = 0
    pair_results["_FN_low_iou"] = 0

    return dict(pair_results)


def calculate_aggregated_metrics(aggregated_results, total_pairs):
    """Calculates final metrics based on aggregated counts and sums."""

    if total_pairs == 0:
        print("Warning: No valid file pairs processed.")
        return {}

    num_tp = aggregated_results["_num_tp"]
    num_fp = aggregated_results["_num_fp"]
    num_fn = aggregated_results["_num_fn"]
    num_gt_original = aggregated_results["_num_gt_original"]
    num_pred_original = aggregated_results["_num_pred_original"]

    sum_edit_distance_intersection_sensitive = aggregated_results[
        "_sum_edit_distance_intersection_sensitive"
    ]
    sum_max_length_intersection = aggregated_results["_sum_max_length_intersection"]
    word_hits_sensitive = aggregated_results["_word_hits_sensitive"]
    word_substitutions_sensitive = aggregated_results["_word_substitutions_sensitive"]
    total_gt_words_in_tps = aggregated_results["_total_gt_words_in_tps"]
    words_in_gt = aggregated_results["_words_in_gt"]

    sum_edit_distance_intersection_ignored_chars = aggregated_results[
        "_sum_edit_distance_intersection_ignored_chars"
    ]
    sum_max_length_intersection_ignored_chars = aggregated_results[
        "_sum_max_length_intersection_ignored_chars"
    ]
    sum_edit_distance_intersection_ignored_singles = aggregated_results[
        "_sum_edit_distance_intersection_ignored_singles"
    ]
    sum_max_length_intersection_ignored_singles = aggregated_results[
        "_sum_max_length_intersection_ignored_singles"
    ]

    text_length_false_positives = aggregated_results["_text_length_false_positives"]
    text_length_false_negatives = aggregated_results["_text_length_false_negatives"]
    text_length_fp_ignored_chars = aggregated_results["_text_length_fp_ignored_chars"]
    text_length_fn_ignored_chars = aggregated_results["_text_length_fn_ignored_chars"]
    text_length_fp_ignored_singles = aggregated_results[
        "_text_length_fp_ignored_singles"
    ]
    text_length_fn_ignored_singles = aggregated_results[
        "_text_length_fn_ignored_singles"
    ]

    num_fn_ignored_chars_becomes_empty = aggregated_results[
        "_num_fn_ignoring_chars_becomes_empty"
    ]
    num_fn_ignored_singles_becomes_empty = aggregated_results[
        "_num_fn_ignoring_singles_becomes_empty"
    ]

    precision = num_tp / (num_tp + num_fp) if (num_tp + num_fp) > 0 else 0.0
    recall = num_tp / (num_tp + num_fn) if (num_tp + num_fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Calculate Normalised Edit Distance (NED) for different filtering strategies.
    # NED is calculated as (Sum of Edit Distances) / (Sum of Max Lengths), typically multiplied by 100.
    # It represents the average character error rate, considering insertions, deletions, and substitutions.
    norm_ed_tp_only = (
        (sum_edit_distance_intersection_sensitive / sum_max_length_intersection) * 100.0
        if sum_max_length_intersection > 0
        else 0.0
    )

    denominator_no_fp = sum_max_length_intersection + text_length_false_negatives
    norm_ed_without_fp = (
        (
            (sum_edit_distance_intersection_sensitive + text_length_false_negatives)
            / denominator_no_fp
        )
        * 100.0
        if denominator_no_fp > 0
        else 0.0
    )

    denominator_all = (
        sum_max_length_intersection
        + text_length_false_negatives
        + text_length_false_positives
    )
    norm_ed_all_cells = (
        (
            (
                sum_edit_distance_intersection_sensitive
                + text_length_false_negatives
                + text_length_false_positives
            )
            / denominator_all
        )
        * 100.0
        if denominator_all > 0
        else 0.0
    )

    denominator_no_fp_ign_chars = (
        sum_max_length_intersection_ignored_chars + text_length_fn_ignored_chars
    )
    norm_ed_ignoring_fp_and_chars = (
        (
            (
                sum_edit_distance_intersection_ignored_chars
                + text_length_fn_ignored_chars
            )
            / denominator_no_fp_ign_chars
        )
        * 100.0
        if denominator_no_fp_ign_chars > 0
        else 0.0
    )

    denominator_all_ign_chars = (
        sum_max_length_intersection_ignored_chars
        + text_length_fn_ignored_chars
        + text_length_fp_ignored_chars
    )
    norm_ed_ignoring_chars = (
        (
            (
                sum_edit_distance_intersection_ignored_chars
                + text_length_fn_ignored_chars
                + text_length_fp_ignored_chars
            )
            / denominator_all_ign_chars
        )
        * 100.0
        if denominator_all_ign_chars > 0
        else 0.0
    )

    denominator_no_fp_ign_singles = (
        sum_max_length_intersection_ignored_singles + text_length_fn_ignored_singles
    )
    norm_ed_ignoring_singles_and_fp = (
        (
            (
                sum_edit_distance_intersection_ignored_singles
                + text_length_fn_ignored_singles
            )
            / denominator_no_fp_ign_singles
        )
        * 100.0
        if denominator_no_fp_ign_singles > 0
        else 0.0
    )

    denominator_all_ign_singles = (
        sum_max_length_intersection_ignored_singles
        + text_length_fn_ignored_singles
        + text_length_fp_ignored_singles
    )
    norm_ed_ignoring_singles = (
        (
            (
                sum_edit_distance_intersection_ignored_singles
                + text_length_fn_ignored_singles
                + text_length_fp_ignored_singles
            )
            / denominator_all_ign_singles
        )
        * 100.0
        if denominator_all_ign_singles > 0
        else 0.0
    )

    # Calculate Accuracy/Edit-Score variants. Edit Score is typically 100 - NED.
    edit_score_tp_only = 100.0 - norm_ed_tp_only
    edit_score_all_cells = 100.0 - norm_ed_all_cells

    word_accuracy_tp_only = (
        (word_hits_sensitive / total_gt_words_in_tps) * 100.0
        if total_gt_words_in_tps > 0
        else 0.0
    )

    word_accuracy_all_cells = (
        (word_hits_sensitive / words_in_gt) * 100.0 if words_in_gt > 0 else 0.0
    )

    # Calculate the Harmonic Mean of the "Norm_ED (Without FP)" accuracy (scaled to 0-1) and Precision.
    # Hmean gives a balanced score between two metrics. Used here to combine geometric precision
    # with text recognition accuracy for matched and missed GT boxes.
    norm_ed_without_fp_accuracy = (100.0 - norm_ed_without_fp) / 100.0
    if (norm_ed_without_fp_accuracy + precision) > 0:
        hmean_norm_ed_precision = (
            2
            * (norm_ed_without_fp_accuracy * precision)
            / (norm_ed_without_fp_accuracy + precision)
        )
    else:
        hmean_norm_ed_precision = 0.0

    final_results = {
        "F1": f1 * 100.0,
        "Recall": recall * 100.0,
        "Precision": precision * 100.0,
        "Norm_ED (TP-Only)": norm_ed_tp_only,
        "Norm_ED (Without FP)": norm_ed_without_fp,
        "Norm_ED (All-cells)": norm_ed_all_cells,
        "Norm_ED (Ignoring FP and some chars)": norm_ed_ignoring_fp_and_chars,
        "Norm_ED (Ignoring some chars)": norm_ed_ignoring_chars,
        "Norm_ED (Ignoring single chars and FP)": norm_ed_ignoring_singles_and_fp,
        "Norm_ED (Ignoring single chars)": norm_ed_ignoring_singles,
        "Word-accuracy (TP-Only)": word_accuracy_tp_only,
        "Word-accuracy (All-cells)": word_accuracy_all_cells,
        "Edit-score (TP-Only)": edit_score_tp_only,
        "Edit-score (All-cells)": edit_score_all_cells,
        "Hmean: (Norm_ED [Without FP], precision)": hmean_norm_ed_precision * 100.0,
        "#Word-Hits": int(round(word_hits_sensitive)),
        "#Word-Substitutions": int(round(word_substitutions_sensitive)),
        "#FN": int(round(num_fn)),
        "#FP": int(round(num_fp)),
        "#FN (Ignoring some chars)": int(round(num_fn_ignored_chars_becomes_empty)),
        "#FN (Ignoring single chars)": int(round(num_fn_ignored_singles_becomes_empty)),
        "#words-in-GT": int(round(words_in_gt)),
        "#FN - zero iou": int(round(aggregated_results["_FN_zero_iou"])),
        "#FP - low iou": int(round(aggregated_results["_FP_low_iou"])),
        "#FN - ambiguous match": int(round(aggregated_results["_FN_ambiguous"])),
        "#FP - ambiguous match": int(round(aggregated_results["_FP_ambiguous"])),
        "#FP - zero iou": int(round(aggregated_results["_FP_zero_iou"])),
        "#Original Model boxes merged": int(round(aggregated_results["_model_merged"])),
        "#Original GT boxes merged": int(round(aggregated_results["_gt_merged"])),
        "#FN - low iou": int(round(aggregated_results["_FN_low_iou"])),
        "_num_tp_gt_based": int(round(num_tp)),
        "_num_fp": int(round(num_fp)),
        "_num_fn": int(round(num_fn)),
        "_num_gt_original": int(round(num_gt_original)),
        "_num_pred_original": int(round(num_pred_original)),
        "_num_matches": int(round(num_tp)),
        "_word_hits_sensitive": int(round(word_hits_sensitive)),
        "_word_substitutions_sensitive": int(round(word_substitutions_sensitive)),
        "_sum_edit_distance_intersection_sensitive": sum_edit_distance_intersection_sensitive,
        "_sum_max_length_intersection": sum_max_length_intersection,
        "_text_length_false_positives": text_length_false_positives,
        "_text_length_false_negatives": text_length_false_negatives,
        "_num_fn_ignoring_chars": int(round(num_fn_ignored_chars_becomes_empty)),
        "_num_fn_ignoring_singles": int(round(num_fn_ignored_singles_becomes_empty)),
    }

    formatted_results = {}
    for key, value in final_results.items():
        if isinstance(value, float):
            formatted_results[key] = round(value, 1)
        elif isinstance(value, (int, np.integer)):
            formatted_results[key] = int(value)
        else:
            formatted_results[key] = value

    # Apply special handling if the total number of True Positives across all files is zero.
    # In this specific case, some error metrics (like TP-Only NED) are defined as 100%,
    # and corresponding accuracy metrics are 0%.
    if num_tp == 0:
        formatted_results["Norm_ED (TP-Only)"] = 100.0
        if denominator_no_fp == 0:
            formatted_results["Norm_ED (Without FP)"] = 100.0
        else:
            formatted_results["Norm_ED (Without FP)"] = round(norm_ed_without_fp, 1)

        if denominator_all == 0:
            formatted_results["Norm_ED (All-cells)"] = 100.0
        else:
            formatted_results["Norm_ED (All-cells)"] = round(norm_ed_all_cells, 1)

        if denominator_no_fp_ign_chars == 0:
            formatted_results["Norm_ED (Ignoring FP and some chars)"] = 100.0
        else:
            formatted_results["Norm_ED (Ignoring FP and some chars)"] = round(
                norm_ed_ignoring_fp_and_chars, 1
            )
        if denominator_all_ign_chars == 0:
            formatted_results["Norm_ED (Ignoring some chars)"] = 100.0
        else:
            formatted_results["Norm_ED (Ignoring some chars)"] = round(
                norm_ed_ignoring_chars, 1
            )

        if denominator_no_fp_ign_singles == 0:
            formatted_results["Norm_ED (Ignoring single chars and FP)"] = 100.0
        else:
            formatted_results["Norm_ED (Ignoring single chars and FP)"] = round(
                norm_ed_ignoring_singles_and_fp, 1
            )
        if denominator_all_ign_singles == 0:
            formatted_results["Norm_ED (Ignoring single chars)"] = 100.0
        else:
            formatted_results["Norm_ED (Ignoring single chars)"] = round(
                norm_ed_ignoring_singles, 1
            )

        formatted_results["Word-accuracy (TP-Only)"] = 0.0
        formatted_results["Edit-score (TP-Only)"] = 0.0
        formatted_results["Edit-score (All-cells)"] = round(
            100.0 - formatted_results["Norm_ED (All-cells)"], 1
        )

        formatted_results["Hmean: (Norm_ED [Without FP], precision)"] = 0.0

    return formatted_results
