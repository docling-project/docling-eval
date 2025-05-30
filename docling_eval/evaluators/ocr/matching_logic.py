from typing import Any, Dict, List, Tuple

from docling_eval.evaluators.ocr.evaluation_models import (
    BenchmarkIntersectionInfo,
    Word,
    _CalculationConstants,
)
from docling_eval.evaluators.ocr.geometry_utils import (
    box_to_key,
    calculate_box_intersection_info,
    calculate_box_intersection_info_extended,
    is_horizontal,
)

detection_match_condition_iou_coverage_threshold: Any = (
    lambda boxes_info: boxes_info.prediction_box_portion_covered > 0.5
    or boxes_info.gt_box_portion_covered > 0.5
)
current_detection_match_condition: Any = (
    detection_match_condition_iou_coverage_threshold
)


def match_ground_truth_to_prediction_words(
    ground_truth_words: List[Word], prediction_words: List[Word]
) -> Tuple[
    Dict[
        Tuple[float, float, float, float], List[Tuple[Word, BenchmarkIntersectionInfo]]
    ],
    Dict[
        Tuple[float, float, float, float], List[Tuple[Word, BenchmarkIntersectionInfo]]
    ],
]:

    gt_to_prediction_map: Dict[
        Tuple[float, float, float, float], List[Tuple[Word, BenchmarkIntersectionInfo]]
    ] = {}
    for gt_word_item in ground_truth_words:
        intersections_list: List[Tuple[Word, BenchmarkIntersectionInfo]] = []
        for pred_word_item in prediction_words:
            (
                x_overlap,
                y_overlap,
                intersect_area,
                union_val,
                iou_val,
                gt_portion_covered,
                pred_portion_covered,
            ) = calculate_box_intersection_info(gt_word_item.bbox, pred_word_item.bbox)
            if intersect_area > 0:
                intersections_list.append(
                    (
                        pred_word_item,
                        BenchmarkIntersectionInfo(
                            x_axis_overlap=x_overlap,
                            y_axis_overlap=y_overlap,
                            intersection_area=intersect_area,
                            union_area=union_val,
                            iou=iou_val,
                            gt_box_portion_covered=gt_portion_covered,
                            prediction_box_portion_covered=pred_portion_covered,
                        ),
                    )
                )
        key_for_gt_box: Tuple[float, float, float, float] = box_to_key(
            gt_word_item.bbox
        )
        if len(intersections_list) > 1:
            intersections_list = sorted(
                intersections_list, key=lambda x: x[1].intersection_area
            )
        gt_to_prediction_map[key_for_gt_box] = intersections_list

    prediction_to_gt_map: Dict[
        Tuple[float, float, float, float], List[Tuple[Word, BenchmarkIntersectionInfo]]
    ] = {}
    for pred_word_item in prediction_words:
        intersections_list_pred_to_gt: List[Tuple[Word, BenchmarkIntersectionInfo]] = []
        for gt_word_item in ground_truth_words:
            (
                x_overlap,
                y_overlap,
                intersect_area,
                union_val,
                iou_val,
                gt_portion_covered,
                pred_portion_covered,
            ) = calculate_box_intersection_info(gt_word_item.bbox, pred_word_item.bbox)
            if intersect_area > 0:
                intersections_list_pred_to_gt.append(
                    (
                        gt_word_item,
                        BenchmarkIntersectionInfo(
                            x_axis_overlap=x_overlap,
                            y_axis_overlap=y_overlap,
                            intersection_area=intersect_area,
                            union_area=union_val,
                            iou=iou_val,
                            gt_box_portion_covered=gt_portion_covered,
                            prediction_box_portion_covered=pred_portion_covered,
                        ),
                    )
                )
        key_for_pred_box: Tuple[float, float, float, float] = box_to_key(
            pred_word_item.bbox
        )
        intersections_list_pred_to_gt = sorted(
            intersections_list_pred_to_gt, key=lambda x: x[1].intersection_area
        )
        prediction_to_gt_map[key_for_pred_box] = intersections_list_pred_to_gt
    return gt_to_prediction_map, prediction_to_gt_map


def refine_prediction_to_many_gt_boxes(
    prediction_word: Word, intersections: List[Tuple[Word, BenchmarkIntersectionInfo]]
) -> Tuple[
    List[Tuple[Word, BenchmarkIntersectionInfo]],
    List[Tuple[Word, BenchmarkIntersectionInfo]],
]:
    sorted_intersections: List[Tuple[Word, BenchmarkIntersectionInfo]] = sorted(
        [(gt_box, boxes_info) for (gt_box, boxes_info) in intersections],
        key=lambda x: x[1].intersection_area,
        reverse=True,
    )
    orientations: List[bool] = [is_horizontal(x) for x, _ in sorted_intersections]
    num_horizontal: int = sum(orientations)
    num_vertical: int = len(orientations) - num_horizontal

    valid_line_intersections: List[Tuple[Word, BenchmarkIntersectionInfo]] = [
        sorted_intersections[0]
    ]
    invalid_line_intersections: List[Tuple[Word, BenchmarkIntersectionInfo]] = []

    for gt_box, boxes_info in sorted_intersections[1:]:
        can_be_added_to_line: bool = True
        for anchor_box, _ in [sorted_intersections[0]]:
            (_, _, _, _, _, x_iou, y_iou) = calculate_box_intersection_info_extended(
                gt_box.bbox, anchor_box.bbox
            )

            height_ratio: float = min(gt_box.bbox.height, anchor_box.bbox.height) / max(
                gt_box.bbox.height + _CalculationConstants.EPS,
                anchor_box.bbox.height + _CalculationConstants.EPS,
            )
            are_words_in_same_line: bool = (
                (x_iou < 0.2 and y_iou > 0)
                if height_ratio < 0.5
                else (x_iou < 0.2 and y_iou > 0.3)
            )
            if not are_words_in_same_line:
                can_be_added_to_line = False
        if can_be_added_to_line:
            valid_line_intersections.append((gt_box, boxes_info))
        else:
            invalid_line_intersections.append((gt_box, boxes_info))

    refined_line_intersections: List[List[Tuple[Word, BenchmarkIntersectionInfo]]] = [
        valid_line_intersections,
        invalid_line_intersections,
    ]

    valid_column_intersections: List[Tuple[Word, BenchmarkIntersectionInfo]] = [
        sorted_intersections[0]
    ]
    invalid_column_intersections: List[Tuple[Word, BenchmarkIntersectionInfo]] = []
    for gt_box, boxes_info in sorted_intersections[1:]:
        can_be_added_to_column: bool = True
        for anchor_box, _ in [sorted_intersections[0]]:
            (_, _, _, _, _, x_iou, y_iou) = calculate_box_intersection_info_extended(
                gt_box.bbox, anchor_box.bbox
            )

            width_ratio: float = min(gt_box.bbox.width, anchor_box.bbox.width) / max(
                gt_box.bbox.width + _CalculationConstants.EPS,
                anchor_box.bbox.width + _CalculationConstants.EPS,
            )

            are_words_in_same_column: bool = (
                (y_iou < 0.2 and x_iou > 0)
                if width_ratio < 0.5
                else (y_iou < 0.2 and x_iou > 0.5)
            )
            if not are_words_in_same_column:
                can_be_added_to_column = False
        if can_be_added_to_column:
            valid_column_intersections.append((gt_box, boxes_info))
        else:
            invalid_column_intersections.append((gt_box, boxes_info))

    refined_column_intersections: List[List[Tuple[Word, BenchmarkIntersectionInfo]]] = [
        valid_column_intersections,
        invalid_column_intersections,
    ]

    chosen_refined_intersections: List[List[Tuple[Word, BenchmarkIntersectionInfo]]] = (
        []
    )
    if num_horizontal > num_vertical:
        chosen_refined_intersections = refined_line_intersections
    else:
        chosen_refined_intersections = refined_column_intersections

    if len(chosen_refined_intersections[1]) > 0:
        return [], intersections
    else:
        return (
            chosen_refined_intersections[0],
            chosen_refined_intersections[1],
        )
