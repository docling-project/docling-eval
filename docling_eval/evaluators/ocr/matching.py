from types import SimpleNamespace

from docling_eval.evaluators.ocr.geometry_utils import (
    box_to_key,
    convert_locations_to_float,
    info_for_boxes,
    info_for_boxes_extended,
    is_horizontal,
)

EPS = 1.0e-6


def match_ground_truth_to_prediction_words(gt_words: list, prediction_words: list):
    convert_locations_to_float(prediction_words)
    convert_locations_to_float(gt_words)

    gt_to_prediction_boxes_map = {}
    for gt_box in gt_words:
        intersections = []
        for prediction_box in prediction_words:
            (
                x_axis_overlap,
                y_axis_overlap,
                intersection_area,
                union_area,
                iou,
                current_gt_box_portion_covered,
                current_prediction_box_portion_covered,
            ) = info_for_boxes(gt_box["location"], prediction_box["location"])
            if intersection_area > 0:
                intersections.append(
                    (
                        prediction_box,
                        SimpleNamespace(
                            x_axis_overlap=x_axis_overlap,
                            y_axis_overlap=y_axis_overlap,
                            intersection_area=intersection_area,
                            union_area=union_area,
                            iou=iou,
                            gt_box_portion_covered=current_gt_box_portion_covered,
                            prediction_box_portion_covered=current_prediction_box_portion_covered,
                        ),
                    )
                )
        box_key = box_to_key(gt_box["location"])
        if len(intersections) > 1:
            intersections = sorted(intersections, key=lambda x: x[1].intersection_area)
        gt_to_prediction_boxes_map[box_key] = intersections

    prediction_to_gt_boxes_map = {}
    for prediction_box in prediction_words:
        intersections = []
        for gt_box in gt_words:
            (
                x_axis_overlap,
                y_axis_overlap,
                intersection_area,
                union_area,
                iou,
                current_gt_box_portion_covered,
                current_prediction_box_portion_covered,
            ) = info_for_boxes(gt_box["location"], prediction_box["location"])
            if intersection_area > 0:
                intersections.append(
                    (
                        gt_box,
                        SimpleNamespace(
                            x_axis_overlap=x_axis_overlap,
                            y_axis_overlap=y_axis_overlap,
                            intersection_area=intersection_area,
                            union_area=union_area,
                            iou=iou,
                            gt_box_portion_covered=current_gt_box_portion_covered,
                            prediction_box_portion_covered=current_prediction_box_portion_covered,
                        ),
                    )
                )
        box_key = box_to_key(prediction_box["location"])
        intersections = sorted(intersections, key=lambda x: x[1].intersection_area)
        prediction_to_gt_boxes_map[box_key] = intersections
    return gt_to_prediction_boxes_map, prediction_to_gt_boxes_map


def refine_prediction_to_many_gt_boxes(prediction_word, intersections):
    s = sorted(
        [(gt_box, boxes_info) for (gt_box, boxes_info) in intersections],
        key=lambda x: x[1].intersection_area,
        reverse=True,
    )
    a = [is_horizontal(x) for x, _ in s]
    num_horizontal = sum(a)
    num_vertical = len(a) - num_horizontal

    valid_intersections_line, invalid_intersections_line = [s[0]], []
    for gt_box, boxes_info in s[1:]:
        can_be_added = True
        for b, _ in [s[0]]:
            (
                x_axis_overlap,
                y_axis_overlap,
                intersection_area,
                union_area,
                iou,
                x_axis_iou,
                y_axis_iou,
            ) = info_for_boxes_extended(gt_box["location"], b["location"])
            height_ratio = min(
                gt_box["location"]["height"], b["location"]["height"]
            ) / max(gt_box["location"]["height"] + EPS, b["location"]["height"] + EPS)
            words_in_same_line = (
                (x_axis_iou < 0.2 and y_axis_iou > 0)
                if height_ratio < 0.5
                else (x_axis_iou < 0.2 and y_axis_iou > 0.3)
            )
            if not words_in_same_line:
                can_be_added = False
        if can_be_added:
            valid_intersections_line.append((gt_box, boxes_info))
        else:
            invalid_intersections_line.append((gt_box, boxes_info))
    line_intersections_refined = [valid_intersections_line, invalid_intersections_line]

    valid_intersections_col, invalid_intersections_col = [s[0]], []
    for gt_box, boxes_info in s[1:]:
        can_be_added = True
        for b, _ in [s[0]]:
            (
                x_axis_overlap,
                y_axis_overlap,
                intersection_area,
                union_area,
                iou,
                x_axis_iou,
                y_axis_iou,
            ) = info_for_boxes_extended(gt_box["location"], b["location"])
            width_ratio = min(
                gt_box["location"]["width"], b["location"]["width"]
            ) / max(gt_box["location"]["width"] + EPS, b["location"]["width"] + EPS)

            words_in_same_column = (
                (y_axis_iou < 0.2 and x_axis_iou > 0)
                if width_ratio < 0.5
                else (y_axis_iou < 0.2 and x_axis_iou > 0.5)
            )
            if not words_in_same_column:
                can_be_added = False
        if can_be_added:
            valid_intersections_col.append((gt_box, boxes_info))
        else:
            invalid_intersections_col.append((gt_box, boxes_info))
    column_intersections_refined = [valid_intersections_col, invalid_intersections_col]

    chosen_intersections = []
    if num_horizontal > num_vertical:
        chosen_intersections = line_intersections_refined
    else:
        chosen_intersections = column_intersections_refined

    if len(chosen_intersections[1]) > 0:
        return [], intersections
    else:
        return (
            chosen_intersections[0],
            chosen_intersections[1],
        )
