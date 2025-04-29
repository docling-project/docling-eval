import copy
import string
from collections import namedtuple
from functools import lru_cache
from types import SimpleNamespace

import edit_distance

__eps = 1.0e-6

edit_distance_chars_map = {}

BoxesTypes = namedtuple("BoxesTypes", ["zero_iou", "low_iou", "ambiguous_match"])


def detection_match_condition_2(boxes_info):
    return (
        boxes_info.model_box_portion_covered > 0.7
        and boxes_info.gt_box_portion_covered > 0.8
    )


def detection_match_condition_1(boxes_info):
    return (
        boxes_info.model_box_portion_covered > 0.5
        or boxes_info.gt_box_portion_covered > 0.5
    )


def detection_match_condition_3(boxes_info):
    return boxes_info.x_axis_overlap > 0.7 and boxes_info.y_axis_overlap > 0.6


detection_match_condition = detection_match_condition_1


def get_metrics_per_single_word(
    detected_word,
    matched_gt_word=None,
    iou=-1,
    distance=-1,
    distance_insensitive=-1,
    norm_ed=-1,
    norm_ed_insensitive=-1,
    is_orientation_correct=0,
):
    metrics = {}
    metrics["predicted_word"] = detected_word["word"]
    metrics["word-weight"] = detected_word["word-weight"]
    metrics["predicted_location"] = detected_word["location"]

    metrics["predicted_word_confidence"] = detected_word.get("confidence", "")
    metrics["predicted_character_confidence"] = detected_word.get(
        "character_confidence", ""
    )

    if matched_gt_word is None:
        metrics["correct_detection"] = False
        return metrics

    metrics["correct_detection"] = True
    metrics["gt_word"] = matched_gt_word["word"]
    metrics["gt_location"] = matched_gt_word["location"]
    metrics["intersection_over_union"] = iou
    metrics["edit_distance"] = distance
    metrics["edit_distance_case_insensitive"] = distance_insensitive
    metrics["is_orientation_correct"] = is_orientation_correct

    max_word_length = max(
        len(detected_word["word"]), len(matched_gt_word["word"]), __eps
    )
    if max_word_length > 0:
        metrics["normalized_edit_distance"] = norm_ed
        metrics["normalized_edit_distance_case_insensitive"] = norm_ed_insensitive
    else:
        metrics["normalized_edit_distance"] = -1
        metrics["normalized_edit_distance_case_insensitive"] = -1

    return metrics


def is_horizontal(word):
    height = word["location"]["height"]
    width = word["location"]["width"]
    if height > int(2 * width) and len(word["word"]) > 1:
        return False
    return True


def refine_detection_to_many_gt_boxes(model_word, intersections):
    sorted_intersections = sorted(
        [(gt_box, boxes_info) for (gt_box, boxes_info) in intersections],
        key=lambda x: x[1].intersection_area,
        reverse=True,
    )

    is_horizontal_list = [is_horizontal(x) for x, _ in sorted_intersections]
    num_horizontal = sum(is_horizontal_list)
    num_vertical = len(is_horizontal_list) - num_horizontal

    # Process for line (row) detection
    valid_line_intersections, invalid_line_intersections = [sorted_intersections[0]], []
    for gt_box, boxes_info in sorted_intersections[1:]:
        can_be_added = True
        for b, _ in [sorted_intersections[0]]:
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
            ) / max(gt_box["location"]["height"], b["location"]["height"])
            words_in_same_line = (
                (x_axis_iou < 0.2 and y_axis_iou > 0)
                if height_ratio < 0.5
                else (x_axis_iou < 0.2 and y_axis_iou > 0.3)
            )
            if not words_in_same_line:
                can_be_added = False
        if can_be_added:
            valid_line_intersections.append((gt_box, boxes_info))
        else:
            invalid_line_intersections.append((gt_box, boxes_info))
    line_intersections = [(valid_line_intersections), (invalid_line_intersections)]

    valid_col_intersections, invalid_col_intersections = [sorted_intersections[0]], []
    for gt_box, boxes_info in sorted_intersections[1:]:
        can_be_added = True
        for b, _ in [sorted_intersections[0]]:
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
            ) / max(gt_box["location"]["width"], b["location"]["width"])
            words_in_same_column = (
                (y_axis_iou < 0.2 and x_axis_iou > 0)
                if width_ratio < 0.5
                else (y_axis_iou < 0.2 and x_axis_iou > 0.5)
            )
            if not words_in_same_column:
                can_be_added = False
        if can_be_added:
            valid_col_intersections.append((gt_box, boxes_info))
        else:
            invalid_col_intersections.append((gt_box, boxes_info))
    column_intersections = [(valid_col_intersections), (invalid_col_intersections)]

    # words_in_row = len(line_intersections[0])
    # words_in_col = len(column_intersections[0])
    if num_horizontal > num_vertical:
        result = line_intersections
    else:
        result = column_intersections

    if len(result[1]) > 0:
        return [], intersections
    else:
        return result[0], result[1]


def replace_chars_by_map(text, map_):
    text_ = "".join(idx if idx not in map_ else map_[idx] for idx in text)
    return text_


def my_edit_distance(hyp, ref, string_normalize_map={}):
    hyp, ref = replace_chars_by_map(hyp, string_normalize_map), replace_chars_by_map(
        ref, string_normalize_map
    )
    # hyp = hyp.replace(',', '.')
    # ref = ref.replace(',', '.')
    if len(edit_distance_chars_map):
        hyp = "".join(
            idx if idx not in edit_distance_chars_map else edit_distance_chars_map[idx]
            for idx in hyp
        )
        ref = "".join(
            idx if idx not in edit_distance_chars_map else edit_distance_chars_map[idx]
            for idx in ref
        )
    sm = edit_distance.SequenceMatcher(hyp, ref)

    return sm.distance()


def valid_match(boxes_info):
    # here we could add a condition that would indicate a gt_box and a detection_box as valid only if they meet a certain criteria
    x = 1
    return True


def unify_words(words, add_space_between_words=True):
    separator = " " if add_space_between_words else ""
    unified_word = {"word": ""}
    left = float("inf")
    top = float("inf")
    right = 0
    bottom = 0
    sorted_words = sorted(words, key=lambda k: k["location"]["left"])
    for word in sorted_words:
        unified_word["word"] = unified_word["word"] + separator + word["word"]
        left = min(left, word["location"]["left"])
        top = min(top, word["location"]["top"])
        right = max(right, word["location"]["right"])
        bottom = max(bottom, word["location"]["bottom"])

    unified_word["word"] = unified_word["word"].lstrip()
    unified_word["word-weight"] = len(sorted_words)
    unified_word["location"] = {}
    unified_word["location"]["top"] = top
    unified_word["location"]["left"] = left
    unified_word["location"]["width"] = right - left
    unified_word["location"]["height"] = bottom - top
    unified_word["location"]["right"] = right
    unified_word["location"]["bottom"] = bottom
    unified_word["polygon"] = create_polygon_from_rect(unified_word["location"])
    unified_word["confidence"] = 0
    if "vertical" in words[0]:
        unified_word["vertical"] = words[0]["vertical"]
    return unified_word


def create_polygon_from_rect(location):
    polygon = []

    point_1 = {"x": location["left"], "y": location["top"]}
    polygon.append(point_1)

    point_2 = {"x": location["left"] + location["width"], "y": location["top"]}
    polygon.append(point_2)

    point_3 = {
        "x": location["left"] + location["width"],
        "y": location["top"] + location["height"],
    }
    polygon.append(point_3)

    point_4 = {"x": location["left"], "y": location["top"] + location["height"]}
    polygon.append(point_4)

    return polygon


def convert_locations_to_float(words):
    for word in words:
        location = word["location"]
        location["top"] = float(location["top"])
        location["left"] = float(location["left"])
        location["width"] = float(location["width"])
        location["height"] = float(location["height"])
        location["right"] = location["left"] + location["width"]
        location["bottom"] = location["top"] + location["height"]
    return


def boxes_are_equal(box1, box2):
    if box1["top"] != box2["top"]:
        return False
    if box1["left"] != box2["left"]:
        return False
    if box1["right"] != box2["right"]:
        return False
    if box1["bottom"] != box2["bottom"]:
        return False
    return True


def info_for_boxes_extended(box1, box2):
    # x_axis_overlap, y_axis_overlap, intersection_area, union_area, iou = 0, 0, 0, 0, 0
    x_axis_overlap = get_x_axis_overlap(box1, box2)
    y_axis_overlap = get_y_axis_overlap(box1, box2)
    intersection_area = x_axis_overlap * y_axis_overlap
    box1_area = box1["width"] * box1["height"]
    box2_area = box2["width"] * box2["height"]
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area

    x_axis_iou = x_axis_overlap / get_x_axis_union(box1, box2)
    y_axis_iou = y_axis_overlap / get_y_axis_union(box1, box2)
    # box1_portion_covered = intersection_area / box1_area
    # box2_portion_covered = intersection_area / box2_area
    return (
        x_axis_overlap,
        y_axis_overlap,
        intersection_area,
        union_area,
        iou,
        x_axis_iou,
        y_axis_iou,
    )


def info_for_boxes(box1, box2):
    (
        x_axis_overlap,
        y_axis_overlap,
        intersection_area,
        union_area,
        iou,
        box1_portion_covered,
        box2_portion_covered,
    ) = (0, 0, 0, 0, 0, 0, 0)

    x_axis_overlap = get_x_axis_overlap(box1, box2)
    if 0 == x_axis_overlap:
        return (
            x_axis_overlap,
            y_axis_overlap,
            intersection_area,
            union_area,
            iou,
            box1_portion_covered,
            box2_portion_covered,
        )

    y_axis_overlap = get_y_axis_overlap(box1, box2)
    if 0 == y_axis_overlap:
        return (
            x_axis_overlap,
            y_axis_overlap,
            intersection_area,
            union_area,
            iou,
            box1_portion_covered,
            box2_portion_covered,
        )
    intersection_area = x_axis_overlap * y_axis_overlap
    box1_area = box1["width"] * box1["height"]
    box2_area = box2["width"] * box2["height"]
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area
    box1_portion_covered = intersection_area / box1_area
    box2_portion_covered = intersection_area / box2_area
    return (
        x_axis_overlap,
        y_axis_overlap,
        intersection_area,
        union_area,
        iou,
        box1_portion_covered,
        box2_portion_covered,
    )


def get_intersection_over_union(rect1, rect2):
    # print('----------------')
    # print (rect1)
    # print(rect2)
    union = get_union_area(rect1, rect2)
    if union == 0:
        print("oh no")
        return 0
    return get_intersection_area(rect1, rect2) / union


def get_intersection_area(rect1, rect2):
    x_overlap = max(
        0, min(rect1["right"], rect2["right"]) - max(rect1["left"], rect2["left"])
    )
    y_overlap = max(
        0, min(rect1["bottom"], rect2["bottom"]) - max(rect1["top"], rect2["top"])
    )
    overlap_area = x_overlap * y_overlap
    return overlap_area


def get_union_area(rect1, rect2):
    union_area = (
        rect1["width"] * rect1["height"]
        + rect2["width"] * rect2["height"]
        - get_intersection_area(rect1, rect2)
    )
    return union_area


def get_y_axis_overlap(rect1, rect2):
    y_overlap = max(
        0, min(rect1["bottom"], rect2["bottom"]) - max(rect1["top"], rect2["top"])
    )
    return y_overlap


def get_x_axis_overlap(rect1, rect2):
    x_overlap = max(
        0, min(rect1["right"], rect2["right"]) - max(rect1["left"], rect2["left"])
    )
    return x_overlap


def get_x_axis_union(rect1, rect2):
    x_overlap = max(
        0, max(rect1["right"], rect2["right"]) - min(rect1["left"], rect2["left"])
    )
    return x_overlap


def get_y_axis_union(rect1, rect2):
    y_overlap = max(
        0, max(rect1["bottom"], rect2["bottom"]) - min(rect1["top"], rect2["top"])
    )
    return y_overlap


def match_gt_2_model(gt_words: list, model_words: list):
    convert_locations_to_float(model_words)
    convert_locations_to_float(gt_words)

    # 1. For each gt_box, find the model_boxes that intersect with it
    gt2model_boxes = {}
    for gt_box in gt_words:
        intersections = []
        for model_box in model_words:
            (
                x_axis_overlap,
                y_axis_overlap,
                intersection_area,
                union_area,
                iou,
                gt_box_portion_covered,
                model_box_portion_covered,
            ) = info_for_boxes(gt_box["location"], model_box["location"])
            if intersection_area > 0:
                intersections.append(
                    (
                        model_box,
                        SimpleNamespace(
                            x_axis_overlap=x_axis_overlap,
                            y_axis_overlap=y_axis_overlap,
                            intersection_area=intersection_area,
                            union_area=union_area,
                            iou=iou,
                            gt_box_portion_covered=gt_box_portion_covered,
                            model_box_portion_covered=model_box_portion_covered,
                        ),
                    )
                )
        box_key = box2key(gt_box["location"])
        if len(intersections) > 1:
            intersections = sorted(intersections, key=lambda x: x[1].intersection_area)
        gt2model_boxes[box_key] = intersections

    # 2. For each model_box, find the gt_boxes that intersect with it
    model2gt_boxes = {}
    for model_box in model_words:
        if model_box["word"] == "containers":
            print("")
        intersections = []
        for gt_box in gt_words:
            (
                x_axis_overlap,
                y_axis_overlap,
                intersection_area,
                union_area,
                iou,
                gt_box_portion_covered,
                model_box_portion_covered,
            ) = info_for_boxes(gt_box["location"], model_box["location"])
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
                            gt_box_portion_covered=gt_box_portion_covered,
                            model_box_portion_covered=model_box_portion_covered,
                        ),
                    )
                )
        box_key = box2key(model_box["location"])
        intersections = sorted(intersections, key=lambda x: x[1].intersection_area)
        model2gt_boxes[box_key] = intersections
    return gt2model_boxes, model2gt_boxes


def box2key(box):
    return (box["top"], box["left"], box["right"], box["bottom"])
