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


def get_x_axis_overlap(rect1, rect2):
    x_overlap = max(
        0, min(rect1["right"], rect2["right"]) - max(rect1["left"], rect2["left"])
    )
    return x_overlap


def get_y_axis_overlap(rect1, rect2):
    y_overlap = max(
        0, min(rect1["bottom"], rect2["bottom"]) - max(rect1["top"], rect2["top"])
    )
    return y_overlap


def get_intersection_area(rect1, rect2):
    x_overlap = get_x_axis_overlap(rect1, rect2)
    y_overlap = get_y_axis_overlap(rect1, rect2)
    overlap_area = x_overlap * y_overlap
    return overlap_area


def get_union_area(rect1, rect2):
    union_area = (
        rect1["width"] * rect1["height"]
        + rect2["width"] * rect2["height"]
        - get_intersection_area(rect1, rect2)
    )
    return union_area


def get_x_axis_union(rect1, rect2):
    x_union = max(
        0, max(rect1["right"], rect2["right"]) - min(rect1["left"], rect2["left"])
    )
    return x_union


def get_y_axis_union(rect1, rect2):
    y_union = max(
        0, max(rect1["bottom"], rect2["bottom"]) - min(rect1["top"], rect2["top"])
    )
    return y_union


def info_for_boxes(box1, box2):
    x_axis_overlap = get_x_axis_overlap(box1, box2)
    if x_axis_overlap == 0:
        return 0, 0, 0, 0, 0, 0, 0

    y_axis_overlap = get_y_axis_overlap(box1, box2)
    if y_axis_overlap == 0:
        return x_axis_overlap, 0, 0, 0, 0, 0, 0

    intersection_area = x_axis_overlap * y_axis_overlap
    box1_area = box1["width"] * box1["height"]
    box2_area = box2["width"] * box2["height"]
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    box1_portion_covered = intersection_area / box1_area if box1_area > 0 else 0
    box2_portion_covered = intersection_area / box2_area if box2_area > 0 else 0
    return (
        x_axis_overlap,
        y_axis_overlap,
        intersection_area,
        union_area,
        iou,
        box1_portion_covered,
        box2_portion_covered,
    )


def info_for_boxes_extended(box1, box2):
    x_axis_overlap = get_x_axis_overlap(box1, box2)
    y_axis_overlap = get_y_axis_overlap(box1, box2)
    intersection_area = x_axis_overlap * y_axis_overlap
    box1_area = box1["width"] * box1["height"]
    box2_area = box2["width"] * box2["height"]
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0

    x_axis_union = get_x_axis_union(box1, box2)
    y_axis_union = get_y_axis_union(box1, box2)
    x_axis_iou = x_axis_overlap / x_axis_union if x_axis_union > 0 else 0
    y_axis_iou = y_axis_overlap / y_axis_union if y_axis_union > 0 else 0
    return (
        x_axis_overlap,
        y_axis_overlap,
        intersection_area,
        union_area,
        iou,
        x_axis_iou,
        y_axis_iou,
    )


def box_to_key(box):
    return (box["top"], box["left"], box["right"], box["bottom"])


def is_horizontal(word):
    h = word["location"]["height"]
    w = word["location"]["width"]
    if h > int(2 * w) and len(word["word"]) > 1:
        return False
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
