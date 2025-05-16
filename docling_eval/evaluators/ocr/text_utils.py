from collections import namedtuple

import edit_distance

EPS = 1.0e-6
edit_distance_chars_map: dict[str, str] = {}
BoxesTypes = namedtuple("BoxesTypes", ["zero_iou", "low_iou", "ambiguous_match"])


def replace_chars_by_map(text, char_map):
    processed_text = "".join(
        char if char not in char_map else char_map[char] for char in text
    )
    return processed_text


def calculate_edit_distance(hypothesis_text, reference_text, string_normalize_map={}):
    hypothesis_text = replace_chars_by_map(hypothesis_text, string_normalize_map)
    reference_text = replace_chars_by_map(reference_text, string_normalize_map)
    if len(edit_distance_chars_map):
        hypothesis_text = "".join(
            (
                char
                if char not in edit_distance_chars_map
                else edit_distance_chars_map[char]
            )
            for char in hypothesis_text
        )
        reference_text = "".join(
            (
                char
                if char not in edit_distance_chars_map
                else edit_distance_chars_map[char]
            )
            for char in reference_text
        )
    sm = edit_distance.SequenceMatcher(hypothesis_text, reference_text)
    return sm.distance()
