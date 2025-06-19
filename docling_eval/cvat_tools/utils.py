"""Utility functions and constants shared across CVAT tools.

This module contains common functionality used by multiple modules to avoid code duplication.
"""

from typing import List, Optional

from .models import CVATElement

# Constants
DEFAULT_PROXIMITY_THRESHOLD = 5.0
"""Default threshold for point-to-element proximity mapping."""


def find_elements_containing_point(
    point: tuple[float, float],
    elements: List[CVATElement],
    proximity_thresh: float = DEFAULT_PROXIMITY_THRESHOLD,
) -> List[CVATElement]:
    """Find all elements whose bounding box contains the given point.

    Args:
        point: (x, y) coordinates of the point
        elements: List of elements to search
        proximity_thresh: Distance threshold for proximity matching

    Returns:
        List of elements containing the point, ordered by area (smallest first)
    """
    if not elements:
        return []

    x, y = point
    candidates = []

    for el in elements:
        bbox = el.bbox
        if (
            bbox.l - proximity_thresh <= x <= bbox.r + proximity_thresh
            and bbox.t - proximity_thresh <= y <= bbox.b + proximity_thresh
        ):
            candidates.append(el)

    # Return sorted by area (smallest first - deepest elements)
    return sorted(candidates, key=lambda e: e.bbox.area())


def get_deepest_element_at_point(
    point: tuple[float, float],
    elements: List[CVATElement],
    proximity_thresh: float = DEFAULT_PROXIMITY_THRESHOLD,
) -> Optional[CVATElement]:
    """Get the deepest (smallest area) element containing the given point.

    Args:
        point: (x, y) coordinates of the point
        elements: List of elements to search
        proximity_thresh: Distance threshold for proximity matching

    Returns:
        The deepest element containing the point, or None if no element found
    """
    candidates = find_elements_containing_point(point, elements, proximity_thresh)
    return candidates[0] if candidates else None


def validate_element_types(
    elements: List[CVATElement],
    element_ids: List[int],
    expected_labels: List[str],
    context: str = "elements",
) -> List[str]:
    """Validate that elements have expected labels.

    Args:
        elements: List of all elements
        element_ids: IDs of elements to validate
        expected_labels: List of acceptable labels
        context: Context string for error messages

    Returns:
        List of validation error messages
    """
    errors = []
    id_to_element = {el.id: el for el in elements}

    for element_id in element_ids:
        element = id_to_element.get(element_id)
        if not element:
            errors.append(f"{context}: Element {element_id} not found")
        elif element.label not in expected_labels:
            errors.append(
                f"{context}: Element {element_id} has label '{element.label}', "
                f"expected one of {expected_labels}"
            )

    return errors


def is_container_element(element: CVATElement) -> bool:
    """Check if an element is a container type.

    Args:
        element: Element to check

    Returns:
        True if the element is a container type
    """
    return element.label in ["table", "picture", "form", "code"]


def is_caption_element(element: CVATElement) -> bool:
    """Check if an element is a caption.

    Args:
        element: Element to check

    Returns:
        True if the element is a caption
    """
    return element.label == "caption"


def is_footnote_element(element: CVATElement) -> bool:
    """Check if an element is a footnote.

    Args:
        element: Element to check

    Returns:
        True if the element is a footnote
    """
    return element.label == "footnote"
