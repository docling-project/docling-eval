"""Geometry helpers for CVAT tooling."""

from __future__ import annotations

from typing import Iterable, Iterator, Optional, Protocol, Sequence, TypeVar

from docling_core.types.doc.base import BoundingBox, CoordOrigin


class HasBoundingBox(Protocol):
    """Protocol for objects exposing a bounding box."""

    bbox: BoundingBox


TElement = TypeVar("TElement", bound=HasBoundingBox)


def bbox_iou(a: BoundingBox, b: BoundingBox, *, eps: float = 1.0e-6) -> float:
    """Return the intersection over union between two bounding boxes."""
    return a.intersection_over_union(b, eps=eps)


def bbox_fraction_inside(
    inner: BoundingBox, outer: BoundingBox, *, eps: float = 1.0e-9
) -> float:
    """Return the fraction of ``inner`` area that lies inside ``outer``."""
    area = inner.area()
    if area <= eps:
        return 0.0
    intersection = inner.intersection_area_with(outer)
    return intersection / max(area, eps)


def bbox_contains(
    inner: BoundingBox, outer: BoundingBox, *, threshold: float, eps: float = 1.0e-9
) -> bool:
    """Return ``True`` when ``inner`` is contained in ``outer`` above ``threshold``."""
    return bbox_fraction_inside(inner, outer, eps=eps) >= threshold


def bbox_intersection(a: BoundingBox, b: BoundingBox) -> Optional[BoundingBox]:
    """Return the intersection of two bounding boxes or ``None`` when disjoint."""
    if a.coord_origin != b.coord_origin:
        raise ValueError("BoundingBoxes have different CoordOrigin")

    left = max(a.l, b.l)
    right = min(a.r, b.r)

    if a.coord_origin == CoordOrigin.TOPLEFT:
        top = max(a.t, b.t)
        bottom = min(a.b, b.b)
        if right <= left or bottom <= top:
            return None
        return BoundingBox(
            l=left, t=top, r=right, b=bottom, coord_origin=a.coord_origin
        )

    top = min(a.t, b.t)
    bottom = max(a.b, b.b)
    if right <= left or top <= bottom:
        return None
    return BoundingBox(l=left, t=top, r=right, b=bottom, coord_origin=a.coord_origin)


def dedupe_items_by_bbox(
    elements: Sequence[TElement],
    *,
    iou_threshold: float = 0.9,
) -> list[TElement]:
    """Return elements whose bounding boxes are unique within ``iou_threshold``."""
    deduped: list[TElement] = []
    for element in elements:
        if all(bbox_iou(element.bbox, kept.bbox) < iou_threshold for kept in deduped):
            deduped.append(element)
    return deduped


def iter_unique_by_bbox(
    elements: Iterable[TElement],
    *,
    iou_threshold: float = 0.9,
) -> Iterator[TElement]:
    """Yield unique elements lazily based on bounding-box IoU."""
    seen: list[TElement] = []
    for element in elements:
        if all(bbox_iou(element.bbox, kept.bbox) < iou_threshold for kept in seen):
            seen.append(element)
            yield element
