"""Geometry helpers for CVAT tooling."""

from __future__ import annotations

import math
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


def bbox_enclosing_rotated_rect(
    unrotated_bbox: BoundingBox, *, rotation_deg: float
) -> BoundingBox:
    """Return the smallest axis-aligned BoundingBox enclosing a rotated rectangle.

    CVAT represents rotated boxes as an unrotated rectangle (xtl/ytl/xbr/ybr) plus a
    `rotation` attribute. The rotation is applied around the center of that rectangle.

    This helper returns an axis-aligned BoundingBox that encloses the rotated rectangle.
    It preserves the input bbox coord origin and requires TOPLEFT coordinates (CVAT).
    """
    if unrotated_bbox.coord_origin != CoordOrigin.TOPLEFT:
        raise ValueError(
            "bbox_enclosing_rotated_rect currently expects CoordOrigin.TOPLEFT"
        )

    normalized = rotation_deg % 360.0
    if normalized == 0.0:
        return unrotated_bbox

    theta = math.radians(normalized)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    cx = (unrotated_bbox.l + unrotated_bbox.r) / 2.0
    cy = (unrotated_bbox.t + unrotated_bbox.b) / 2.0

    corners = (
        (unrotated_bbox.l, unrotated_bbox.t),
        (unrotated_bbox.r, unrotated_bbox.t),
        (unrotated_bbox.r, unrotated_bbox.b),
        (unrotated_bbox.l, unrotated_bbox.b),
    )

    xs: list[float] = []
    ys: list[float] = []
    for x, y in corners:
        dx = x - cx
        dy = y - cy
        rx = cx + (dx * cos_t) - (dy * sin_t)
        ry = cy + (dx * sin_t) + (dy * cos_t)
        xs.append(rx)
        ys.append(ry)

    return BoundingBox(
        l=min(xs),
        t=min(ys),
        r=max(xs),
        b=max(ys),
        coord_origin=unrotated_bbox.coord_origin,
    )
