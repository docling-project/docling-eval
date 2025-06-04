from typing import Any, List, Optional

from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.document import ContentLayer
from docling_core.types.doc.labels import DocItemLabel
from pydantic import BaseModel, Field


class CVATElement(BaseModel):
    """A rectangle element (box) in CVAT annotation, using BoundingBox from docling_core."""

    id: int
    label: DocItemLabel
    bbox: BoundingBox
    content_layer: ContentLayer
    type: Optional[str] = None
    level: Optional[int] = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class CVATAnnotationPath(BaseModel):
    """A polyline path in CVAT annotation (reading-order, merge, group, etc)."""

    id: int
    label: str
    points: list[tuple[float, float]]
    level: Optional[int] = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class CVATValidationError(BaseModel):
    """Validation error for reporting issues in annotation."""

    error_type: str
    message: str
    element_id: Optional[int] = None
    path_id: Optional[int] = None


class CVATValidationReport(BaseModel):
    """Validation report for a single sample."""

    sample_name: str
    errors: List[CVATValidationError]


class CVATValidationRunReport(BaseModel):
    """Validation report for a run of multiple samples."""

    samples: List[CVATValidationReport]


class CVATImageInfo(BaseModel):
    """Information about an image in CVAT annotation."""

    width: float
    height: float
    name: str
