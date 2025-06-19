from enum import Enum
from typing import Any, List, Optional, Tuple

from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.document import ContentLayer
from docling_core.types.doc.labels import DocItemLabel
from pydantic import BaseModel, Field


class ValidationSeverity(str, Enum):
    """Severity levels for validation errors."""

    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


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
    severity: ValidationSeverity = ValidationSeverity.ERROR
    element_id: Optional[int] = None
    path_id: Optional[int] = None
    path_ids: Optional[List[int]] = None
    point_index: Optional[int] = None
    point_coords: Optional[Tuple[float, float]] = None


class CVATValidationReport(BaseModel):
    """Validation report for a single sample."""

    sample_name: str
    errors: List[CVATValidationError]

    def has_fatal_errors(self) -> bool:
        """Check if there are any fatal validation errors."""
        return any(e.severity == ValidationSeverity.FATAL for e in self.errors)

    def has_errors(self) -> bool:
        """Check if there are any validation errors."""
        return any(e.severity == ValidationSeverity.ERROR for e in self.errors)

    def has_warnings(self) -> bool:
        """Check if there are any warning validation errors."""
        return any(e.severity == ValidationSeverity.WARNING for e in self.errors)


class CVATValidationRunReport(BaseModel):
    """Validation report for a run of multiple samples."""

    samples: List[CVATValidationReport]


class CVATImageInfo(BaseModel):
    """Information about an image in CVAT annotation."""

    width: float
    height: float
    name: str
