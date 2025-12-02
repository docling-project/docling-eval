from enum import Enum
from typing import Any, List, Optional, Tuple, Union

from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.document import ContentLayer
from docling_core.types.doc.labels import DocItemLabel, GraphCellLabel
from pydantic import BaseModel, Field


class ValidationSeverity(str, Enum):
    """Severity levels for validation errors."""

    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


class TableStructLabel(str, Enum):
    TABLE_ROW = "table_row"
    TABLE_COLUMN = "table_column"
    TABLE_MERGED_CELL = "table_merged_cell"
    COL_HEADER = "col_header"
    ROW_HEADER = "row_header"
    ROW_SECTION = "table_row_section"
    TABLE_FILLABLE_CELLS = "fillable_cells"
    BODY = "body"


class CVATElement(BaseModel):
    """A rectangle element (box) in CVAT annotation, using BoundingBox from docling_core."""

    id: int
    label: Union[DocItemLabel, GraphCellLabel, TableStructLabel]
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


class CVATValidationStatistics(BaseModel):
    """Global statistics for a validation run."""

    total_errors: int = Field(
        description="Total number of validation errors across all samples"
    )
    total_fatal: int = Field(description="Total number of FATAL errors")
    total_error: int = Field(description="Total number of ERROR errors")
    total_warning: int = Field(description="Total number of WARNING errors")
    samples_with_any_error: int = Field(
        description="Number of samples affected by at least one validation error"
    )
    samples_with_fatal: int = Field(
        description="Number of samples affected by at least one FATAL error"
    )
    samples_with_error: int = Field(
        description="Number of samples affected by at least one ERROR error"
    )
    samples_with_warning: int = Field(
        description="Number of samples affected by at least one WARNING error"
    )


class CVATValidationRunReport(BaseModel):
    """Validation report for a run of multiple samples."""

    samples: List[CVATValidationReport]
    statistics: CVATValidationStatistics

    @staticmethod
    def compute_statistics(
        samples: List[CVATValidationReport],
    ) -> CVATValidationStatistics:
        """Compute global statistics from a list of validation reports."""
        total_errors = 0
        total_fatal = 0
        total_error = 0
        total_warning = 0
        samples_with_any_error = 0
        samples_with_fatal = 0
        samples_with_error = 0
        samples_with_warning = 0

        for sample_report in samples:
            # Count total errors by severity
            sample_fatal = sum(
                1
                for e in sample_report.errors
                if e.severity == ValidationSeverity.FATAL
            )
            sample_error = sum(
                1
                for e in sample_report.errors
                if e.severity == ValidationSeverity.ERROR
            )
            sample_warning = sum(
                1
                for e in sample_report.errors
                if e.severity == ValidationSeverity.WARNING
            )

            total_errors += len(sample_report.errors)
            total_fatal += sample_fatal
            total_error += sample_error
            total_warning += sample_warning

            # Count samples with at least one error of each type
            if len(sample_report.errors) > 0:
                samples_with_any_error += 1
            if sample_fatal > 0:
                samples_with_fatal += 1
            if sample_error > 0:
                samples_with_error += 1
            if sample_warning > 0:
                samples_with_warning += 1

        return CVATValidationStatistics(
            total_errors=total_errors,
            total_fatal=total_fatal,
            total_error=total_error,
            total_warning=total_warning,
            samples_with_any_error=samples_with_any_error,
            samples_with_fatal=samples_with_fatal,
            samples_with_error=samples_with_error,
            samples_with_warning=samples_with_warning,
        )


class CVATImageInfo(BaseModel):
    """Information about an image in CVAT annotation."""

    width: float
    height: float
    name: str
