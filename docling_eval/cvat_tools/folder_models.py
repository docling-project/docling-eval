"""Data models for representing CVAT folder structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from docling_eval.datamodels.cvat_types import AnnotationOverview


@dataclass
class CVATPageInfo:
    """Information about a single page in a CVAT task folder."""

    image_filename: str
    doc_hash: str
    page_number: int
    task_id: int
    xml_path: Path
    image_path: Path


@dataclass
class CVATDocument:
    """Represents a complete logical document reconstructed from CVAT pages."""

    doc_hash: str
    doc_name: str
    bin_file: Path
    pages: List[CVATPageInfo] = field(default_factory=list)
    mime_type: str = ""


@dataclass
class CVATFolderStructure:
    """Top-level representation of a CVAT export folder."""

    folder_path: Path
    overview: AnnotationOverview
    documents: Dict[str, CVATDocument]
    xml_pattern: str
    tasks_dir: Path
