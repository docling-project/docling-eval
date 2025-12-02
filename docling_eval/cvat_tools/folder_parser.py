"""Utilities for parsing CVAT folder exports."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from docling_eval.datamodels.cvat_types import AnnotationOverview

from .folder_models import CVATDocument, CVATFolderStructure, CVATPageInfo


def parse_cvat_folder(
    folder_path: Path,
    xml_pattern: str = "task_{xx}_set_A",
    *,
    tasks_root: Optional[Path] = None,
) -> CVATFolderStructure:
    """Parse a CVAT export folder and build document groupings."""
    overview_path = folder_path / "cvat_overview.json"
    if not overview_path.exists():
        raise FileNotFoundError(f"cvat_overview.json not found in {folder_path}")

    overview = AnnotationOverview.load_from_json(overview_path)

    tasks_dir = _resolve_tasks_dir(folder_path, tasks_root)

    xml_files = find_xml_files_by_pattern(tasks_dir, xml_pattern)
    if not xml_files:
        raise FileNotFoundError(
            f"No XML files matching pattern '{xml_pattern}' found in {tasks_dir}"
        )

    all_pages: List[CVATPageInfo] = []
    for xml_path in xml_files:
        all_pages.extend(
            parse_pages_from_xml(
                xml_path=xml_path,
                assets_root=folder_path,
                tasks_dir=tasks_dir,
            )
        )

    documents = group_pages_by_document(all_pages, overview, folder_path)

    return CVATFolderStructure(
        folder_path=folder_path,
        overview=overview,
        documents=documents,
        xml_pattern=xml_pattern,
        tasks_dir=tasks_dir,
    )


def find_xml_files_by_pattern(tasks_dir: Path, pattern: str) -> List[Path]:
    """Locate XML files inside ``tasks_dir`` matching the provided pattern."""
    if not tasks_dir.exists():
        raise FileNotFoundError(f"cvat_tasks directory not found in {tasks_dir}")

    # Convert placeholder pattern to regex (e.g. task_{xx}_set_A -> task_(\d+)_set_A)
    escaped = re.escape(pattern)
    regex_pattern = escaped.replace("\\{xx\\}", r"(\d+)") + r"\.xml"
    regex = re.compile(f"^{regex_pattern}$")

    matching = [
        xml_file for xml_file in tasks_dir.glob("*.xml") if regex.match(xml_file.name)
    ]
    return sorted(matching)


def parse_pages_from_xml(
    xml_path: Path,
    *,
    assets_root: Path,
    tasks_dir: Path,
) -> List[CVATPageInfo]:
    """Extract page information from a CVAT task XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    task_id = extract_task_id_from_xml_path(xml_path)
    pages: List[CVATPageInfo] = []

    for image_el in root.findall(".//image"):
        image_name = image_el.attrib["name"]

        # Extract hash and page number from filename
        # Expected format: doc_{hash}_page_{number}.{ext}
        doc_hash = extract_doc_hash_from_filename(image_name)
        page_number = extract_page_number_from_filename(image_name)

        # Find the actual file path
        # Prefer cvat_tasks/ over page_imgs/ to eliminate redundancy
        # (images are identical, cvat_tasks/ is always created for CVAT upload)
        image_path: Optional[Path] = None

        # First, check cvat_tasks/ directories
        task_dir_candidates = [
            tasks_dir / f"task_{task_id:02d}",
            assets_root / "cvat_tasks" / f"task_{task_id:02d}",
            xml_path.parent,
        ]
        for candidate_dir in task_dir_candidates:
            candidate_file = candidate_dir / image_name
            if candidate_file.exists():
                image_path = candidate_file
                break

        # Fallback to page_imgs/ for backward compatibility with existing datasets
        if image_path is None:
            page_imgs_path = assets_root / "page_imgs" / image_name
            if page_imgs_path.exists():
                image_path = page_imgs_path

        if image_path is None:
            raise FileNotFoundError(f"Image file not found for {image_name}")

        pages.append(
            CVATPageInfo(
                image_filename=image_name,
                doc_hash=doc_hash,
                page_number=page_number,
                task_id=task_id,
                xml_path=xml_path,
                image_path=image_path,
            )
        )

    return pages


def extract_doc_hash_from_filename(filename: str) -> str:
    """Return the document hash encoded in CVAT page filenames."""
    match = re.match(r"doc_([a-f0-9]+)_page_\d+\.[^.]+", filename)
    if not match:
        raise ValueError(f"Invalid CVAT page filename format: {filename}")
    return match.group(1)


def extract_page_number_from_filename(filename: str) -> int:
    """Return the page index encoded in CVAT page filenames."""
    match = re.match(r"doc_[a-f0-9]+_page_(\d+)\.[^.]+", filename)
    if not match:
        raise ValueError(f"Invalid CVAT page filename format: {filename}")
    return int(match.group(1))


def extract_task_id_from_xml_path(xml_path: Path) -> int:
    """Extract the integer task identifier from a CVAT XML filename."""
    match = re.search(r"task_(\d+)", xml_path.stem)
    if not match:
        raise ValueError(f"Invalid XML filename pattern: {xml_path.name}")
    return int(match.group(1))


def group_pages_by_document(
    pages: Iterable[CVATPageInfo],
    overview: AnnotationOverview,
    folder_path: Path,
) -> Dict[str, CVATDocument]:
    """Group page entries by their document hash and enrich with overview metadata."""
    doc_lookup = {doc.doc_hash: doc for doc in overview.doc_annotations}

    grouped: Dict[str, List[CVATPageInfo]] = {}
    for page in pages:
        grouped.setdefault(page.doc_hash, []).append(page)

    documents: Dict[str, CVATDocument] = {}
    for doc_hash, page_infos in grouped.items():
        doc_ann = doc_lookup.get(doc_hash)
        if doc_ann is None:
            raise ValueError(
                f"Document hash {doc_hash} not present in cvat_overview.json"
            )

        page_infos.sort(key=lambda p: p.page_number)

        bin_path = _resolve_path(folder_path, Path(doc_ann.bin_file))

        documents[doc_hash] = CVATDocument(
            doc_hash=doc_hash,
            doc_name=doc_ann.doc_name,
            bin_file=bin_path,
            pages=page_infos,
            mime_type=doc_ann.mime_type,
        )

    return documents


def _resolve_tasks_dir(folder_path: Path, tasks_root: Optional[Path]) -> Path:
    """Resolve the directory that contains CVAT task annotation XML files."""

    candidates: List[Path] = []

    def add_candidate(path: Path) -> None:
        resolved = path
        if resolved not in candidates:
            candidates.append(resolved)

    if tasks_root is not None:
        add_candidate(tasks_root / "cvat_tasks")
        add_candidate(tasks_root)

    add_candidate(folder_path / "cvat_tasks")

    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        if candidate.name == "cvat_tasks":
            return candidate
        nested = candidate / "cvat_tasks"
        if nested.exists() and nested.is_dir():
            return nested
    raise FileNotFoundError(
        "Unable to locate a 'cvat_tasks' directory using the provided paths"
    )


def _resolve_path(base: Path, candidate: Path) -> Path:
    """Resolve a potentially relative path against ``base`` when possible."""
    candidate = Path(candidate)
    if not str(candidate):
        return candidate

    if candidate.is_absolute():
        if candidate.exists():
            return candidate

        anchor_name = base.name
        if anchor_name in candidate.parts:
            anchor_index = candidate.parts.index(anchor_name)
            relative_parts = candidate.parts[anchor_index + 1 :]
            rebased = base.joinpath(*relative_parts) if relative_parts else base
            if rebased.exists():
                return rebased
            return rebased

        return candidate

    if candidate.exists():
        return candidate

    resolved = base / candidate
    if resolved.exists():
        return resolved

    alt = base / candidate.name
    if alt.exists():
        return alt

    return resolved
