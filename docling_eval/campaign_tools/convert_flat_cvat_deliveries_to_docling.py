#!/usr/bin/env python3
from __future__ import annotations

"""
Convert flat CVAT XML deliveries into DoclingDocument JSON + visualization HTML.

Expected input layout:
1) A flat directory of batch XML files, for example:
   batch_01_forms_datasheets_accounting_set.xml
2) A data root containing matching batch folders, each with an images/ directory:
   <data_root>/batch_01_forms_datasheets_accounting/images/*.png
3) In document mode, a matching per-batch PDF can be used as fallback:
   <data_root>/batch_01_forms_datasheets_accounting/batch_01_forms_datasheets_accounting.pdf
4) Alternatively, per-batch folders containing annotations.xml:
   <xml_root>/<batch_name>/annotations.xml

The tool normalizes XML names by stripping the trailing "_set" suffix (including
duplicate variants like "_set 2"), then resolves the corresponding data folder.
"""

import argparse
import json
import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Final, Iterable, List, Literal, Sequence, Tuple, cast

from pydantic import BaseModel, Field

from docling_core.types import DoclingDocument

# CVAT tools are optional - provided by docling-cvat-tools
try:
    from docling_cvat_tools.cvat_tools.cvat_to_docling import (
        CVATToDoclingConverter,
        convert_cvat_to_docling,
        load_document_pages,
    )
    from docling_cvat_tools.cvat_tools.parser import (
        ParsedCVATFile,
        get_all_images_from_cvat_xml,
        parse_cvat_file,
    )
    from docling_cvat_tools.cvat_tools.validator import validate_cvat_sample
    from docling_cvat_tools.visualisation.visualisations import (
        save_single_document_html,
    )
except ImportError as e:
    raise ImportError(
        "Flat CVAT conversion requires docling-cvat-tools. "
        "Install with: pip install docling-eval[cvat_tools]"
    ) from e

_LOGGER = logging.getLogger(__name__)

_BATCH_SET_SUFFIX_PATTERN = re.compile(r"_set(?:\s+\d+)?$")
_FLAT_XML_PATTERN = "batch_*_set*.xml"
_DELIVERY_XML_NAME = "annotations.xml"
_FOLDER_MODE_BATCH: Final["FolderMode"] = "batch"
_FOLDER_MODE_DOCUMENT: Final["FolderMode"] = "document"
FolderMode = Literal["batch", "document"]
_DEFAULT_STORAGE_SCALE = 2.0
_IMAGE_PAGE_SUFFIX_PATTERN = re.compile(r"(?:-|_)(\d+)(?:\.[^.]+)?$")


def _normalize_batch_name(xml_stem: str) -> str:
    return _BATCH_SET_SUFFIX_PATTERN.sub("", xml_stem)


def _choose_preferred_xml(candidates: Sequence[Path]) -> Path:
    """Prefer canonical '*_set.xml' names over duplicate variants."""
    sorted_candidates = sorted(candidates)
    canonical = [path for path in sorted_candidates if path.stem.endswith("_set")]
    if canonical:
        return canonical[0]
    return sorted_candidates[0]


def _discover_batch_xmls(
    xml_root: Path,
) -> tuple[Dict[str, Path], Dict[str, List[Path]], int]:
    grouped: Dict[str, List[Path]] = {}
    discovered_xml_count = 0

    for xml_path in sorted(xml_root.glob(_FLAT_XML_PATTERN)):
        batch_name = _normalize_batch_name(xml_path.stem)
        grouped.setdefault(batch_name, []).append(xml_path)
        discovered_xml_count += 1

    for candidate_dir in sorted(p for p in xml_root.iterdir() if p.is_dir()):
        xml_path = candidate_dir / _DELIVERY_XML_NAME
        if not xml_path.exists():
            continue

        grouped.setdefault(candidate_dir.name, []).append(xml_path)
        discovered_xml_count += 1

    selected: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}
    for batch_name, candidates in grouped.items():
        selected[batch_name] = _choose_preferred_xml(candidates)
        if len(candidates) > 1:
            duplicates[batch_name] = sorted(candidates)

    return selected, duplicates, discovered_xml_count


@dataclass(frozen=True)
class ConversionTask:
    batch_name: str
    xml_path: Path
    image_name: str
    image_path: Path
    json_path: Path
    html_path: Path
    cvat_input_scale: float


@dataclass(frozen=True)
class HtmlRenderTask:
    image_name: str
    json_path: Path
    html_path: Path


@dataclass(frozen=True)
class BatchDataPaths:
    images_dir: Path
    document_pdf_path: Path | None


class ConversionError(BaseModel):
    image_name: str
    reason: str


class BatchReport(BaseModel):
    batch_name: str
    xml_path: Path
    images_dir: Path
    output_json_dir: Path
    output_html_dir: Path
    xml_image_count: int = 0
    missing_image_count: int = 0
    skipped_existing_count: int = 0
    converted_count: int = 0
    failed_count: int = 0
    missing_images: List[str] = Field(default_factory=list)
    failures: List[ConversionError] = Field(default_factory=list)


class RunReport(BaseModel):
    xml_root: Path
    data_root: Path
    output_root: Path
    folder_mode: FolderMode = _FOLDER_MODE_BATCH
    dry_run: bool
    overwrite_existing: bool
    workers: int
    cvat_input_scale: float
    started_at: str
    finished_at: str | None = None
    discovered_xml_count: int = 0
    selected_batch_count: int = 0
    duplicate_batch_count: int = 0
    duplicate_details: Dict[str, List[Path]] = Field(default_factory=dict)
    batches: List[BatchReport] = Field(default_factory=list)

    @property
    def total_converted(self) -> int:
        return sum(batch.converted_count for batch in self.batches)

    @property
    def total_failed(self) -> int:
        return sum(batch.failed_count for batch in self.batches)

    @property
    def total_missing_images(self) -> int:
        return sum(batch.missing_image_count for batch in self.batches)

    @property
    def total_skipped_existing(self) -> int:
        return sum(batch.skipped_existing_count for batch in self.batches)


class TaskResult(BaseModel):
    image_name: str
    success: bool
    reason: str | None = None


def _layout_visualization_path(base_html_path: Path) -> Path:
    return base_html_path.with_name(
        f"{base_html_path.stem}_layout{base_html_path.suffix}"
    )


def _convert_single_task(task: ConversionTask) -> TaskResult:
    try:
        doc = convert_cvat_to_docling(
            xml_path=task.xml_path,
            input_path=task.image_path,
            image_identifier=task.image_name,
            cvat_input_scale=task.cvat_input_scale,
        )
        if doc is None:
            return TaskResult(
                image_name=task.image_name,
                success=False,
                reason="Conversion returned None",
            )

        task.json_path.parent.mkdir(parents=True, exist_ok=True)
        task.html_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save_as_json(task.json_path)
        save_single_document_html(task.html_path, doc, draw_reading_order=True)
        return TaskResult(image_name=task.image_name, success=True)
    except Exception as exc:  # noqa: BLE001
        return TaskResult(
            image_name=task.image_name,
            success=False,
            reason=str(exc),
        )


def _render_single_html_task(task: HtmlRenderTask) -> TaskResult:
    try:
        doc = DoclingDocument.load_from_json(task.json_path)
        task.html_path.parent.mkdir(parents=True, exist_ok=True)
        save_single_document_html(task.html_path, doc, draw_reading_order=True)
        return TaskResult(image_name=task.image_name, success=True)
    except Exception as exc:  # noqa: BLE001
        return TaskResult(
            image_name=task.image_name,
            success=False,
            reason=str(exc),
        )


def _get_sorted_xml_image_names(xml_path: Path) -> List[str]:
    return sorted(get_all_images_from_cvat_xml(xml_path))


def _build_multipage_doc(
    batch_name: str, page_docs: Sequence[Tuple[str, DoclingDocument]]
) -> DoclingDocument:
    if not page_docs:
        raise ValueError(
            f"No page documents available to build multipage document for {batch_name}."
        )

    merged_doc = DoclingDocument.concatenate([page_doc for _, page_doc in page_docs])
    merged_doc.name = batch_name
    return merged_doc


def _extract_page_number_from_image_name(image_name: str) -> int | None:
    match = _IMAGE_PAGE_SUFFIX_PATTERN.search(image_name)
    if match is None:
        return None
    return int(match.group(1))


def _resolve_xml_page_numbers(parsed_file: ParsedCVATFile) -> Dict[str, int]:
    page_numbers: Dict[str, int] = {}
    for index, image_name in enumerate(parsed_file.image_names, start=1):
        parsed_page = _extract_page_number_from_image_name(image_name)
        page_numbers[image_name] = parsed_page if parsed_page is not None else index
    return page_numbers


def _convert_pdf_page_from_xml_image(
    *,
    xml_path: Path,
    pdf_path: Path,
    image_name: str,
    page_number: int,
    cvat_input_scale: float,
    parsed_file: ParsedCVATFile,
) -> DoclingDocument | None:
    validated_sample = validate_cvat_sample(
        xml_path,
        image_name,
        parsed_file=parsed_file,
    )
    if validated_sample.report.has_fatal_errors():
        _LOGGER.error(
            "Fatal validation errors on sample %s in %s; skipping page conversion.",
            image_name,
            xml_path,
        )
        return None

    load_result = load_document_pages(
        input_path=pdf_path,
        page_numbers=[page_number],
        force_ocr=False,
        ocr_scale=1.0,
        cvat_input_scale=cvat_input_scale,
    )

    seg_page = load_result.segmented_pages.get(page_number)
    page_image = load_result.page_images.get(page_number)
    if seg_page is None or page_image is None:
        _LOGGER.error(
            "Unable to load PDF page %d for %s from %s",
            page_number,
            image_name,
            pdf_path,
        )
        return None

    converter = CVATToDoclingConverter(
        validated_sample.structure,
        segmented_pages={1: seg_page},
        page_images={1: page_image},
        document_filename=pdf_path.name,
        cvat_input_scale=cvat_input_scale,
        storage_scale=_DEFAULT_STORAGE_SCALE,
    )
    return converter.convert()


def _iter_conversion_tasks(
    *,
    batch_name: str,
    xml_path: Path,
    images_dir: Path,
    output_json_dir: Path,
    output_html_dir: Path,
    overwrite_existing: bool,
    batch_report: BatchReport,
    cvat_input_scale: float,
) -> Iterable[ConversionTask]:
    image_names = _get_sorted_xml_image_names(xml_path)
    batch_report.xml_image_count = len(image_names)

    for image_name in image_names:
        image_path = images_dir / image_name
        if not image_path.exists():
            batch_report.missing_image_count += 1
            batch_report.missing_images.append(image_name)
            continue

        image_stem = Path(image_name).stem
        json_path = output_json_dir / f"{image_stem}.json"
        html_path = output_html_dir / f"{image_stem}.html"
        layout_path = _layout_visualization_path(html_path)

        if not overwrite_existing and json_path.exists() and layout_path.exists():
            batch_report.skipped_existing_count += 1
            continue

        yield ConversionTask(
            batch_name=batch_name,
            xml_path=xml_path,
            image_name=image_name,
            image_path=image_path,
            json_path=json_path,
            html_path=html_path,
            cvat_input_scale=cvat_input_scale,
        )


def _iter_html_render_tasks(
    *,
    xml_path: Path,
    output_json_dir: Path,
    output_html_dir: Path,
    overwrite_existing: bool,
    batch_report: BatchReport,
) -> Iterable[HtmlRenderTask]:
    image_names = _get_sorted_xml_image_names(xml_path)
    batch_report.xml_image_count = len(image_names)

    for image_name in image_names:
        image_stem = Path(image_name).stem
        json_path = output_json_dir / f"{image_stem}.json"
        html_path = output_html_dir / f"{image_stem}.html"
        layout_path = _layout_visualization_path(html_path)

        if not json_path.exists():
            batch_report.failed_count += 1
            batch_report.failures.append(
                ConversionError(
                    image_name=image_name,
                    reason=f"Missing JSON: {json_path}",
                )
            )
            continue

        if not overwrite_existing and layout_path.exists():
            batch_report.skipped_existing_count += 1
            continue

        yield HtmlRenderTask(
            image_name=image_name,
            json_path=json_path,
            html_path=html_path,
        )


def _run_batch_tasks(tasks: List[ConversionTask], workers: int) -> List[TaskResult]:
    if not tasks:
        return []

    if workers <= 1:
        return [_convert_single_task(task) for task in tasks]

    results: List[TaskResult] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_convert_single_task, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _run_html_tasks(tasks: List[HtmlRenderTask], workers: int) -> List[TaskResult]:
    if not tasks:
        return []

    if workers <= 1:
        return [_render_single_html_task(task) for task in tasks]

    results: List[TaskResult] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_render_single_html_task, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _resolve_batch_data_paths(data_root: Path, batch_name: str) -> BatchDataPaths:
    batch_root = data_root / batch_name
    images_dir = batch_root / "images"
    pdf_path = batch_root / f"{batch_name}.pdf"
    document_pdf_path = pdf_path if pdf_path.exists() else None
    return BatchDataPaths(
        images_dir=images_dir,
        document_pdf_path=document_pdf_path,
    )


def run_conversion(
    *,
    xml_root: Path,
    data_root: Path,
    output_root: Path,
    folder_mode: FolderMode,
    json_dirname: str,
    html_dirname: str,
    html_only: bool,
    workers: int,
    cvat_input_scale: float,
    dry_run: bool,
    overwrite_existing: bool,
    allow_partial_document: bool,
) -> RunReport:
    selected_xmls, duplicates, discovered_xml_count = _discover_batch_xmls(xml_root)

    report = RunReport(
        xml_root=xml_root,
        data_root=data_root,
        output_root=output_root,
        folder_mode=folder_mode,
        dry_run=dry_run,
        overwrite_existing=overwrite_existing,
        workers=workers,
        cvat_input_scale=cvat_input_scale,
        started_at=datetime.now(timezone.utc).isoformat(),
        discovered_xml_count=discovered_xml_count,
        selected_batch_count=len(selected_xmls),
        duplicate_batch_count=len(duplicates),
        duplicate_details=duplicates,
    )

    if duplicates:
        _LOGGER.warning("Detected duplicate XML batches: %d", len(duplicates))
        for batch_name, candidate_paths in duplicates.items():
            selected = _choose_preferred_xml(candidate_paths)
            _LOGGER.warning(
                "Batch %s has %d XML files; using %s",
                batch_name,
                len(candidate_paths),
                selected.name,
            )

    output_root.mkdir(parents=True, exist_ok=True)

    for batch_name in sorted(selected_xmls.keys()):
        xml_path = selected_xmls[batch_name]
        batch_data_paths = _resolve_batch_data_paths(data_root, batch_name)
        images_dir = batch_data_paths.images_dir
        document_pdf_path = batch_data_paths.document_pdf_path
        expected_document_pdf_path = data_root / batch_name / f"{batch_name}.pdf"
        output_batch_dir = output_root / batch_name
        output_json_dir = output_batch_dir / json_dirname
        output_html_dir = output_batch_dir / html_dirname

        batch_report = BatchReport(
            batch_name=batch_name,
            xml_path=xml_path,
            images_dir=images_dir,
            output_json_dir=output_json_dir,
            output_html_dir=output_html_dir,
        )
        report.batches.append(batch_report)

        if not html_only and not images_dir.exists():
            if folder_mode == _FOLDER_MODE_DOCUMENT and document_pdf_path is not None:
                _LOGGER.info(
                    "Batch %s: images directory not found; using PDF fallback %s",
                    batch_name,
                    document_pdf_path,
                )
            else:
                message = (
                    f"Missing images directory: {images_dir}"
                    if folder_mode == _FOLDER_MODE_BATCH
                    else (
                        f"Missing images directory: {images_dir}; "
                        f"expected PDF fallback at {expected_document_pdf_path}"
                    )
                )
                _LOGGER.error("%s", message)
                batch_report.failures.append(
                    ConversionError(image_name="__batch__", reason=message)
                )
                batch_report.failed_count = 1
                continue

        output_json_dir.mkdir(parents=True, exist_ok=True)
        output_html_dir.mkdir(parents=True, exist_ok=True)

        if folder_mode == _FOLDER_MODE_BATCH:
            conversion_tasks: List[ConversionTask] = []
            html_tasks: List[HtmlRenderTask] = []
            if html_only:
                html_tasks = list(
                    _iter_html_render_tasks(
                        xml_path=xml_path,
                        output_json_dir=output_json_dir,
                        output_html_dir=output_html_dir,
                        overwrite_existing=overwrite_existing,
                        batch_report=batch_report,
                    )
                )
            else:
                conversion_tasks = list(
                    _iter_conversion_tasks(
                        batch_name=batch_name,
                        xml_path=xml_path,
                        images_dir=images_dir,
                        output_json_dir=output_json_dir,
                        output_html_dir=output_html_dir,
                        overwrite_existing=overwrite_existing,
                        batch_report=batch_report,
                        cvat_input_scale=cvat_input_scale,
                    )
                )

            _LOGGER.info(
                "Batch %s: xml_images=%d pending=%d missing=%d skipped_existing=%d",
                batch_name,
                batch_report.xml_image_count,
                len(html_tasks) if html_only else len(conversion_tasks),
                batch_report.missing_image_count,
                batch_report.skipped_existing_count,
            )

            if dry_run:
                continue

            if html_only:
                task_results = _run_html_tasks(html_tasks, workers=workers)
            else:
                task_results = _run_batch_tasks(conversion_tasks, workers=workers)

            for result in task_results:
                if result.success:
                    batch_report.converted_count += 1
                    continue

                batch_report.failed_count += 1
                assert result.reason is not None
                batch_report.failures.append(
                    ConversionError(image_name=result.image_name, reason=result.reason)
                )
            continue

        image_names = _get_sorted_xml_image_names(xml_path)
        batch_report.xml_image_count = len(image_names)
        doc_json_path = output_json_dir / f"{batch_name}.json"
        doc_html_path = output_html_dir / f"{batch_name}.html"
        layout_path = _layout_visualization_path(doc_html_path)

        if html_only:
            if not doc_json_path.exists():
                message = f"Missing JSON: {doc_json_path}"
                batch_report.failed_count += 1
                batch_report.failures.append(
                    ConversionError(image_name="__document__", reason=message)
                )
                _LOGGER.error("%s", message)
                continue

            if not overwrite_existing and layout_path.exists():
                batch_report.skipped_existing_count += 1
                _LOGGER.info(
                    "Batch %s: xml_images=%d pending=0 missing=%d skipped_existing=%d",
                    batch_name,
                    batch_report.xml_image_count,
                    batch_report.missing_image_count,
                    batch_report.skipped_existing_count,
                )
                continue

            _LOGGER.info(
                "Batch %s (document): xml_images=%d pending=1 missing=%d skipped_existing=%d",
                batch_name,
                batch_report.xml_image_count,
                batch_report.missing_image_count,
                batch_report.skipped_existing_count,
            )
            if dry_run:
                continue

            render_result = _render_single_html_task(
                HtmlRenderTask(
                    image_name="__document__",
                    json_path=doc_json_path,
                    html_path=doc_html_path,
                )
            )
            if render_result.success:
                batch_report.converted_count += 1
            else:
                batch_report.failed_count += 1
                assert render_result.reason is not None
                batch_report.failures.append(
                    ConversionError(
                        image_name=render_result.image_name, reason=render_result.reason
                    )
                )
            continue

        if not overwrite_existing and doc_json_path.exists() and layout_path.exists():
            batch_report.skipped_existing_count += 1
            _LOGGER.info(
                "Batch %s (document): xml_images=%d pending=0 missing=%d skipped_existing=%d",
                batch_name,
                batch_report.xml_image_count,
                batch_report.missing_image_count,
                batch_report.skipped_existing_count,
            )
            continue

        use_pdf_fallback = (
            not images_dir.exists()
            and folder_mode == _FOLDER_MODE_DOCUMENT
            and document_pdf_path is not None
        )
        parsed_file = parse_cvat_file(xml_path) if use_pdf_fallback else None
        page_numbers_by_image = (
            _resolve_xml_page_numbers(parsed_file) if parsed_file is not None else {}
        )
        page_docs: List[Tuple[str, DoclingDocument]] = []
        for image_name in image_names:
            if use_pdf_fallback:
                assert document_pdf_path is not None
                page_number = page_numbers_by_image.get(image_name)
                if page_number is None:
                    batch_report.failed_count += 1
                    batch_report.failures.append(
                        ConversionError(
                            image_name=image_name,
                            reason=f"Unable to resolve PDF page number for {image_name}",
                        )
                    )
                    if allow_partial_document:
                        continue
                    page_docs = []
                    break
            else:
                input_path = images_dir / image_name
                if not input_path.exists():
                    batch_report.missing_image_count += 1
                    batch_report.missing_images.append(image_name)
                    continue

            if dry_run:
                continue

            if use_pdf_fallback:
                assert parsed_file is not None
                assert page_number is not None
                assert document_pdf_path is not None
                pdf_path = document_pdf_path
                converted = _convert_pdf_page_from_xml_image(
                    xml_path=xml_path,
                    pdf_path=pdf_path,
                    image_name=image_name,
                    page_number=page_number,
                    cvat_input_scale=cvat_input_scale,
                    parsed_file=parsed_file,
                )
            else:
                converted = convert_cvat_to_docling(
                    xml_path=xml_path,
                    input_path=input_path,
                    image_identifier=image_name,
                    cvat_input_scale=cvat_input_scale,
                )
            if converted is None:
                batch_report.failed_count += 1
                batch_report.failures.append(
                    ConversionError(
                        image_name=image_name,
                        reason=f"Conversion returned None for page {image_name}",
                    )
                )
                if allow_partial_document:
                    continue
                page_docs = []
                break
            page_docs.append((image_name, converted))

        pending_documents = (
            1
            if (
                len(page_docs) > 0
                if allow_partial_document
                else batch_report.missing_image_count == 0
                and batch_report.failed_count == 0
            )
            else 0
        )

        _LOGGER.info(
            "Batch %s (document): xml_images=%d pending=%d missing=%d skipped_existing=%d",
            batch_name,
            batch_report.xml_image_count,
            pending_documents,
            batch_report.missing_image_count,
            batch_report.skipped_existing_count,
        )

        if dry_run:
            continue

        if not allow_partial_document and batch_report.missing_image_count > 0:
            batch_report.failed_count += 1
            batch_report.failures.append(
                ConversionError(
                    image_name="__document__",
                    reason=(
                        f"Cannot build multipage document for {batch_name}; "
                        f"{batch_report.missing_image_count} images are missing."
                    ),
                )
            )
            continue

        if allow_partial_document and len(page_docs) == 0:
            batch_report.failed_count += 1
            batch_report.failures.append(
                ConversionError(
                    image_name="__document__",
                    reason=(
                        f"Cannot build multipage document for {batch_name}; "
                        "no valid pages were converted."
                    ),
                )
            )
            continue

        if not allow_partial_document and batch_report.failed_count > 0:
            continue

        try:
            merged_doc = _build_multipage_doc(
                batch_name=batch_name, page_docs=page_docs
            )
            merged_doc.save_as_json(doc_json_path)
            save_single_document_html(
                doc_html_path, merged_doc, draw_reading_order=True
            )
            batch_report.converted_count += 1
        except Exception as exc:  # noqa: BLE001
            batch_report.failed_count += 1
            batch_report.failures.append(
                ConversionError(image_name="__document__", reason=str(exc))
            )

    report.finished_at = datetime.now(timezone.utc).isoformat()
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert flat CVAT XML deliveries into Docling JSON + visualization HTML."
    )
    parser.add_argument(
        "--xml-root",
        type=Path,
        required=True,
        help=(
            "Directory containing either flat batch XML files (batch_*_set.xml) "
            "or per-batch folders with annotations.xml."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Directory containing batch folders. Expected per batch: images/ subdirectory; "
            "in --folder-mode=document, <batch_name>.pdf is accepted as fallback."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination directory for conversion outputs.",
    )
    parser.add_argument(
        "--json-dirname",
        type=str,
        default="docling_json",
        help="Name of the JSON output folder under each batch.",
    )
    parser.add_argument(
        "--html-dirname",
        type=str,
        default="docling_html_visualization",
        help="Name of the HTML visualization folder under each batch.",
    )
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Only render visualization HTML from existing JSON files.",
    )
    parser.add_argument(
        "--folder-mode",
        choices=[_FOLDER_MODE_BATCH, _FOLDER_MODE_DOCUMENT],
        required=True,
        help=(
            "How to interpret each folder under xml/data roots. "
            "'batch' converts each image independently. "
            "'document' treats each folder as one multipage document "
            "ordered lexicographically by image name. If images/ is missing in document mode, "
            "<data-root>/<batch_name>/<batch_name>.pdf is used when available."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for conversion (default: 1).",
    )
    parser.add_argument(
        "--cvat-input-scale",
        type=float,
        default=2.0,
        help=(
            "CVAT annotation input scale used when converting PDF-backed pages "
            "(default: 2.0)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only plan and validate mappings, do not run conversions.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Recreate outputs even if JSON and HTML already exist.",
    )
    parser.add_argument(
        "--allow-partial-document",
        action="store_true",
        help=(
            "When --folder-mode=document is used, keep and merge only pages that convert successfully. "
            "Without this flag, any missing/failed page skips the whole document."
        ),
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="conversion_report.json",
        help="Filename of the run report written under output-root.",
    )
    return parser.parse_args(argv)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )


def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)

    xml_root = args.xml_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    workers = max(int(args.workers), 1)
    folder_mode = cast(FolderMode, args.folder_mode)

    if not xml_root.exists():
        raise FileNotFoundError(f"XML root does not exist: {xml_root}")
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    report = run_conversion(
        xml_root=xml_root,
        data_root=data_root,
        output_root=output_root,
        folder_mode=folder_mode,
        json_dirname=args.json_dirname,
        html_dirname=args.html_dirname,
        html_only=args.html_only,
        workers=workers,
        cvat_input_scale=float(args.cvat_input_scale),
        dry_run=args.dry_run,
        overwrite_existing=args.overwrite_existing,
        allow_partial_document=args.allow_partial_document,
    )

    report_path = output_root / args.report_name
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    _LOGGER.info("Report written to %s", report_path)
    _LOGGER.info(
        "Summary: batches=%d converted=%d failed=%d missing_images=%d skipped_existing=%d",
        len(report.batches),
        report.total_converted,
        report.total_failed,
        report.total_missing_images,
        report.total_skipped_existing,
    )


if __name__ == "__main__":
    main()
