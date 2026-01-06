import glob
import json
import logging
import os
import sys
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

from docling_core.types.doc.document import DoclingDocument, TableData, TableItem
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.io import DocumentStream
from tqdm import tqdm

# CVAT tools are optional - provided by docling-cvat-tools
try:
    from docling_cvat_tools.cvat_tools.cvat_to_docling import (
        CVATConversionResult,
        convert_cvat_folder_to_docling,
    )
    from docling_cvat_tools.cvat_tools.folder_parser import parse_cvat_folder
    from docling_cvat_tools.datamodels.cvat_types import (
        AnnotationOverview,
        BenchMarkDirs,
    )
except ImportError as e:
    raise ImportError(
        "CVAT dataset builder requires docling-cvat-tools. "
        "Install with: pip install docling-eval[cvat_tools]"
    ) from e
from docling_eval.datamodels.dataset_record import DatasetRecord
from docling_eval.datamodels.types import BenchMarkColumns, EvaluationModality
from docling_eval.dataset_builders.dataset_builder import BaseEvaluationDatasetBuilder
from docling_eval.utils.utils import extract_images, get_binary, get_binhash

# Configure logging
_log = logging.getLogger(__name__)

# Labels to export in HTML visualization
TRUE_HTML_EXPORT_LABELS = {
    DocItemLabel.TITLE,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.TABLE,
    DocItemLabel.PICTURE,
    DocItemLabel.FORMULA,
    DocItemLabel.CHECKBOX_UNSELECTED,
    DocItemLabel.CHECKBOX_SELECTED,
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CODE,
    DocItemLabel.REFERENCE,
    DocItemLabel.CAPTION,
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.FOOTNOTE,
}


class CvatDatasetBuilder(BaseEvaluationDatasetBuilder):
    """
    Dataset builder that creates a dataset from CVAT annotations.

    This class uses the modern `convert_cvat_folder_to_docling()` converter
    to process CVAT annotations and creates a new ground truth dataset in parquet format.
    """

    def __init__(
        self,
        name: str,
        dataset_source: Path,
        target: Path,
        split: str = "test",
    ):
        """
        Initialize the CvatDatasetBuilder.

        Args:
            name: Name of the dataset
            dataset_source: Directory containing CVAT annotations
            target: Path where the new dataset will be saved
            split: Dataset split to use
        """
        super().__init__(
            name=name,
            dataset_source=dataset_source,
            target=target,
            dataset_local_path=None,
            split=split,
        )
        self.must_retrieve = False
        self.benchmark_dirs = BenchMarkDirs()
        self.benchmark_dirs.set_up_directory_structure(
            source=dataset_source, target=dataset_source
        )
        self._temp_json_dir: Optional[Path] = None

    def _resolve_dataset_path(self, path: Union[Path, str]) -> Path:
        """Resolve paths relative to the dataset root when necessary."""

        if path is None:
            return Path("")

        path_obj = Path(path)
        if not str(path_obj):
            return path_obj

        if path_obj.is_absolute():
            return path_obj

        return self.benchmark_dirs.target_dir / path_obj

    def unzip_annotation_files(self, output_dir: Path) -> List[Path]:
        """
        Unzip annotation files to the specified directory.

        Args:
            output_dir: Directory to unzip files to

        Returns:
            List of paths to unzipped files
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        unzipped_files = []
        zip_files = sorted(
            glob.glob(str(self.benchmark_dirs.annotations_zip_dir / "*.zip"))
        )

        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file, "r") as zf:
                for file_name in zf.namelist():
                    # Resolve filename collisions
                    original_file_name = file_name
                    file_path = os.path.join(output_dir, file_name)
                    base, ext = os.path.splitext(file_name)
                    counter = 1
                    while os.path.exists(file_path):
                        # Append a numeric suffix to resolve collisions
                        file_name = f"{base}_{counter}{ext}"
                        file_path = os.path.join(output_dir, file_name)
                        counter += 1

                    # Extract file and add to the list
                    with open(file_path, "wb") as f:
                        f.write(zf.read(original_file_name))
                    unzipped_files.append(Path(file_path))

        return unzipped_files

    def get_annotation_files(self) -> List[Path]:
        """
        Get annotation files from zip files or directly from directory.

        Returns:
            List of paths to annotation files
        """
        zip_files = sorted(
            glob.glob(str(self.benchmark_dirs.annotations_zip_dir / "*.zip"))
        )

        if len(zip_files) > 0:
            _log.info(f"Found {len(zip_files)} zip files")

            existing_xml_files = sorted(
                glob.glob(str(self.benchmark_dirs.annotations_xml_dir / "*.xml"))
            )
            _log.info(
                f"Found {len(existing_xml_files)} existing XML files, clearing..."
            )

            for xml_file in existing_xml_files:
                os.remove(xml_file)

            xml_files = self.unzip_annotation_files(
                self.benchmark_dirs.annotations_xml_dir
            )
        else:
            xml_files = sorted(self.benchmark_dirs.annotations_xml_dir.glob("*.xml"))

        _log.info(f"Processing {len(xml_files)} XML annotation files")
        return xml_files

    def save_to_disk(
        self,
        chunk_size: int = 80,
        max_num_chunks: int = sys.maxsize,
        do_visualization: bool = False,
    ) -> None:
        if do_visualization:
            html_output_dir = self.target / "visualizations"
            os.makedirs(html_output_dir, exist_ok=True)

        super().save_to_disk(
            chunk_size, max_num_chunks, do_visualization=do_visualization
        )

    def _detect_xml_pattern(self) -> str:
        """Detect the XML pattern from available XML files in cvat_tasks directory."""
        assert isinstance(self.dataset_source, Path), "dataset_source must be a Path"
        tasks_dir = self.dataset_source / "cvat_tasks"
        if tasks_dir.exists():
            xml_files = sorted(tasks_dir.glob("task_*_set_*.xml"))
            if xml_files:
                # Check if we have set_A files
                set_a_files = [f for f in xml_files if "_set_A" in f.name]
                if set_a_files:
                    return "task_{xx}_set_A"
                # Extract pattern from first file
                import re

                first_file = xml_files[0].name
                match = re.match(r"task_(\d+)_set_([A-Z])\.xml", first_file)
                if match:
                    return f"task_{{xx}}_set_{match.group(2)}"

        # Default pattern
        return "task_{xx}_set_A"

    def _create_dataset_record_from_json(
        self, json_path: Path, doc_hash: str, folder_structure
    ) -> Optional[DatasetRecord]:
        """Create a DatasetRecord from a JSON file and corresponding PDF."""
        try:
            # Load DoclingDocument from JSON
            doc = DoclingDocument.load_from_json(json_path)

            # Find PDF file from folder structure
            if doc_hash not in folder_structure.documents:
                _log.warning(f"Document hash {doc_hash} not found in folder structure")
                return None

            cvat_doc = folder_structure.documents[doc_hash]
            pdf_path = cvat_doc.bin_file

            if not pdf_path.exists():
                _log.warning(f"PDF file {pdf_path} not found for {doc_hash}")
                return None

            # Extract images from document
            doc, pictures, page_images = extract_images(
                document=doc,
                pictures_column=BenchMarkColumns.GROUNDTRUTH_PICTURES.value,
                page_images_column=BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES.value,
            )

            # Load PDF as binary data
            pdf_bytes = get_binary(pdf_path)
            pdf_stream = DocumentStream(name=pdf_path.name, stream=BytesIO(pdf_bytes))

            # Create dataset record
            record = DatasetRecord(
                doc_id=doc.name or json_path.stem,
                doc_path=str(json_path.stem),
                doc_hash=get_binhash(pdf_bytes),
                ground_truth_doc=doc,
                ground_truth_pictures=pictures,
                ground_truth_page_images=page_images,
                original=pdf_stream,
                mime_type=cvat_doc.mime_type or "application/pdf",
                modalities=[
                    EvaluationModality.LAYOUT,
                    EvaluationModality.READING_ORDER,
                    EvaluationModality.CAPTIONING,
                ],
            )

            return record

        except Exception as e:
            _log.error(f"Error creating dataset record from {json_path}: {e}")
            return None

    def iterate(self) -> Iterable[DatasetRecord]:
        """
        Create dataset records from CVAT annotations using modern converter.

        This method uses `convert_cvat_folder_to_docling()` to convert CVAT annotations
        to DoclingDocument JSON files, then converts those to DatasetRecord objects.

        Returns:
            Iterable of DatasetRecord objects
        """
        assert isinstance(self.dataset_source, Path), "dataset_source must be a Path"
        # Detect XML pattern
        xml_pattern = self._detect_xml_pattern()
        _log.info(f"Using XML pattern: {xml_pattern}")

        # Parse folder structure
        try:
            folder_structure = parse_cvat_folder(self.dataset_source, xml_pattern)
            _log.info(
                f"Found {len(folder_structure.documents)} documents in folder structure"
            )
        except Exception as e:
            _log.error(f"Failed to parse CVAT folder: {e}")
            raise

        # Create temporary directory for JSON files
        self._temp_json_dir = Path(tempfile.mkdtemp(prefix="cvat_dataset_builder_"))
        _log.info(f"Using temporary JSON directory: {self._temp_json_dir}")

        try:
            # Step 1: Convert CVAT folder to DoclingDocument JSON files
            _log.info("Converting CVAT annotations to DoclingDocument JSON files...")
            conversion_results = convert_cvat_folder_to_docling(
                folder_path=self.dataset_source,
                xml_pattern=xml_pattern,
                output_dir=self._temp_json_dir,
                save_formats=["json"],
                folder_structure=folder_structure,
                log_validation=False,
                force_ocr=False,
                cvat_input_scale=2.0,
                storage_scale=2.0,
            )

            # Step 2: Convert JSON results to DatasetRecord objects
            # Sort by document name for consistent ordering
            sorted_results = sorted(
                conversion_results.items(),
                key=lambda x: folder_structure.documents[x[0]].doc_name,
            )

            # Calculate total items for effective indices
            total_items = len(sorted_results)
            begin, end = self.get_effective_indices(total_items)

            # Log statistics
            self.log_dataset_stats(total_items, end - begin)

            _log.info(f"Processing documents from index {begin} to {end}")

            item_count = 0
            for doc_hash, result in tqdm(
                sorted_results,
                total=len(sorted_results),
                ncols=128,
                desc="Creating dataset records",
            ):
                # Skip failed conversions
                if result.error is not None:
                    _log.warning(
                        f"Skipping {doc_hash} due to conversion error: {result.error}"
                    )
                    continue

                # Apply index filtering
                item_count += 1
                if item_count < begin:
                    continue
                if item_count >= end:
                    break

                # Find JSON file path
                cvat_doc = folder_structure.documents[doc_hash]
                json_path = self._temp_json_dir / f"{cvat_doc.doc_name}.json"

                if not json_path.exists():
                    _log.warning(
                        f"JSON file not found: {json_path} (conversion may have failed silently)"
                    )
                    continue

                # Create DatasetRecord from JSON
                record = self._create_dataset_record_from_json(
                    json_path, doc_hash, folder_structure
                )

                if record is not None:
                    yield record

        finally:
            # Clean up temporary directory
            if self._temp_json_dir and self._temp_json_dir.exists():
                import shutil

                try:
                    shutil.rmtree(self._temp_json_dir)
                    _log.debug(f"Cleaned up temporary directory: {self._temp_json_dir}")
                except Exception as e:
                    _log.warning(f"Failed to clean up temporary directory: {e}")


def find_table_data(doc: DoclingDocument, prov, iou_cutoff: float = 0.90):
    """
    Find table data in a document based on provenance.

    Args:
        doc: Document to search in
        prov: Provenance to match
        iou_cutoff: IoU threshold for matching

    Returns:
        TableData structure from the matching table or an empty structure
    """
    for item, _ in doc.iterate_items():
        if isinstance(item, TableItem):
            for item_prov in item.prov:
                if item_prov.page_no != prov.page_no:
                    continue

                page_height = doc.pages[item_prov.page_no].size.height

                item_bbox_bl = item_prov.bbox.to_bottom_left_origin(
                    page_height=page_height
                )
                prov_bbox_bl = prov.bbox.to_bottom_left_origin(page_height=page_height)

                # iou = item_prov.bbox.intersection_over_union(prov.bbox)
                iou = item_bbox_bl.intersection_over_union(prov_bbox_bl)

                if iou > iou_cutoff:
                    _log.debug(f"Found matching table data with IoU: {iou:.2f}")
                    return item.data

    _log.warning("No matching table data found")

    # Return empty table data
    return TableData(num_rows=-1, num_cols=-1, table_cells=[])
