import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List

from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    CoordOrigin,
    DocItem,
    DocItemLabel,
    ImageRef,
    PageItem,
    Size,
)
from docling_core.types.io import DocumentStream
from PIL import Image
from pydantic import ValidationError
from tqdm import tqdm

from docling_eval.datamodels.dataset_record import DatasetRecord, DatasetRecordWithBBox
from docling_eval.datamodels.types import BenchMarkColumns
from docling_eval.dataset_builders.dataset_builder import BaseEvaluationDatasetBuilder
from docling_eval.utils.utils import (
    extract_images,
    from_pil_to_base64uri,
    get_binary,
    get_binhash,
)

_log = logging.getLogger(__name__)

_PAGE_SUFFIX_PATTERN = re.compile(r"_page_(\d+)$", re.IGNORECASE)


class DoclingSDGDatasetBuilder(BaseEvaluationDatasetBuilder):
    """
    Dataset builder for local Docling JSON + PNG source files.

    Expected input layout in ``dataset_source``:
    - one Docling JSON file per document (``<doc_id>.json``), and
    - either one PNG (``<doc_id>.png``) or page-wise PNGs
      (``<doc_id>_page_000001.png``, ...).
    """

    def __init__(
        self,
        dataset_source: Path,
        target: Path,
        split: str = "test",
        begin_index: int = 0,
        end_index: int = -1,
    ):
        super().__init__(
            name="DoclingSDG",
            dataset_source=dataset_source,
            target=target,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
        )

        self.must_retrieve = False

    @staticmethod
    def _sort_by_page_suffix(path: Path) -> tuple[int, str]:
        match = _PAGE_SUFFIX_PATTERN.search(path.stem)
        page_no = int(match.group(1)) if match else 0
        return page_no, path.name.lower()

    def _find_json_files(self) -> List[Path]:
        assert isinstance(self.dataset_source, Path)

        files = list(self.dataset_source.glob("*.json"))
        files.extend(self.dataset_source.glob("*.JSON"))

        deduped = {f.resolve(): f for f in files}
        return sorted(deduped.values(), key=lambda p: p.name.lower())

    def _build_png_indices(self) -> tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
        assert isinstance(self.dataset_source, Path)

        png_candidates = list(self.dataset_source.glob("*.png"))
        png_candidates.extend(self.dataset_source.glob("*.PNG"))

        exact_index: Dict[str, List[Path]] = {}
        paged_index: Dict[str, List[Path]] = {}

        for png_path in png_candidates:
            stem = png_path.stem
            page_match = _PAGE_SUFFIX_PATTERN.search(stem)

            if page_match is None:
                exact_index.setdefault(stem, []).append(png_path)
                continue

            base_name = stem[: page_match.start()]
            paged_index.setdefault(base_name, []).append(png_path)

        for key, values in exact_index.items():
            exact_index[key] = sorted(
                {f.resolve(): f for f in values}.values(),
                key=lambda p: p.name.lower(),
            )

        for key, values in paged_index.items():
            paged_index[key] = sorted(
                {f.resolve(): f for f in values}.values(),
                key=self._sort_by_page_suffix,
            )

        return exact_index, paged_index

    def _find_png_files_for_doc(
        self,
        doc_id: str,
        exact_index: Dict[str, List[Path]],
        paged_index: Dict[str, List[Path]],
    ) -> List[Path]:
        base_names = [doc_id]
        if doc_id.lower().endswith(".png"):
            base_names.append(doc_id[:-4])

        for base_name in dict.fromkeys(base_names):
            if not base_name:
                continue
            exact_matches = exact_index.get(base_name)
            if exact_matches:
                return exact_matches

        for base_name in dict.fromkeys(base_names):
            if not base_name:
                continue
            paged_matches = paged_index.get(base_name)
            if paged_matches:
                return paged_matches

        return []

    @staticmethod
    def _load_png_images(files: List[Path]) -> List[Image.Image]:
        images: List[Image.Image] = []

        for png_path in files:
            with Image.open(png_path) as img:
                images.append(img.convert("RGB"))

        return images

    @staticmethod
    def _attach_page_images(
        document: DoclingDocument,
        page_images: List[Image.Image],
    ) -> DoclingDocument:
        for page_no, page_image in enumerate(page_images, start=1):
            image_ref = ImageRef(
                mimetype="image/png",
                dpi=72,
                size=Size(width=page_image.width, height=page_image.height),
                uri=from_pil_to_base64uri(page_image),
            )

            if page_no in document.pages:
                page_item = document.pages[page_no]
                page_item.image = image_ref
                if (
                    page_item.size is None
                    or page_item.size.width <= 0
                    or page_item.size.height <= 0
                ):
                    page_item.size = Size(
                        width=float(page_image.width),
                        height=float(page_image.height),
                    )
            else:
                document.pages[page_no] = PageItem(
                    page_no=page_no,
                    size=Size(
                        width=float(page_image.width),
                        height=float(page_image.height),
                    ),
                    image=image_ref,
                )

        return document

    @staticmethod
    def _page_images_to_pdf_bytes(page_images: List[Image.Image]) -> bytes:
        if not page_images:
            raise ValueError("page_images must not be empty")

        pdf_buffer = BytesIO()
        first_page = page_images[0]
        other_pages = page_images[1:]

        first_page.save(
            pdf_buffer,
            format="PDF",
            save_all=True,
            append_images=other_pages,
        )

        return pdf_buffer.getvalue()

    @staticmethod
    def _label_to_category_id(label: DocItemLabel) -> int:
        """Stable integer category id based on DocItemLabel enum order."""
        return list(DocItemLabel).index(label) + 1

    def _extract_top_level_bboxes(
        self,
        document: DoclingDocument,
        page_no_to_index: Dict[int, int],
        page_images: List[Image.Image],
    ) -> Dict[int, List[Dict]]:
        """
        Extract top-level layout item bboxes preserving document iteration order.

        The returned mapping uses zero-based page indices as required by
        DatasetRecordWithBBox.
        """
        bboxes_by_page: Dict[int, List[Dict]] = {}

        for item, level in document.iterate_items():
            if level != 1 or not isinstance(item, DocItem):
                continue

            if not item.prov:
                continue

            prov = item.prov[0]
            page_no = prov.page_no
            page_index = page_no_to_index.get(page_no)
            if page_index is None or page_index >= len(page_images):
                continue

            page_image = page_images[page_index]
            page_width, page_height = page_image.size

            page = document.pages.get(page_no)
            if page is None:
                continue

            page_size = page.size
            bbox = prov.bbox.to_top_left_origin(page_height=page_size.height)

            # Clamp to image bounds to satisfy DatasetRecordWithBBox validation.
            left = max(0.0, min(float(bbox.l), float(page_width)))
            top = max(0.0, min(float(bbox.t), float(page_height)))
            right = max(left, min(float(bbox.r), float(page_width)))
            bottom = max(top, min(float(bbox.b), float(page_height)))

            width = max(0.0, right - left)
            height = max(0.0, bottom - top)

            bboxes_by_page.setdefault(page_index, []).append(
                {
                    "label": item.label.value,
                    "category_id": self._label_to_category_id(item.label),
                    "bbox": [left, top, width, height],
                    "ltrb": [left, top, right, bottom],
                    "coord_origin": CoordOrigin.TOPLEFT.value,
                }
            )

        return bboxes_by_page

    def get_record_type(self) -> type[DatasetRecord]:
        return DatasetRecordWithBBox

    def iterate(self) -> Iterable[DatasetRecordWithBBox]:
        assert isinstance(self.dataset_source, Path)

        json_files = self._find_json_files()
        exact_png_index, paged_png_index = self._build_png_indices()

        begin, end = self.get_effective_indices(len(json_files))
        selected_json_files = json_files[begin:end]

        self.log_dataset_stats(len(json_files), len(selected_json_files))
        _log.info(
            "Processing DoclingSDG dataset with %d documents",
            len(selected_json_files),
        )

        for json_path in tqdm(
            selected_json_files,
            desc="Processing files for DoclingSDG",
            ncols=128,
        ):
            doc_id = json_path.stem

            try:
                document = DoclingDocument.load_from_json(json_path)
            except ValidationError as exc:
                _log.warning("Validation error for %s: %s. Skipping.", json_path, exc)
                continue
            except Exception as exc:  # noqa: BLE001
                _log.warning("Failed to load %s: %s. Skipping.", json_path, exc)
                continue

            png_files = self._find_png_files_for_doc(
                doc_id=doc_id,
                exact_index=exact_png_index,
                paged_index=paged_png_index,
            )
            if len(png_files) == 0:
                _log.warning(
                    "No matching PNG found for %s. Expected '%s.png' or '%s_page_*.png'. Skipping.",
                    json_path.name,
                    doc_id,
                    doc_id,
                )
                continue

            try:
                page_images = self._load_png_images(png_files)
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "Failed to read PNG files for %s: %s. Skipping.", doc_id, exc
                )
                continue

            page_no_to_index = {
                page_no: idx
                for idx, page_no in enumerate(sorted(document.pages.keys()))
            }

            self._attach_page_images(document, page_images)
            document, pictures, extracted_page_images = extract_images(
                document=document,
                pictures_column=BenchMarkColumns.GROUNDTRUTH_PICTURES.value,
                page_images_column=BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES.value,
            )

            if len(extracted_page_images) == 0:
                extracted_page_images = page_images

            ground_truth_bboxes = self._extract_top_level_bboxes(
                document=document,
                page_no_to_index=page_no_to_index,
                page_images=extracted_page_images,
            )

            if len(png_files) == 1:
                original_bytes = get_binary(png_files[0])
                original_stream = DocumentStream(
                    name=png_files[0].name,
                    stream=BytesIO(original_bytes),
                )
                mime_type = "image/png"
            else:
                original_bytes = self._page_images_to_pdf_bytes(page_images)
                original_stream = DocumentStream(
                    name=f"{doc_id}.pdf",
                    stream=BytesIO(original_bytes),
                )
                mime_type = "application/pdf"

            yield DatasetRecordWithBBox(
                doc_id=doc_id,
                doc_path=json_path,
                doc_hash=get_binhash(original_bytes),
                ground_truth_doc=document,
                ground_truth_pictures=pictures,
                ground_truth_page_images=extracted_page_images,
                GroundTruthBboxOnPageImages=ground_truth_bboxes,
                original=original_stream,
                mime_type=mime_type,
            )
