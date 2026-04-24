import copy
import logging
import math
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple

from docling_core.transforms.visualizer.table_visualizer import TableVisualizer
from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItem,
    DocItemLabel,
    ImageRef,
    PageItem,
    Size,
    TableItem,
)
from docling_core.types.io import DocumentStream
from PIL import Image
from pydantic import ValidationError
from tqdm import tqdm

from docling_eval.datamodels.dataset_record import DatasetRecord, DatasetRecordWithBBox
from docling_eval.datamodels.types import BenchMarkColumns, EvaluationModality
from docling_eval.dataset_builders.dataset_builder import BaseEvaluationDatasetBuilder
from docling_eval.utils.utils import (
    extract_images,
    from_pil_to_base64,
    from_pil_to_base64uri,
    get_binary,
    get_binhash,
    insert_images_from_pil,
)
from docling_eval.visualisation.visualisations import get_missing_pageimg

_log = logging.getLogger(__name__)

_PAGE_SUFFIX_PATTERN = re.compile(r"_page_(\d+)$", re.IGNORECASE)
_BBOX_OVERLAP_EPSILON = 1e-3
_ROTATION_ASPECT_THRESHOLD = 1.15
_COORD_TOLERANCE = 1.0
_ROW_LEFT_DUPLICATE_TOLERANCE = 0.5

_TABLE_REGION_CATEGORY_IDS: Dict[str, int] = {
    "table": 1,
    "row": 2,
    "column": 3,
    "cell_column_header": 4,
    "cell_row_header": 5,
    "cell_section_header": 6,
    "cell_single": 7,
    "cell_merged": 8,
}


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
        modality: str = EvaluationModality.LAYOUT.value,
        split: str = "test",
        begin_index: int = 0,
        end_index: int = -1,
    ):
        try:
            parsed_modality = EvaluationModality(modality)
        except ValueError as exc:
            raise ValueError(
                "Unsupported DoclingSDG modality "
                f"'{modality}'. Expected one of: "
                f"{EvaluationModality.LAYOUT.value}, {EvaluationModality.TABLE_REGIONS.value}"
            ) from exc

        if parsed_modality not in (
            EvaluationModality.LAYOUT,
            EvaluationModality.TABLE_REGIONS,
        ):
            raise ValueError(
                "DoclingSDG modality must be "
                f"'{EvaluationModality.LAYOUT.value}' or "
                f"'{EvaluationModality.TABLE_REGIONS.value}', "
                f"got '{parsed_modality.value}'."
            )

        super().__init__(
            name="DoclingSDG",
            dataset_source=dataset_source,
            target=target,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
        )

        self.modality = parsed_modality
        self.must_retrieve = False

    @staticmethod
    def _sort_by_page_suffix(path: Path) -> tuple[int, str]:
        match = _PAGE_SUFFIX_PATTERN.search(path.stem)
        page_no = int(match.group(1)) if match else 0
        return page_no, path.name.lower()

    def _find_json_files(self) -> List[Path]:
        assert isinstance(self.dataset_source, Path)

        files = list(self.dataset_source.rglob("*.json"))
        files.extend(self.dataset_source.rglob("*.JSON"))

        deduped = {f.resolve(): f for f in files}
        return sorted(
            deduped.values(),
            key=lambda p: (str(p.parent).lower(), p.name.lower()),
        )

    def _build_png_indices(
        self,
    ) -> tuple[Dict[str, Dict[Path, List[Path]]], Dict[str, Dict[Path, List[Path]]]]:
        assert isinstance(self.dataset_source, Path)

        png_candidates = list(self.dataset_source.rglob("*.png"))
        png_candidates.extend(self.dataset_source.rglob("*.PNG"))

        exact_index: Dict[str, Dict[Path, List[Path]]] = {}
        paged_index: Dict[str, Dict[Path, List[Path]]] = {}

        for png_path in png_candidates:
            stem = png_path.stem
            page_match = _PAGE_SUFFIX_PATTERN.search(stem)
            parent = png_path.parent.resolve()

            if page_match is None:
                exact_index.setdefault(stem, {}).setdefault(parent, []).append(png_path)
                continue

            base_name = stem[: page_match.start()]
            paged_index.setdefault(base_name, {}).setdefault(parent, []).append(
                png_path
            )

        for parent_map in exact_index.values():
            for parent, values in parent_map.items():
                parent_map[parent] = sorted(
                    {f.resolve(): f for f in values}.values(),
                    key=lambda p: p.name.lower(),
                )

        for parent_map in paged_index.values():
            for parent, values in parent_map.items():
                parent_map[parent] = sorted(
                    {f.resolve(): f for f in values}.values(),
                    key=self._sort_by_page_suffix,
                )

        return exact_index, paged_index

    @staticmethod
    def _find_local_png_files_for_doc(doc_id: str, json_path: Path) -> List[Path]:
        local_exact = [json_path.with_suffix(".png"), json_path.with_suffix(".PNG")]
        for exact_path in local_exact:
            if exact_path.exists():
                return [exact_path]

        paged = list(json_path.parent.glob(f"{doc_id}_page_*.png"))
        paged.extend(json_path.parent.glob(f"{doc_id}_page_*.PNG"))
        return sorted(paged, key=DoclingSDGDatasetBuilder._sort_by_page_suffix)

    @staticmethod
    def _select_matches_from_index(
        *,
        base_name: str,
        index: Dict[str, Dict[Path, List[Path]]],
        preferred_parent: Path,
    ) -> List[Path]:
        parent_map = index.get(base_name)
        if not parent_map:
            return []

        if preferred_parent in parent_map:
            return parent_map[preferred_parent]

        if len(parent_map) == 1:
            return next(iter(parent_map.values()))

        selected_parent = sorted(parent_map.keys(), key=lambda p: str(p))[0]
        _log.warning(
            "Multiple PNG groups matched '%s'. Picking %s (requested near %s).",
            base_name,
            selected_parent,
            preferred_parent,
        )
        return parent_map[selected_parent]

    def _find_png_files_for_doc(
        self,
        doc_id: str,
        json_path: Path,
        exact_index: Dict[str, Dict[Path, List[Path]]],
        paged_index: Dict[str, Dict[Path, List[Path]]],
    ) -> List[Path]:
        local_matches = self._find_local_png_files_for_doc(
            doc_id=doc_id, json_path=json_path
        )
        if local_matches:
            return local_matches

        preferred_parent = json_path.parent.resolve()
        base_names = [doc_id]
        if doc_id.lower().endswith(".png"):
            base_names.append(doc_id[:-4])

        for base_name in dict.fromkeys(base_names):
            if not base_name:
                continue
            exact_matches = self._select_matches_from_index(
                base_name=base_name,
                index=exact_index,
                preferred_parent=preferred_parent,
            )
            if exact_matches:
                return exact_matches

        for base_name in dict.fromkeys(base_names):
            if not base_name:
                continue
            paged_matches = self._select_matches_from_index(
                base_name=base_name,
                index=paged_index,
                preferred_parent=preferred_parent,
            )
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

    @staticmethod
    def _table_region_category_id(label: str) -> int:
        return _TABLE_REGION_CATEGORY_IDS[label]

    @staticmethod
    def _safe_page_height(page: Optional[PageItem], page_image: Image.Image) -> float:
        if (
            page is not None
            and page.size is not None
            and page.size.height is not None
            and page.size.height > 0
        ):
            return float(page.size.height)
        return float(page_image.height)

    @staticmethod
    def _normalize_bbox_to_page_image(
        bbox: BoundingBox,
        *,
        page_height: float,
        page_image: Image.Image,
        reject_out_of_bounds: bool = False,
    ) -> Optional[Tuple[float, float, float, float]]:
        if bbox.coord_origin != CoordOrigin.TOPLEFT:
            bbox = bbox.to_top_left_origin(page_height=page_height)

        left = float(bbox.l)
        top = float(bbox.t)
        right = float(bbox.r)
        bottom = float(bbox.b)
        if not all(math.isfinite(v) for v in (left, top, right, bottom)):
            return None
        if right - left <= 0.0 or bottom - top <= 0.0:
            return None

        if reject_out_of_bounds:
            if (
                left < -_COORD_TOLERANCE
                or top < -_COORD_TOLERANCE
                or right > float(page_image.width) + _COORD_TOLERANCE
                or bottom > float(page_image.height) + _COORD_TOLERANCE
            ):
                return None
            return (
                max(0.0, left),
                max(0.0, top),
                min(float(page_image.width), right),
                min(float(page_image.height), bottom),
            )

        left = max(0.0, min(left, float(page_image.width)))
        top = max(0.0, min(top, float(page_image.height)))
        right = max(left, min(right, float(page_image.width)))
        bottom = max(top, min(bottom, float(page_image.height)))
        return left, top, right, bottom

    @staticmethod
    def _bbox_payload(
        *,
        label: str,
        category_id: int,
        rect: Tuple[float, float, float, float],
        extras: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        left, top, right, bottom = rect
        payload: Dict[str, Any] = {
            "label": label,
            "category_id": category_id,
            "bbox": [left, top, right - left, bottom - top],
            "ltrb": [left, top, right, bottom],
            "coord_origin": CoordOrigin.TOPLEFT.value,
        }
        if extras:
            payload.update(extras)
        return payload

    @staticmethod
    def _union_rectangles(
        rects: List[Tuple[float, float, float, float]],
    ) -> Tuple[float, float, float, float]:
        return (
            min(rect[0] for rect in rects),
            min(rect[1] for rect in rects),
            max(rect[2] for rect in rects),
            max(rect[3] for rect in rects),
        )

    @staticmethod
    def _ranges_overlap(
        start_a: int,
        end_a: int,
        start_b: int,
        end_b: int,
    ) -> bool:
        return not (end_a <= start_b or end_b <= start_a)

    @staticmethod
    def _rectangles_overlap(
        rect_a: Tuple[float, float, float, float],
        rect_b: Tuple[float, float, float, float],
    ) -> bool:
        inter_w = min(rect_a[2], rect_b[2]) - max(rect_a[0], rect_b[0])
        inter_h = min(rect_a[3], rect_b[3]) - max(rect_a[1], rect_b[1])
        return inter_w > _BBOX_OVERLAP_EPSILON and inter_h > _BBOX_OVERLAP_EPSILON

    @staticmethod
    def _is_table_rotated_90(
        row_rects: List[Tuple[float, float, float, float]],
        col_rects: List[Tuple[float, float, float, float]],
    ) -> bool:
        if not row_rects or not col_rects:
            return False

        row_aspects = [
            (rect[2] - rect[0]) / max(rect[3] - rect[1], 1e-9) for rect in row_rects
        ]
        col_aspects = [
            (rect[2] - rect[0]) / max(rect[3] - rect[1], 1e-9) for rect in col_rects
        ]

        median_row_aspect = median(row_aspects)
        median_col_aspect = median(col_aspects)
        return (
            median_row_aspect < (1.0 / _ROTATION_ASPECT_THRESHOLD)
            and median_col_aspect > _ROTATION_ASPECT_THRESHOLD
        )

    @staticmethod
    def _cell_label(
        *,
        is_column_header: bool,
        is_row_header: bool,
        is_row_section: bool,
        row_span: int,
        col_span: int,
    ) -> str:
        if is_column_header:
            return "cell_column_header"
        if is_row_header:
            return "cell_row_header"
        if is_row_section:
            return "cell_section_header"
        if row_span > 1 or col_span > 1:
            return "cell_merged"
        return "cell_single"

    @staticmethod
    def _find_duplicate_left_in_row_band(
        cell_entries: List[Dict[str, Any]],
    ) -> Optional[str]:
        rows_to_cells: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        for entry in cell_entries:
            rows_to_cells[(int(entry["row_start"]), int(entry["row_end"]))].append(
                entry
            )

        for (row_start, row_end), entries in rows_to_cells.items():
            sorted_entries = sorted(entries, key=lambda e: float(e["rect"][0]))
            for i, entry_a in enumerate(sorted_entries):
                left_a = float(entry_a["rect"][0])
                col_a = (int(entry_a["col_start"]), int(entry_a["col_end"]))
                for entry_b in sorted_entries[i + 1 :]:
                    left_b = float(entry_b["rect"][0])
                    if left_b - left_a > _ROW_LEFT_DUPLICATE_TOLERANCE:
                        break
                    col_b = (int(entry_b["col_start"]), int(entry_b["col_end"]))
                    if (
                        col_a != col_b
                        and abs(left_a - left_b) <= _ROW_LEFT_DUPLICATE_TOLERANCE
                    ):
                        return (
                            "Cells in the same row band share the same left coordinate "
                            f"(row_start={row_start}, row_end={row_end}, left~{left_a:.3f})."
                        )
        return None

    @staticmethod
    def _is_empty_text(value: Any) -> bool:
        if value is None:
            return True
        return str(value).strip() == ""

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _cell_end_col(cls, cell: Any) -> int:
        start_col = cls._as_int(getattr(cell, "start_col_offset_idx", 0), 0)
        end_col_raw = getattr(cell, "end_col_offset_idx", None)
        if end_col_raw is not None:
            end_col = cls._as_int(end_col_raw, start_col + 1)
            if end_col > start_col:
                return end_col
        col_span = max(1, cls._as_int(getattr(cell, "col_span", 1), 1))
        return start_col + col_span

    @classmethod
    def _cell_end_row(cls, cell: Any) -> int:
        start_row = cls._as_int(getattr(cell, "start_row_offset_idx", 0), 0)
        end_row_raw = getattr(cell, "end_row_offset_idx", None)
        if end_row_raw is not None:
            end_row = cls._as_int(end_row_raw, start_row + 1)
            if end_row > start_row:
                return end_row
        row_span = max(1, cls._as_int(getattr(cell, "row_span", 1), 1))
        return start_row + row_span

    @staticmethod
    def _is_data_cell(cell: Any) -> bool:
        return (
            not bool(getattr(cell, "column_header", False))
            and not bool(getattr(cell, "row_header", False))
            and not bool(getattr(cell, "row_section", False))
        )

    @classmethod
    def _has_non_empty_data_cell(cls, row_cells: List[Any]) -> bool:
        return any(
            cls._is_data_cell(cell)
            and not cls._is_empty_text(getattr(cell, "text", ""))
            for cell in row_cells
        )

    def _uniformize_table_structure(self, table: TableItem) -> None:
        """
        Uniformize table structure using only rules 1..5b.

        Intentionally excludes rule 0 and rule 6.
        """
        table_data = table.data
        if table_data is None or not table_data.table_cells:
            return

        cells_by_row: Dict[int, List[Any]] = defaultdict(list)
        for cell in table_data.table_cells:
            row_idx = self._as_int(getattr(cell, "start_row_offset_idx", 0), 0)
            cells_by_row[row_idx].append(cell)

        # Step 1: empty data cells become column headers if row already has non-empty column headers.
        for row_cells in cells_by_row.values():
            has_non_empty_column_header = any(
                bool(getattr(cell, "column_header", False))
                and not self._is_empty_text(getattr(cell, "text", ""))
                for cell in row_cells
            )
            if not has_non_empty_column_header:
                continue
            for cell in row_cells:
                if self._is_data_cell(cell) and self._is_empty_text(
                    getattr(cell, "text", "")
                ):
                    cell.column_header = True
                    cell.row_header = False
                    cell.row_section = False

        # Step 2a: convert row data cells to column headers if row starts with empty data cell
        # and row is first or row above is all column headers.
        for row_idx in sorted(cells_by_row):
            row_cells = cells_by_row[row_idx]
            first_col_cell = next(
                (
                    cell
                    for cell in row_cells
                    if self._as_int(getattr(cell, "start_col_offset_idx", -1), -1) == 0
                ),
                None,
            )
            if first_col_cell is None:
                continue

            first_is_empty_data = self._is_data_cell(
                first_col_cell
            ) and self._is_empty_text(getattr(first_col_cell, "text", ""))
            if not first_is_empty_data:
                continue

            convert_row = False
            if row_idx == 0:
                convert_row = True
            else:
                prev_row_cells = cells_by_row.get(row_idx - 1, [])
                if prev_row_cells and all(
                    bool(getattr(cell, "column_header", False))
                    for cell in prev_row_cells
                ):
                    convert_row = True

            if not convert_row:
                continue
            for cell in row_cells:
                if self._is_data_cell(cell):
                    cell.column_header = True
                    cell.row_header = False
                    cell.row_section = False

        # Step 2b: convert data cells into row headers when their column headers are all empty.
        cells_by_col: Dict[int, List[Any]] = defaultdict(list)
        for cell in table_data.table_cells:
            col_idx = self._as_int(getattr(cell, "start_col_offset_idx", 0), 0)
            cells_by_col[col_idx].append(cell)

        cols_with_empty_headers: List[int] = []
        for col_idx in sorted(cells_by_col):
            col_cells = cells_by_col[col_idx]
            col_header_cells = [
                cell
                for cell in col_cells
                if bool(getattr(cell, "column_header", False))
            ]
            if col_header_cells and all(
                self._is_empty_text(getattr(cell, "text", ""))
                for cell in col_header_cells
            ):
                cols_with_empty_headers.append(col_idx)

        for col_idx in cols_with_empty_headers:
            can_convert = False
            if col_idx == 0:
                can_convert = True
            else:
                prev_col_cells = cells_by_col.get(col_idx - 1, [])
                can_convert = any(
                    bool(getattr(cell, "row_header", False)) for cell in prev_col_cells
                )
            if not can_convert:
                continue

            for cell in cells_by_col.get(col_idx, []):
                if self._is_data_cell(cell):
                    cell.row_header = True
                    cell.column_header = False
                    cell.row_section = False

        # Step 3: merge row-header/row-section rows whose non-header cells are all empty data cells.
        cells_to_remove: List[Any] = []
        if cells_by_row:
            last_row_idx = max(cells_by_row)
            for row_idx in sorted(cells_by_row):
                if row_idx == 0 or row_idx == last_row_idx:
                    continue

                prev_row_cells = cells_by_row.get(row_idx - 1, [])
                next_row_cells = cells_by_row.get(row_idx + 1, [])
                if not self._has_non_empty_data_cell(prev_row_cells):
                    continue
                if not self._has_non_empty_data_cell(next_row_cells):
                    continue

                row_cells = cells_by_row[row_idx]
                header_cells = [
                    cell
                    for cell in row_cells
                    if bool(getattr(cell, "row_header", False))
                    or bool(getattr(cell, "row_section", False))
                ]
                for header_cell in header_cells:
                    other_cells = [
                        cell for cell in row_cells if cell is not header_cell
                    ]
                    if not other_cells:
                        continue
                    if not all(
                        self._is_data_cell(cell)
                        and self._is_empty_text(getattr(cell, "text", ""))
                        for cell in other_cells
                    ):
                        continue
                    all_cells = [header_cell, *other_cells]
                    if not all(
                        getattr(cell, "bbox", None) is not None for cell in all_cells
                    ):
                        continue

                    coord_origin = header_cell.bbox.coord_origin
                    if any(
                        cell.bbox.coord_origin != coord_origin for cell in all_cells
                    ):
                        continue

                    min_l = min(float(cell.bbox.l) for cell in all_cells)
                    min_t = min(float(cell.bbox.t) for cell in all_cells)
                    max_r = max(float(cell.bbox.r) for cell in all_cells)
                    max_b = max(float(cell.bbox.b) for cell in all_cells)
                    if not (min_l < max_r and min_t < max_b):
                        continue

                    header_cell.bbox = BoundingBox(
                        l=min_l,
                        r=max_r,
                        t=min_t,
                        b=max_b,
                        coord_origin=coord_origin,
                    )

                    min_col = min(
                        self._as_int(getattr(cell, "start_col_offset_idx", 0), 0)
                        for cell in all_cells
                    )
                    max_col = max(self._cell_end_col(cell) for cell in all_cells)
                    if max_col > min_col:
                        header_cell.start_col_offset_idx = min_col
                        header_cell.end_col_offset_idx = max_col
                        header_cell.col_span = max_col - min_col

                    for cell in other_cells:
                        if cell not in cells_to_remove:
                            cells_to_remove.append(cell)
                    break

        if cells_to_remove:
            table_data.table_cells = [
                cell for cell in table_data.table_cells if cell not in cells_to_remove
            ]

        # Step 4: row-header spanning all columns becomes row-section.
        num_cols_raw = getattr(table_data, "num_cols", None)
        if num_cols_raw is None:
            num_cols = max(
                (
                    self._cell_end_col(cell)
                    for cell in table_data.table_cells
                    if getattr(cell, "start_col_offset_idx", None) is not None
                ),
                default=0,
            )
        else:
            num_cols = self._as_int(num_cols_raw, 0)

        if num_cols > 0:
            for cell in table_data.table_cells:
                if not bool(getattr(cell, "row_header", False)):
                    continue
                start_col = self._as_int(getattr(cell, "start_col_offset_idx", 0), 0)
                end_col = self._cell_end_col(cell)
                if start_col == 0 and end_col >= num_cols:
                    cell.row_section = True
                    cell.row_header = False

        # Step 5a/5b: align bboxes to row/column boundaries according to spans.
        try:
            col_bboxes = table_data.get_column_bounding_boxes(minimal=False)
            row_bboxes = table_data.get_row_bounding_boxes(minimal=False)
        except Exception:  # noqa: BLE001
            return

        for cell in table_data.table_cells:
            cell_bbox = getattr(cell, "bbox", None)
            if cell_bbox is None:
                continue

            new_l = float(cell_bbox.l)
            new_r = float(cell_bbox.r)
            new_t = float(cell_bbox.t)
            new_b = float(cell_bbox.b)

            start_col = self._as_int(getattr(cell, "start_col_offset_idx", 0), 0)
            start_row = self._as_int(getattr(cell, "start_row_offset_idx", 0), 0)
            col_span = max(
                1,
                self._as_int(
                    getattr(cell, "col_span", self._cell_end_col(cell) - start_col),
                    self._cell_end_col(cell) - start_col,
                ),
            )
            row_span = max(
                1,
                self._as_int(
                    getattr(cell, "row_span", self._cell_end_row(cell) - start_row),
                    self._cell_end_row(cell) - start_row,
                ),
            )

            end_col_inclusive = start_col + col_span - 1
            end_row_inclusive = start_row + row_span - 1

            if start_col in col_bboxes and end_col_inclusive in col_bboxes:
                new_l = float(col_bboxes[start_col].l)
                new_r = float(col_bboxes[end_col_inclusive].r)
            if start_row in row_bboxes and end_row_inclusive in row_bboxes:
                new_t = float(row_bboxes[start_row].t)
                new_b = float(row_bboxes[end_row_inclusive].b)

            if new_l < new_r and new_t < new_b:
                cell.bbox = BoundingBox(
                    l=new_l,
                    r=new_r,
                    t=new_t,
                    b=new_b,
                    coord_origin=cell_bbox.coord_origin,
                )

    def _uniformize_tables_after_malformed_check(
        self, document: DoclingDocument
    ) -> None:
        for table in document.tables:
            if not isinstance(table, TableItem):
                continue
            try:
                self._uniformize_table_structure(table)
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "Table uniformization failed for table in %s: %s",
                    document.name,
                    exc,
                )

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
            page = document.pages.get(page_no)
            if page is None:
                continue

            page_height = self._safe_page_height(page, page_image)
            normalized = self._normalize_bbox_to_page_image(
                prov.bbox, page_height=page_height, page_image=page_image
            )
            if normalized is None:
                continue

            bboxes_by_page.setdefault(page_index, []).append(
                self._bbox_payload(
                    label=item.label.value,
                    category_id=self._label_to_category_id(item.label),
                    rect=normalized,
                )
            )

        return bboxes_by_page

    def _extract_table_region_bboxes(
        self,
        document: DoclingDocument,
        page_no_to_index: Dict[int, int],
        page_images: List[Image.Image],
    ) -> Tuple[Dict[int, List[Dict[str, Any]]], bool, Optional[str]]:
        bboxes_by_page: Dict[int, List[Dict[str, Any]]] = {}
        has_rotated_90 = False
        found_table = False

        for item, level in document.iterate_items():
            if level != 1 or not isinstance(item, TableItem):
                continue

            found_table = True
            if not item.prov:
                return {}, has_rotated_90, "Table item has no provenance."

            prov = item.prov[0]
            page_no = prov.page_no
            page_index = page_no_to_index.get(page_no)
            if page_index is None or page_index >= len(page_images):
                return {}, has_rotated_90, f"Table page {page_no} is not available."

            page_image = page_images[page_index]
            page_height = self._safe_page_height(
                document.pages.get(page_no), page_image
            )

            table_rect = self._normalize_bbox_to_page_image(
                prov.bbox,
                page_height=page_height,
                page_image=page_image,
                reject_out_of_bounds=True,
            )
            if table_rect is None:
                return {}, has_rotated_90, "Invalid table bbox (out of image bounds)."

            table_data = item.data
            if table_data is None:
                return {}, has_rotated_90, "Table has no table_data payload."
            table_cells = list(table_data.table_cells)
            if not table_cells:
                return {}, has_rotated_90, "Table has no cells."

            row_rects_map: Dict[int, List[Tuple[float, float, float, float]]] = (
                defaultdict(list)
            )
            col_rects_map: Dict[int, List[Tuple[float, float, float, float]]] = (
                defaultdict(list)
            )
            occupancy: Dict[Tuple[int, int], int] = {}
            cell_entries: List[Dict[str, Any]] = []

            for cell_idx, cell in enumerate(table_cells):
                row_start = int(cell.start_row_offset_idx)
                col_start = int(cell.start_col_offset_idx)
                row_span = max(1, int(cell.row_span or 1))
                col_span = max(1, int(cell.col_span or 1))

                row_end_candidate = (
                    int(cell.end_row_offset_idx)
                    if cell.end_row_offset_idx is not None
                    else row_start + row_span
                )
                col_end_candidate = (
                    int(cell.end_col_offset_idx)
                    if cell.end_col_offset_idx is not None
                    else col_start + col_span
                )
                row_end = max(row_start + row_span, row_end_candidate)
                col_end = max(col_start + col_span, col_end_candidate)

                if row_start < 0 or col_start < 0:
                    return {}, has_rotated_90, "Negative row/column indices detected."
                if row_end <= row_start or col_end <= col_start:
                    return {}, has_rotated_90, "Invalid row/column span detected."

                cell_rect = self._normalize_bbox_to_page_image(
                    cell.bbox,
                    page_height=page_height,
                    page_image=page_image,
                    reject_out_of_bounds=True,
                )
                if cell_rect is None:
                    return (
                        {},
                        has_rotated_90,
                        "Invalid cell bbox (out of image bounds).",
                    )

                for row_id in range(row_start, row_end):
                    row_rects_map[row_id].append(cell_rect)
                for col_id in range(col_start, col_end):
                    col_rects_map[col_id].append(cell_rect)

                for row_id in range(row_start, row_end):
                    for col_id in range(col_start, col_end):
                        slot = (row_id, col_id)
                        if slot in occupancy:
                            return (
                                {},
                                has_rotated_90,
                                "Overlapping cells in table grid.",
                            )
                        occupancy[slot] = cell_idx

                label = self._cell_label(
                    is_column_header=bool(cell.column_header),
                    is_row_header=bool(cell.row_header),
                    is_row_section=bool(cell.row_section),
                    row_span=row_span,
                    col_span=col_span,
                )

                cell_entries.append(
                    {
                        "rect": cell_rect,
                        "label": label,
                        "text": cell.text,
                        "row_id": row_start,
                        "col_id": col_start,
                        "row_span": row_span,
                        "col_span": col_span,
                        "row_start": row_start,
                        "row_end": row_end,
                        "col_start": col_start,
                        "col_end": col_end,
                    }
                )

            if not row_rects_map or not col_rects_map:
                return {}, has_rotated_90, "Missing row/column regions."

            active_cells: List[Dict[str, Any]] = []
            for entry in sorted(cell_entries, key=lambda e: e["rect"][0]):
                left = float(entry["rect"][0])
                active_cells = [
                    other
                    for other in active_cells
                    if float(other["rect"][2]) > left + _BBOX_OVERLAP_EPSILON
                ]

                for other in active_cells:
                    if not self._rectangles_overlap(entry["rect"], other["rect"]):
                        continue

                    same_grid = self._ranges_overlap(
                        int(entry["row_start"]),
                        int(entry["row_end"]),
                        int(other["row_start"]),
                        int(other["row_end"]),
                    ) and self._ranges_overlap(
                        int(entry["col_start"]),
                        int(entry["col_end"]),
                        int(other["col_start"]),
                        int(other["col_end"]),
                    )
                    if same_grid:
                        return (
                            {},
                            has_rotated_90,
                            "Cell bbox overlap in same table grid.",
                        )
                    return (
                        {},
                        has_rotated_90,
                        "Cell bbox overlap in disjoint grid regions.",
                    )

                active_cells.append(entry)

            page_boxes = bboxes_by_page.setdefault(page_index, [])
            page_boxes.append(
                self._bbox_payload(
                    label="table",
                    category_id=self._table_region_category_id("table"),
                    rect=table_rect,
                )
            )

            row_rects: List[Tuple[float, float, float, float]] = []
            for row_id in sorted(row_rects_map):
                row_rect = self._union_rectangles(row_rects_map[row_id])
                row_rects.append(row_rect)
                page_boxes.append(
                    self._bbox_payload(
                        label="row",
                        category_id=self._table_region_category_id("row"),
                        rect=row_rect,
                        extras={"row_id": row_id},
                    )
                )

            col_rects: List[Tuple[float, float, float, float]] = []
            for col_id in sorted(col_rects_map):
                col_rect = self._union_rectangles(col_rects_map[col_id])
                col_rects.append(col_rect)
                page_boxes.append(
                    self._bbox_payload(
                        label="column",
                        category_id=self._table_region_category_id("column"),
                        rect=col_rect,
                        extras={"col_id": col_id},
                    )
                )

            if self._is_table_rotated_90(row_rects=row_rects, col_rects=col_rects):
                has_rotated_90 = True
            else:
                row_left_issue = self._find_duplicate_left_in_row_band(cell_entries)
                if row_left_issue is not None:
                    return {}, has_rotated_90, row_left_issue

            for entry in cell_entries:
                page_boxes.append(
                    self._bbox_payload(
                        label=str(entry["label"]),
                        category_id=self._table_region_category_id(str(entry["label"])),
                        rect=entry["rect"],
                        extras={
                            "text": entry["text"] if entry["text"] is not None else "",
                            "row_id": entry["row_id"],
                            "col_id": entry["col_id"],
                            "row_span": entry["row_span"],
                            "col_span": entry["col_span"],
                        },
                    )
                )

        if not found_table:
            return {}, has_rotated_90, "Document has no table items."

        return bboxes_by_page, has_rotated_90, None

    def get_record_type(self) -> type[DatasetRecord]:
        return DatasetRecordWithBBox

    @staticmethod
    def _save_table_regions_visualization_html(
        *,
        filename: Path,
        page_visualizations: Dict[Optional[int], Image.Image],
    ) -> None:
        page_nos_int = sorted(
            page_no for page_no in page_visualizations if page_no is not None
        )
        page_nos: List[Optional[int]] = [*page_nos_int]
        if not page_nos and None in page_visualizations:
            page_nos = [None]

        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8' />",
            "<title>Table Regions Visualization</title>",
            "<style>",
            "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 16px; }",
            "h1 { margin-bottom: 16px; }",
            ".page-block { margin-bottom: 24px; }",
            ".page-title { margin: 0 0 8px 0; font-size: 18px; }",
            "img { max-width: 100%; height: auto; border: 1px solid #ddd; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Table Regions Visualization</h1>",
        ]

        for page_no in page_nos:
            page_img = page_visualizations.get(page_no)
            if page_img is None:
                page_img = get_missing_pageimg()
            image_b64 = from_pil_to_base64(page_img)
            page_label = "Unknown" if page_no is None else str(page_no)

            html_parts.extend(
                [
                    "<div class='page-block'>",
                    f"<h2 class='page-title'>Page {page_label}</h2>",
                    f"<img src='data:image/png;base64,{image_b64}' />",
                    "</div>",
                ]
            )

        html_parts.extend(["</body>", "</html>"])
        with open(filename, "w", encoding="utf-8") as file_handle:
            file_handle.write("\n".join(html_parts))

    def save_ground_truth_visualization(
        self,
        record: DatasetRecord,
        viz_path_split: Path,
    ) -> None:
        if self.modality != EvaluationModality.TABLE_REGIONS:
            super().save_ground_truth_visualization(record, viz_path_split)
            return

        # Use TableVisualizer output for table_regions instead of layout visualizer.
        table_doc = insert_images_from_pil(
            document=copy.deepcopy(record.ground_truth_doc),
            pictures=record.ground_truth_pictures,
            page_images=record.ground_truth_page_images,
        )
        table_visualizer = TableVisualizer(
            params=TableVisualizer.Params(
                show_cells=True,
                show_rows=False,
                show_cols=False,
                minimal_row_bboxes=False,
                minimal_col_bboxes=False,
            )
        )
        try:
            page_visualizations = table_visualizer.get_visualization(doc=table_doc)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "TableVisualizer failed for %s: %s. Falling back to default visualization.",
                record.doc_id,
                exc,
            )
            super().save_ground_truth_visualization(record, viz_path_split)
            return

        output_filename = viz_path_split.with_name(
            f"{viz_path_split.stem}_layout{viz_path_split.suffix}"
        )
        self._save_table_regions_visualization_html(
            filename=output_filename,
            page_visualizations=page_visualizations,
        )

    def iterate(self) -> Iterable[DatasetRecordWithBBox]:
        assert isinstance(self.dataset_source, Path)

        json_files = self._find_json_files()
        exact_png_index, paged_png_index = self._build_png_indices()

        begin, end = self.get_effective_indices(len(json_files))
        selected_json_files = json_files[begin:end]

        self.log_dataset_stats(len(json_files), len(selected_json_files))
        _log.info(
            "Processing DoclingSDG dataset with %d documents (modality=%s)",
            len(selected_json_files),
            self.modality.value,
        )

        processed_docs = 0
        exported_docs = 0
        skipped_load_errors = 0
        skipped_missing_png = 0
        skipped_png_errors = 0
        skipped_malformed = 0
        skipped_no_table = 0
        malformed_reason_counts: Dict[str, int] = defaultdict(int)

        for json_path in tqdm(
            selected_json_files,
            desc="Processing files for DoclingSDG",
            ncols=128,
        ):
            processed_docs += 1
            doc_id = json_path.stem

            try:
                document = DoclingDocument.load_from_json(json_path)
            except ValidationError as exc:
                _log.warning("Validation error for %s: %s. Skipping.", json_path, exc)
                skipped_load_errors += 1
                continue
            except Exception as exc:  # noqa: BLE001
                _log.warning("Failed to load %s: %s. Skipping.", json_path, exc)
                skipped_load_errors += 1
                continue

            png_files = self._find_png_files_for_doc(
                doc_id=doc_id,
                json_path=json_path,
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
                skipped_missing_png += 1
                continue

            try:
                page_images = self._load_png_images(png_files)
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "Failed to read PNG files for %s: %s. Skipping.", doc_id, exc
                )
                skipped_png_errors += 1
                continue

            self._attach_page_images(document, page_images)
            try:
                document, pictures, extracted_page_images = extract_images(
                    document=document,
                    pictures_column=BenchMarkColumns.GROUNDTRUTH_PICTURES.value,
                    page_images_column=BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES.value,
                )
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "Failed to extract page images for %s: %s. Skipping.", doc_id, exc
                )
                skipped_malformed += 1
                continue

            if len(extracted_page_images) == 0:
                extracted_page_images = page_images

            page_no_to_index = {
                page_no: idx
                for idx, page_no in enumerate(sorted(document.pages.keys()))
            }

            tags: List[str] = []
            if self.modality == EvaluationModality.TABLE_REGIONS:
                if len(document.tables) == 0:
                    reason = "Document has no table items."
                    _log.warning(
                        "Skipping malformed table sample %s: %s",
                        doc_id,
                        reason,
                    )
                    skipped_malformed += 1
                    skipped_no_table += 1
                    malformed_reason_counts[reason] = (
                        malformed_reason_counts.get(reason, 0) + 1
                    )
                    continue
                try:
                    (
                        _,
                        _,
                        malformed_reason,
                    ) = self._extract_table_region_bboxes(
                        document=document,
                        page_no_to_index=page_no_to_index,
                        page_images=extracted_page_images,
                    )
                except Exception as exc:  # noqa: BLE001
                    _log.warning(
                        "Skipping malformed table sample %s due pre-check extraction error: %s",
                        doc_id,
                        exc,
                    )
                    skipped_malformed += 1
                    malformed_reason_counts["table_region_precheck_error"] = (
                        malformed_reason_counts.get("table_region_precheck_error", 0)
                        + 1
                    )
                    continue
                if malformed_reason is not None:
                    _log.warning(
                        "Skipping malformed table sample %s: %s",
                        doc_id,
                        malformed_reason,
                    )
                    skipped_malformed += 1
                    if malformed_reason == "Document has no table items.":
                        skipped_no_table += 1
                    malformed_reason_counts[malformed_reason] = (
                        malformed_reason_counts.get(malformed_reason, 0) + 1
                    )
                    continue

                document_before_uniformization = copy.deepcopy(document)
                self._uniformize_tables_after_malformed_check(document)
                uniformized_extraction_exception: Optional[Exception] = None
                try:
                    (
                        ground_truth_bboxes,
                        has_rotated_90,
                        malformed_reason,
                    ) = self._extract_table_region_bboxes(
                        document=document,
                        page_no_to_index=page_no_to_index,
                        page_images=extracted_page_images,
                    )
                except Exception as exc:  # noqa: BLE001
                    malformed_reason = None
                    uniformized_extraction_exception = exc

                if (
                    uniformized_extraction_exception is not None
                    or malformed_reason is not None
                ):
                    if uniformized_extraction_exception is not None:
                        _log.warning(
                            "Uniformized extraction failed for %s (%s). Falling back to pre-uniformized document.",
                            doc_id,
                            uniformized_extraction_exception,
                        )
                    else:
                        _log.warning(
                            "Uniformized extraction produced malformed sample %s (%s). Falling back to pre-uniformized document.",
                            doc_id,
                            malformed_reason,
                        )

                    try:
                        (
                            ground_truth_bboxes,
                            has_rotated_90,
                            fallback_reason,
                        ) = self._extract_table_region_bboxes(
                            document=document_before_uniformization,
                            page_no_to_index=page_no_to_index,
                            page_images=extracted_page_images,
                        )
                    except Exception as exc:  # noqa: BLE001
                        _log.warning(
                            "Skipping malformed table sample %s: fallback extraction failed: %s",
                            doc_id,
                            exc,
                        )
                        skipped_malformed += 1
                        malformed_reason_counts[
                            "fallback_table_region_extraction_error"
                        ] = (
                            malformed_reason_counts.get(
                                "fallback_table_region_extraction_error", 0
                            )
                            + 1
                        )
                        continue

                    if fallback_reason is not None:
                        _log.warning(
                            "Skipping malformed table sample %s: fallback extraction still malformed: %s",
                            doc_id,
                            fallback_reason,
                        )
                        skipped_malformed += 1
                        malformed_reason_counts[f"fallback::{fallback_reason}"] = (
                            malformed_reason_counts.get(
                                f"fallback::{fallback_reason}", 0
                            )
                            + 1
                        )
                        continue

                    document = document_before_uniformization
                    malformed_reason_counts["uniformization_fallback_used"] = (
                        malformed_reason_counts.get("uniformization_fallback_used", 0)
                        + 1
                    )

                if has_rotated_90:
                    tags.append("90_degree")
            else:
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
                modalities=[self.modality],
                tags=tags,
            )
            exported_docs += 1

        skipped_total = processed_docs - exported_docs
        _log.info(
            (
                "DoclingSDG processing summary (modality=%s): "
                "processed=%d, exported=%d, skipped=%d "
                "(load_errors=%d, missing_png=%d, png_read_errors=%d, malformed=%d, no_table=%d)"
            ),
            self.modality.value,
            processed_docs,
            exported_docs,
            skipped_total,
            skipped_load_errors,
            skipped_missing_png,
            skipped_png_errors,
            skipped_malformed,
            skipped_no_table,
        )
        if malformed_reason_counts:
            _log.info(
                "DoclingSDG malformed reason counts: %s",
                dict(sorted(malformed_reason_counts.items())),
            )
