"""Convert CVAT DocumentStructure to DoclingDocument.

This module provides functionality to convert a populated DocumentStructure
from the CVAT parser into a DoclingDocument, handling text extraction via OCR
or PDF parsing, reading order, containment hierarchy, groups, merges, and 
caption/footnote relationships.

Coordinate System Invariants:
-----------------------------
- CVAT annotations: Always TOP_LEFT origin (pixel coordinates)
- PDF parser output: Always BOTTOM_LEFT origin (point coordinates at native 72 DPI)
- OCR output: Always TOP_LEFT origin (pixel coordinates)
- DoclingDocument provenance: Always BOTTOM_LEFT origin

Scale Handling:
--------------
- All processing happens at cvat_input_scale (matches CVAT annotation scale)
- SegmentedPages are loaded at cvat_input_scale
- Final scaling to storage_scale happens at the end in _scale_document_to_storage()
"""

import copy
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from docling_core.types.doc import RichTableCell, TableCell
from docling_core.types.doc.base import BoundingBox, CoordOrigin, ImageRefMode
from docling_core.types.doc.document import (
    ContentLayer,
    DocItem,
    DocItemLabel,
    DoclingDocument,
    FloatingItem,
    GraphCell,
    GraphData,
    GraphLink,
    GroupItem,
    GroupLabel,
    ImageRef,
    ListItem,
    NodeItem,
    PictureClassificationClass,
    PictureClassificationData,
    ProvenanceItem,
    Size,
    TableData,
)
from docling_core.types.doc.labels import GraphCellLabel, GraphLinkLabel
from docling_core.types.doc.page import (
    BoundingRectangle,
    Coord2D,
    PageGeometry,
    PdfPageGeometry,
    SegmentedPage,
    SegmentedPdfPage,
    TextCell,
    TextCellUnit,
)
from PIL import Image as PILImage

from docling_eval.cvat_tools.document import DocumentStructure
from docling_eval.cvat_tools.folder_models import CVATDocument, CVATFolderStructure
from docling_eval.cvat_tools.folder_parser import parse_cvat_folder
from docling_eval.cvat_tools.geometry import (
    bbox_contains,
    bbox_intersection,
    dedupe_items_by_bbox,
)
from docling_eval.cvat_tools.models import (
    CVATElement,
    CVATValidationError,
    CVATValidationReport,
    TableStructLabel,
    ValidationSeverity,
)
from docling_eval.cvat_tools.parser import (
    MissingImageInCVATXML,
    ParsedCVATFile,
    parse_cvat_file,
)
from docling_eval.cvat_tools.tree import (
    TreeNode,
    apply_reading_order_to_tree,
    build_global_reading_order,
    find_node_by_element_id,
)
from docling_eval.cvat_tools.validator import Validator, validate_cvat_sample
from docling_eval.utils.utils import classify_cells, sort_cell_ids

_logger = logging.getLogger(__name__)


class SemClass(str, Enum):
    COL_HEADER = "col_header"
    ROW_HEADER = "row_header"
    ROW_SECTION = "row_section"
    BODY = "body"


SEM_TO_TABLE_LABEL: dict[SemClass, TableStructLabel] = {
    SemClass.COL_HEADER: TableStructLabel.COL_HEADER,
    SemClass.ROW_HEADER: TableStructLabel.ROW_HEADER,
    SemClass.ROW_SECTION: TableStructLabel.ROW_SECTION,
    SemClass.BODY: TableStructLabel.BODY,
}

DEFAULT_TABLE_PAIR_IOU: float = 0.20
DEFAULT_CONTAINMENT_THRESH: float = 0.50
DEFAULT_SEM_MATCH_IOU: float = 0.30

pic_classes = {
    "BARCODE": "bar_code",
    "CHART": "chart",
    "DECORATION": "decoration",
    "ILLUSTRATION": "illustration",
    "INFOGRAPHIC": "infographic",
    "LOGO": "logo",
    "OTHER": "other",
    "PERSON": "person",
    "PICTOGRAM": "icon",
    "SCREENSHOT": "screenshot",
    "UI_ELEMENT": "ui_element",
}


@dataclass(frozen=True)
class Cell:
    start_row: int
    end_row: int
    start_column: int
    end_column: int
    row_span_length: int
    column_span_length: int
    bbox: BoundingBox  # cell bounding box
    column_header: bool = False
    row_header: bool = False
    row_section: bool = False
    fillable_cell: bool = False


def is_bbox_within(
    bbox_a: BoundingBox, bbox_b: BoundingBox, threshold: float = 0.5
) -> bool:
    """Return ``True`` when ``bbox_b`` lies within ``bbox_a`` above ``threshold``."""

    return bbox_contains(bbox_b, bbox_a, threshold=threshold)


def compute_cells(
    rows: List[CVATElement],
    columns: List[CVATElement],
    merges: List[CVATElement],
    col_headers: List[CVATElement],
    row_headers: List[CVATElement],
    row_sections: List[CVATElement],
    fillable_cells: List[CVATElement],
    row_overlap_threshold: float = 0.5,  # how much of a row a merge must cover vertically
    col_overlap_threshold: float = 0.5,  # how much of a column a merge must cover horizontally
) -> List[Cell]:
    """
    rows, columns, merges are lists of BoundingBox(l,t,r,b).
    Returns 0-based indexed Cells. Merged cells are aligned to grid boundaries.
    """
    rows.sort(key=lambda r: (r.bbox.t + r.bbox.b) / 2.0)
    columns.sort(key=lambda c: (c.bbox.l + c.bbox.r) / 2.0)

    n_rows, n_cols = len(rows), len(columns)

    def span_from_merge(
        m: BoundingBox, lines: List[CVATElement], axis: str, frac_threshold: float
    ) -> Optional[Tuple[int, int]]:
        """
        Map a merge bbox to an inclusive index span over rows or columns.
        axis='row' uses vertical overlap vs row height; axis='col' uses horizontal overlap vs col width.
        If nothing meets threshold, pick the single best-overlapping line if overlap>0; else return None.
        """
        idxs = []
        best_i, best_len = None, 0.0
        for i, elem in enumerate(lines):
            inter = bbox_intersection(m, elem.bbox)
            if not inter:
                continue
            if axis == "row":
                overlap_len = inter.height
                base = max(1e-9, elem.bbox.height)
            else:
                overlap_len = inter.width
                base = max(1e-9, elem.bbox.width)

            frac = overlap_len / base
            if frac >= frac_threshold:
                idxs.append(i)

            if overlap_len > best_len:
                best_len, best_i = overlap_len, i

        if idxs:
            return min(idxs), max(idxs)
        if best_i is not None and best_len > 0.0:
            return best_i, best_i
        return None

    cells: List[Cell] = []
    covered: Set[Tuple[int, int]] = set()
    seen_merge_rects: Set[Tuple[int, int, int, int]] = set()

    # 1) Add merged cells first (and mark their covered simple cells)
    for m in merges:
        rspan = span_from_merge(
            m.bbox, rows, axis="row", frac_threshold=row_overlap_threshold
        )
        cspan = span_from_merge(
            m.bbox, columns, axis="col", frac_threshold=col_overlap_threshold
        )
        if rspan is None or cspan is None:
            # Can't confidently map this merge to grid -> skip it
            continue

        sr, er = rspan
        sc, ec = cspan
        rect_key = (sr, er, sc, ec)
        if rect_key in seen_merge_rects:
            continue
        seen_merge_rects.add(rect_key)

        # Grid-aligned bbox for the merged cell
        grid_bbox = BoundingBox(
            l=columns[sc].bbox.l,
            t=rows[sr].bbox.t,
            r=columns[ec].bbox.r,
            b=rows[er].bbox.b,
        )
        c_column_header, c_row_header, c_row_section, c_fillable = (
            process_table_headers(
                grid_bbox, col_headers, row_headers, row_sections, fillable_cells
            )
        )

        cells.append(
            Cell(
                start_row=sr,
                end_row=er,
                start_column=sc,
                end_column=ec,
                row_span_length=er - sr + 1,
                column_span_length=ec - sc + 1,
                bbox=grid_bbox,
                column_header=c_column_header,
                row_header=c_row_header,
                row_section=c_row_section,
                fillable_cell=c_fillable,
            )
        )

        for ri in range(sr, er + 1):
            for ci in range(sc, ec + 1):
                covered.add((ri, ci))

    # 2) Add simple (1x1) cells where not covered by merges
    for ri, row in enumerate(rows):
        for ci, col in enumerate(columns):
            if (ri, ci) in covered:
                continue
            inter = bbox_intersection(row.bbox, col.bbox)
            if not inter:
                # In degenerate cases (big gaps), there might be no intersection; skip.
                continue
            c_column_header, c_row_header, c_row_section, c_fillable = (
                process_table_headers(
                    inter, col_headers, row_headers, row_sections, fillable_cells
                )
            )
            cells.append(
                Cell(
                    start_row=ri,
                    end_row=ri,
                    start_column=ci,
                    end_column=ci,
                    row_span_length=1,
                    column_span_length=1,
                    bbox=inter,
                    column_header=c_column_header,
                    row_header=c_row_header,
                    row_section=c_row_section,
                    fillable_cell=c_fillable,
                )
            )
    return cells


def process_table_headers(
    bbox: BoundingBox,
    col_headers: List[CVATElement],
    row_headers: List[CVATElement],
    row_sections: List[CVATElement],
    fillable_cells: List[CVATElement],
) -> Tuple[bool, bool, bool, bool]:
    c_column_header = False
    c_row_header = False
    c_row_section = False
    c_fillable = False

    for col_header in col_headers:
        if is_bbox_within(col_header.bbox, bbox):
            c_column_header = True
    for row_header in row_headers:
        if is_bbox_within(row_header.bbox, bbox):
            c_row_header = True
    for row_section in row_sections:
        if is_bbox_within(row_section.bbox, bbox):
            c_row_section = True
    for fillable_cell in fillable_cells:
        if is_bbox_within(fillable_cell.bbox, bbox):
            c_fillable = True
    return c_column_header, c_row_header, c_row_section, c_fillable


class ListHierarchyManager:
    """Manages list hierarchy creation and tracking.

    Consolidates the responsibility of managing list containers, sublists,
    and level tracking that was previously scattered across multiple data structures.
    """

    def __init__(self, doc: DoclingDocument):
        self.doc = doc

        # Single source of truth for all group containers
        self.group_containers: Dict[int, NodeItem] = {}  # path_id -> container

        # Track level hierarchy for nested lists
        self.level_stack: Dict[int, ListItem] = {}  # level -> most recent list item

        # Track sublist containers for parent items
        self.sublist_containers: Dict[str, NodeItem] = (
            {}
        )  # parent_ref -> sublist_container

    def clear(self):
        """Reset all list hierarchy state."""
        self.group_containers.clear()
        self.level_stack.clear()
        self.sublist_containers.clear()

    def get_or_create_list_container(
        self,
        group_id: Optional[int],
        element: CVATElement,
        group_parent_finder,
        existing_groups: Optional[Dict[int, GroupItem]] = None,
    ) -> NodeItem:
        """Get or create a list container for top-level list items."""
        if group_id is not None:
            if group_id not in self.group_containers:
                # Check if group already exists in existing_groups (for mixed content groups)
                if existing_groups and group_id in existing_groups:
                    self.group_containers[group_id] = existing_groups[group_id]
                else:
                    group_parent = group_parent_finder(element)
                    self.group_containers[group_id] = self.doc.add_group(
                        label=GroupLabel.LIST,
                        name=f"group_{group_id}",
                        parent=group_parent,
                    )
                    # Sync back to existing_groups if provided
                    if existing_groups:
                        # GroupItem is a subclass of NodeItem, so this is safe
                        existing_groups[group_id] = self.group_containers[group_id]  # type: ignore
            return self.group_containers[group_id]
        else:
            # Create standalone list container
            group_parent = group_parent_finder(element)
            return self.doc.add_group(
                label=GroupLabel.LIST,
                name=f"list_standalone_{element.id}",
                parent=group_parent,
            )

    def get_or_create_sublist_container(
        self, level: int, element: CVATElement, group_parent_finder, doc_structure=None
    ) -> Optional[NodeItem]:
        """Get or create a sublist container for nested list items."""
        if (level - 1) not in self.level_stack:
            # Orphaned item - log warning and create fallback
            _logger.warning(
                f"Orphaned list item {element.id} at level {level} "
                f"has no parent at level {level-1}. Creating fallback structure."
            )
            # Try to find group_id for fallback
            group_id = (
                self._find_group_id_for_element(element, doc_structure)
                if doc_structure
                else None
            )
            return self.get_or_create_list_container(
                group_id, element, group_parent_finder, None
            )

        parent_list_item = self.level_stack[level - 1]
        parent_ref = parent_list_item.self_ref

        if parent_ref not in self.sublist_containers:
            # Find the container that holds the parent list item
            try:
                parent_container = (
                    parent_list_item.parent.resolve(self.doc)
                    if parent_list_item.parent
                    else None
                )
            except Exception as e:
                _logger.warning(
                    f"Failed to resolve parent for list item {element.id}: {e}"
                )
                parent_container = None

            # Create sublist group at the same level as the parent list item
            self.sublist_containers[parent_ref] = self.doc.add_group(
                label=GroupLabel.LIST,
                name=f"sublist_level_{level}_of_{parent_ref.split('/')[-1]}",
                parent=parent_container,
            )

        return self.sublist_containers[parent_ref]

    def update_level_stack(self, level: int, list_item: ListItem):
        """Update the level stack and clear higher levels."""
        self.level_stack[level] = list_item

        # Clear higher levels from stack since they're now out of scope
        levels_to_remove = [l for l in self.level_stack if l > level]
        for l in levels_to_remove:
            self.level_stack.pop(l, None)

    def _find_group_id_for_element(
        self, element: CVATElement, doc_structure
    ) -> Optional[int]:
        """Helper to find group ID for an element (for fallback scenarios)."""
        for path_id, group_element_ids in doc_structure.path_mappings.group.items():
            if element.id in group_element_ids:
                return path_id
        return None


class CVATToDoclingConverter:
    """Converts CVAT DocumentStructure to DoclingDocument.

    Caption and Footnote Support:
    ----------------------------
    The following containers (FloatingItem subclasses) can have captions and footnotes:
    - TableItem (DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX)
    - PictureItem (DocItemLabel.PICTURE, DocItemLabel.HANDWRITTEN_TEXT)
    - CodeItem (DocItemLabel.CODE)
    - FormItem (DocItemLabel.FORM)
    - KeyValueItem (DocItemLabel.KEY_VALUE_REGION)

    Caption/footnote relationships support both directions:
    - Forward: container -> caption/footnote (standard case)
    - Backward: caption/footnote -> container (handles annotation errors)

    The converter is resilient to caption elements being labeled as TEXT instead
    of CAPTION in the CVAT annotations - it will create the proper DocItemLabel.CAPTION
    items regardless of the original label.
    """

    def __init__(
        self,
        doc_structure: DocumentStructure,
        segmented_pages: Dict[int, SegmentedPage],
        page_images: Dict[int, PILImage.Image],
        document_filename: Optional[str] = None,
        cvat_input_scale: float = 2.0,
        storage_scale: float = 2.0,
    ):
        """Initialize the converter.

        Args:
            doc_structure: The populated DocumentStructure from CVAT parser
            segmented_pages: Dictionary mapping page numbers to SegmentedPage objects (at cvat_input_scale)
            page_images: Dictionary mapping page numbers to PIL images (at cvat_input_scale)
            document_filename: Optional filename for the document
            cvat_input_scale: Scale at which CVAT annotations and SegmentedPages are provided
            storage_scale: Scale for final stored coordinates in DoclingDocument
        """
        self.doc_structure = doc_structure
        self.segmented_pages = segmented_pages
        self.page_images = page_images
        self.document_filename = document_filename or "document"
        self.num_pages = len(segmented_pages)

        # SegmentedPages are at cvat_input_scale, we'll scale to storage_scale at the end
        self.cvat_input_scale = cvat_input_scale
        self.storage_scale = storage_scale

        # Initialize empty DoclingDocument
        self.doc = DoclingDocument(name=Path(self.document_filename).stem)

        # Maps for tracking created items
        self.element_to_item: Dict[int, Optional[NodeItem]] = {}
        self.processed_elements: Set[int] = set()

        # Track which groups have been created
        self.created_groups: Dict[int, GroupItem] = {}  # path_id -> GroupItem

        # Centralized list hierarchy management
        self.list_manager = ListHierarchyManager(self.doc)

        # Calculate page widths for multi-page handling (in CVAT pixel coordinates at cvat_input_scale)
        # SegmentedPages are already at cvat_input_scale, so we can use their dimensions directly
        self.page_widths = {}
        cumulative_width = 0.0

        # To track tables, structures and rich cell elements
        self.tabular_data: List[object] = []

        # Use CVAT image info dimensions if available, otherwise calculate from pages
        if self.doc_structure.image_info:
            # For multi-page documents, CVAT concatenates pages horizontally
            total_cvat_width = self.doc_structure.image_info.width
            if self.num_pages > 1:
                # Distribute width proportionally based on page widths at cvat_input_scale
                total_page_width = sum(
                    p.dimension.width for p in segmented_pages.values()
                )
                for page_no in sorted(segmented_pages.keys()):
                    self.page_widths[page_no] = cumulative_width
                    page_width_ratio = (
                        segmented_pages[page_no].dimension.width / total_page_width
                    )
                    page_width_cvat_pixels = total_cvat_width * page_width_ratio
                    cumulative_width += page_width_cvat_pixels
            else:
                # Single page
                self.page_widths[1] = 0.0
        else:
            # Fallback: use page dimensions directly (already at cvat_input_scale)
            for page_no in sorted(segmented_pages.keys()):
                self.page_widths[page_no] = cumulative_width
                cumulative_width += segmented_pages[page_no].dimension.width

    def _process_element_bbox(
        self, element: CVATElement
    ) -> Tuple[int, str, ProvenanceItem]:
        """Process element bbox to extract page, text, and create provenance.

        All processing happens at cvat_input_scale (same as SegmentedPages).
        Scaling to storage_scale happens later in convert().

        Coordinate system invariant:
        - CVAT bboxes: Always TOP_LEFT origin
        - DoclingDocument provenance: Always BOTTOM_LEFT origin

        Returns:
            Tuple of (page_no, text, provenance_item in BOTTOM_LEFT at cvat_input_scale)
        """
        # Get page number from bbox position (bbox is in CVAT TOP_LEFT coordinates at cvat_input_scale)
        page_no = self._get_page_number_from_bbox(element.bbox)

        # Extract text (SegmentedPage is also at cvat_input_scale, coordinates match perfectly)
        text = self._extract_text_from_bbox(element.bbox, page_no)

        # Adjust bbox for multi-page (still in CVAT TOP_LEFT coordinates at cvat_input_scale)
        adjusted_bbox = self._adjust_bbox_for_page(element.bbox, page_no)

        # Convert from CVAT TOP_LEFT to DoclingDocument BOTTOM_LEFT (still at cvat_input_scale)
        seg_page = self.segmented_pages[page_no]
        page_height = seg_page.dimension.height
        prov_bbox = adjusted_bbox.to_bottom_left_origin(page_height)

        # Create provenance (BOTTOM_LEFT at cvat_input_scale - will be scaled later)
        provenance = ProvenanceItem(
            page_no=page_no, bbox=prov_bbox, charspan=(0, len(text))
        )

        return page_no, text, provenance

    def convert(self) -> DoclingDocument:
        """Convert the DocumentStructure to DoclingDocument.

        Returns:
            The converted DoclingDocument with coordinates scaled to storage_scale
        """
        # Reset list processing state to ensure clean conversion
        self._reset_list_state()

        # Add pages to document
        self._add_pages()

        # Apply reading order to tree
        self._apply_reading_order()

        # Build global reading order
        global_order = self._build_global_reading_order()

        # Process elements in reading order, building list hierarchy on-demand
        self._process_elements_in_order(global_order)

        # Process table data
        self._process_table_data()

        # Process captions and footnotes
        self._process_captions_and_footnotes()

        # Process to_value relationships
        self._process_to_value_relationships()

        # Remove groups left without any children during conversion
        self._prune_empty_groups()

        # Scale document coordinates from cvat_input_scale to storage_scale
        if self.storage_scale != self.cvat_input_scale:
            self._scale_document_to_storage()

        return self.doc

    def _scale_document_to_storage(self):
        """Scale all coordinates in the document from cvat_input_scale to storage_scale.

        This is the final step that transforms coordinates from the CVAT annotation scale
        to the desired storage scale for the DoclingDocument.
        """
        scale_factor = self.storage_scale / self.cvat_input_scale

        # Scale page sizes and images
        for page_no, page_item in self.doc.pages.items():
            page_item.size = Size(
                width=page_item.size.width * scale_factor,
                height=page_item.size.height * scale_factor,
            )
            # Scale and resize actual image if present
            if page_item.image and page_no in self.page_images:
                original_image = self.page_images[page_no]
                new_width = int(original_image.width * scale_factor)
                new_height = int(original_image.height * scale_factor)

                # Resize the image
                resized_image = original_image.resize(
                    (new_width, new_height), PILImage.Resampling.LANCZOS
                )

                # Create new ImageRef with resized image
                # DPI = 72 * storage_scale (since storage_scale is relative to base 72 DPI)
                page_item.image = ImageRef.from_pil(
                    resized_image, dpi=int(72 * self.storage_scale)
                )

        # Scale all item provenances
        for item, _ in self.doc.iterate_items(
            traverse_pictures=True, included_content_layers=set(ContentLayer)
        ):
            if isinstance(item, DocItem):
                for prov in item.prov:
                    prov.bbox = prov.bbox.scaled(scale_factor)

        # Scale graph cell provenances (for key-value items)
        for item in self.doc.key_value_items:
            if item.graph and item.graph.cells:
                for cell in item.graph.cells:
                    if cell.prov:
                        cell.prov.bbox = cell.prov.bbox.scaled(scale_factor)

        # Scale table cell bboxes
        for table_item in self.doc.tables:
            if table_item.data and table_item.data.table_cells:
                for cell in table_item.data.table_cells:
                    if isinstance(cell, TableCell) and cell.bbox:
                        cell.bbox = cell.bbox.scaled(scale_factor)

    def _reset_list_state(self):
        """Reset list processing state for clean conversion."""
        self.list_manager.clear()

    def _add_pages(self):
        """Add page information to the document."""
        for page_no, seg_page in self.segmented_pages.items():
            # All SegmentedPages (PDF or image) have the same dimension interface
            page_size = Size(
                width=seg_page.dimension.width,
                height=seg_page.dimension.height,
            )

            # Create image reference if available
            image_ref = None
            if page_no in self.page_images:
                image_ref = ImageRef.from_pil(self.page_images[page_no], dpi=72)

            # Add page
            self.doc.add_page(page_no=page_no, size=page_size, image=image_ref)

    def _apply_reading_order(self):
        """Apply reading order to the containment tree."""
        # Get all reading order element mappings
        reading_order_mappings = self.doc_structure.path_mappings.reading_order

        # Combine all reading order elements into a global order
        all_ordered_elements = []
        for path_id, element_ids in reading_order_mappings.items():
            for el_id in element_ids:
                if el_id not in all_ordered_elements:
                    all_ordered_elements.append(el_id)

        # Apply to tree
        apply_reading_order_to_tree(self.doc_structure.tree_roots, all_ordered_elements)

    def _build_global_reading_order(self) -> List[int]:
        """Build global reading order from paths."""
        return build_global_reading_order(
            self.doc_structure.paths,
            self.doc_structure.path_mappings.reading_order,
            self.doc_structure.path_to_container,
            self.doc_structure.tree_roots,
        )

    def _get_group_for_element(
        self, element_id: int
    ) -> Optional[Tuple[int, List[int]]]:
        """Check if element is part of a group and return group info.

        Returns:
            Tuple of (path_id, element_ids) if element is in a group, None otherwise
        """

        for path_id, element_ids in self.doc_structure.path_mappings.group.items():
            if element_id in element_ids and len(element_ids) >= 2:
                return (path_id, element_ids)
        return None

    def _create_group_on_demand(
        self, path_id: int, element_ids: List[int], parent: Optional[NodeItem]
    ) -> NodeItem:
        """Create a group when first encountered."""
        # Check if already created
        if path_id in self.created_groups:
            return self.created_groups[path_id]

        # Determine group label based on contained elements
        group_label = self._determine_group_label(element_ids)

        # Create group with proper parent
        group = self.doc.add_group(
            label=group_label, name=f"group_{path_id}", parent=parent
        )

        # Track that we created this group
        self.created_groups[path_id] = group

        return group

    def _determine_group_label(self, element_ids: List[int]) -> GroupLabel:
        """Determine appropriate group label based on elements."""
        labels = set()
        for el_id in element_ids:
            element = self.doc_structure.get_element_by_id(el_id)
            if element:
                labels.add(element.label)

        # If all elements are pictures, use PICTURE_AREA
        if len(labels) == 1 and DocItemLabel.PICTURE in labels:
            return GroupLabel.PICTURE_AREA
        # If all elements are list items, use LIST
        elif len(labels) == 1 and DocItemLabel.LIST_ITEM in labels:
            return GroupLabel.LIST
        # If contains form elements
        elif any(
            label in [DocItemLabel.CHECKBOX_SELECTED, DocItemLabel.CHECKBOX_UNSELECTED]
            for label in labels
        ):
            return GroupLabel.FORM_AREA
        else:
            return GroupLabel.UNSPECIFIED

    def _find_group_parent(self, element: CVATElement) -> Optional[NodeItem]:
        """Find the parent for a list group based on containment tree."""
        node = find_node_by_element_id(self.doc_structure.tree_roots, element.id)
        if node:
            parent_node = self._find_parent_node(node)
            if parent_node and parent_node.element.id in self.element_to_item:
                return self.element_to_item[parent_node.element.id]
        return None

    def _process_list_item_with_hierarchy(
        self,
        element: CVATElement,
        global_order: List[int],
        current_position: Optional[int],
    ) -> Optional[ListItem]:
        """Process a list item with proper hierarchy based on level."""
        level = element.level or 1

        # Find which group this element belongs to
        group_id = self._find_group_id_for_element(element)

        actual_parent: Optional[NodeItem] = None
        # Determine the appropriate parent for this list item
        if level == 1:
            # Top-level item - needs a list container (group)
            actual_parent = self.list_manager.get_or_create_list_container(
                group_id, element, self._find_group_parent, self.created_groups
            )
        else:
            # Nested item - create sublist group
            actual_parent = self.list_manager.get_or_create_sublist_container(
                level, element, self._find_group_parent, self.doc_structure
            )
            if actual_parent is None:
                return None

        # Create the list item
        page_no, text, provenance = self._process_element_bbox(element)
        list_item = self.doc.add_list_item(
            text=text,
            prov=provenance,
            parent=actual_parent,
            content_layer=element.content_layer,
        )

        # Update level stack
        self.list_manager.update_level_stack(level, list_item)

        # Process logical child text elements immediately if position is available
        if current_position is not None:
            child_elements = self._find_logical_children_for_list_item(
                element, global_order, current_position
            )
            for child_element in child_elements:
                # Create child item with list_item as parent
                child_item = self._create_single_item(child_element, list_item)
                if child_item:
                    # Mark as processed to prevent duplicate processing
                    self.processed_elements.add(child_element.id)
                    self.element_to_item[child_element.id] = child_item

        return list_item

    def _find_group_id_for_element(self, element: CVATElement) -> Optional[int]:
        """Find which group this element belongs to."""
        for (
            path_id,
            group_element_ids,
        ) in self.doc_structure.path_mappings.group.items():
            if element.id in group_element_ids:
                return path_id
        return None

    def _prune_empty_groups(self) -> None:
        """Remove group containers that ended up without children."""
        empty_groups: List[GroupItem] = []

        for item, _ in self.doc.iterate_items(with_groups=True):
            if (
                isinstance(item, GroupItem)
                and not item.children
                and item.parent is not None
            ):
                empty_groups.append(item)

        if not empty_groups:
            return

        self.doc.delete_items(node_items=empty_groups)

        # Keep local bookkeeping in sync for any removed groups
        for path_id, group in list(self.created_groups.items()):
            if group in empty_groups:
                self.created_groups.pop(path_id)

    def _find_logical_children_for_list_item(
        self, list_element: CVATElement, global_order: List[int], current_pos: int
    ) -> List[CVATElement]:
        """Find elements that should be children of this list item.

        Looks ahead in reading order to find elements between this list item
        and the next list item, stopping at group/structural boundaries.

        Args:
            list_element: The list item element to find children for
            global_order: The global reading order
            current_pos: Current position in the global_order

        Returns:
            List of elements that should be children of this list item
        """
        group_id = self._find_group_id_for_element(list_element)
        list_content_layer = list_element.content_layer

        children = []
        # Look ahead in reading order
        for i in range(current_pos + 1, len(global_order)):
            next_element_id = global_order[i]
            next_element = self.doc_structure.get_element_by_id(next_element_id)

            if not next_element:
                continue

            # Stop if we hit another list item (any list item, any group)
            if next_element.label == DocItemLabel.LIST_ITEM:
                break

            # Stop if we hit structural elements that indicate new sections
            if next_element.label in [
                DocItemLabel.SECTION_HEADER,
                DocItemLabel.TITLE,
                DocItemLabel.TABLE,
                DocItemLabel.PICTURE,
                DocItemLabel.FORM,
            ]:
                break

            # Collect any elements (except list items) on the same content layer
            if next_element.content_layer == list_content_layer:

                next_group_id = self._find_group_id_for_element(next_element)

                # If we have a group, only include elements from same group or no group
                if group_id is not None:
                    if next_group_id is None or next_group_id == group_id:
                        children.append(next_element)
                    else:
                        # Different group means we've moved to a new section
                        break
                else:
                    # If list item has no group, only include elements with no group
                    if next_group_id is None:
                        children.append(next_element)
                    else:
                        # Element has a group but list doesn't - stop here
                        break

        return children

    def _process_elements_in_order(self, global_order: List[int]):
        """Process elements in reading order."""
        # Process elements in global reading order
        for i, element_id in enumerate(global_order):
            # Skip if already processed
            if element_id in self.processed_elements:
                continue

            # Find the node containing this element
            node = find_node_by_element_id(self.doc_structure.tree_roots, element_id)
            if node:
                self._process_node(
                    node,
                    None,
                    parent_item=None,
                    global_order=global_order,
                    current_position=i,
                )

    def _process_table_data(self):
        # After all CVAT elements have been processed,
        # go over tables and populate them with cells
        # This includes rich cell elements

        for tind, table_item in enumerate(self.doc.tables):
            # table_item.children - would be rich elements for each table cell
            table_cell_data = self.tabular_data[tind]["computed_table_cells"]
            page_no = self.tabular_data[tind]["page_no"]
            page_height = self.doc.pages[page_no].size.height

            table_item.children = []
            all_items = []
            for item, _ in self.doc.iterate_items(page_no=page_no):
                all_items.append(item.get_ref())

            for c in table_cell_data:
                # Get page number from bbox position
                # Get text to populate TableData
                cell_text = self._extract_text_from_bbox(c.bbox, page_no)

                # Define if cell is Rich
                rich_cell = False
                provs_in_cell = []

                # Convert cell bbox to BOTTOM_LEFT once (provs are in BOTTOM_LEFT)
                cell_bbox_bl = c.bbox.to_bottom_left_origin(page_height)

                # FIND RICH ELEMENTS REFs HERE, MAKE A GROUP IF MANY
                for item_ref in all_items:
                    item = item_ref.resolve(self.doc)

                    # Skip the table itself - it should never be processed as cell content
                    if item == table_item:
                        continue

                    if isinstance(item, DocItem):
                        for prov in item.prov:
                            # Both are now in BOTTOM_LEFT, no conversion needed
                            if is_bbox_within(cell_bbox_bl, prov.bbox):
                                # At least one child is inside the cell!
                                rich_cell = True
                                item_parent = (
                                    item.parent.resolve(self.doc)
                                    if item.parent
                                    else None
                                )
                                # Only remove from parent if parent is not the table itself
                                if (
                                    item_parent
                                    and item_parent != table_item
                                    and item.get_ref() in item_parent.children
                                ):
                                    item_parent.children.remove(item.get_ref())
                                item.parent = table_item.get_ref()
                                provs_in_cell.append(item.get_ref())
                if rich_cell:
                    # Get Ref
                    ref_for_rich_cell = provs_in_cell[0]

                    if len(provs_in_cell) > 1:
                        group_element = self.doc.add_group(
                            label=GroupLabel.UNSPECIFIED,
                            name="rich_cell_group_{}_{}_{}".format(
                                tind, c.start_column, c.start_row
                            ),
                            parent=table_item,
                        )
                        for pr in provs_in_cell:
                            group_element.children.append(pr)
                            pr_item = pr.resolve(self.doc)
                            pr_item.parent = group_element.get_ref()
                        ref_for_rich_cell = group_element.get_ref()

                    cell = RichTableCell(
                        # bbox=c.bbox,
                        text=cell_text,
                        row_span=c.row_span_length,
                        col_span=c.column_span_length,
                        start_row_offset_idx=c.start_row,
                        end_row_offset_idx=c.start_row + c.row_span_length,
                        start_col_offset_idx=c.start_column,
                        end_col_offset_idx=c.start_column + c.column_span_length,
                        column_header=c.column_header,
                        row_header=c.row_header,
                        row_section=c.row_section,
                        fillable=c.fillable_cell,
                        ref=ref_for_rich_cell,  # points to an artificial group around children, or to child directly
                    )
                    self.doc.add_table_cell(table_item=table_item, cell=cell)
                else:
                    cell = TableCell(
                        bbox=c.bbox,
                        text=cell_text,
                        row_span=c.row_span_length,
                        col_span=c.column_span_length,
                        start_row_offset_idx=c.start_row,
                        end_row_offset_idx=c.start_row + c.row_span_length,
                        start_col_offset_idx=c.start_column,
                        end_col_offset_idx=c.start_column + c.column_span_length,
                        column_header=c.column_header,
                        row_header=c.row_header,
                        fillable=c.fillable_cell,
                        row_section=c.row_section,
                    )
                    self.doc.add_table_cell(table_item=table_item, cell=cell)

    def _find_parent_node(self, node: TreeNode) -> Optional[TreeNode]:
        """Find the parent node of a given node in the tree."""

        def search_parent(current: TreeNode, target: TreeNode) -> Optional[TreeNode]:
            for child in current.children:
                if child == target:
                    return current
                result = search_parent(child, target)
                if result:
                    return result
            return None

        for root in self.doc_structure.tree_roots:
            if root == node:
                return None  # Root has no parent
            result = search_parent(root, node)
            if result:
                return result
        return None

    def _process_node(
        self,
        node: TreeNode,
        parentTreeNode: Optional[TreeNode],
        parent_item: Optional[NodeItem],
        global_order: List[int],
        current_position: Optional[int] = None,
    ):
        """Process a tree node and its children."""
        element = node.element

        # Skip if already processed
        if element.id in self.processed_elements:
            return

        # Skip if this element is handled by caption/footnote relationships
        if self._is_caption_or_footnote_target(element.id):
            return

        # Handle list items with hierarchy
        if element.label == DocItemLabel.LIST_ITEM:
            list_item = self._process_list_item_with_hierarchy(
                element, global_order, current_position
            )
            if list_item:
                self.element_to_item[element.id] = list_item
                self.processed_elements.add(element.id)
            return

        # Determine the actual parent based on containment tree
        if parent_item is None:
            parent_node = self._find_parent_node(node)
            if parent_node and parent_node.element.id in self.element_to_item:
                parent_item = self.element_to_item[parent_node.element.id]

        # Check if this element is part of a group
        group_info = self._get_group_for_element(element.id)

        item_parent: Optional[NodeItem] = None
        if group_info:
            path_id, element_ids = group_info
            # Create group on demand if not already created
            group = self._create_group_on_demand(path_id, element_ids, parent_item)
            # Use the group as the parent for this element
            item_parent = group
        else:
            # Use the regular parent
            item_parent = parent_item

        # Check if this element is part of a merge
        merge_elements = self._get_merge_elements(element.id)

        item: Optional[NodeItem] = None
        if merge_elements:
            # Process as merged item
            item = self._create_merged_item(merge_elements, item_parent)
            # Mark all merged elements as processed
            for el in merge_elements:
                self.processed_elements.add(el.id)
                if el.id not in self.element_to_item:
                    self.element_to_item[el.id] = item
        else:
            # Process as single item
            item = self._create_single_item(element, item_parent)
            self.processed_elements.add(element.id)
            self.element_to_item[element.id] = item

        # Process children in order
        if node.children:
            # Sort children by their position in global order
            sorted_children = sorted(
                node.children,
                key=lambda child: (
                    global_order.index(child.element.id)
                    if child.element.id in global_order
                    else float("inf")
                ),
            )

            for child in sorted_children:
                self._process_node(
                    child, node, item, global_order, current_position=None
                )

    def _has_valid_container_relationship(
        self, element_id_1: int, element_id_2: int
    ) -> bool:
        """Check if at least one element in a relationship is a container.

        Returns True if at least one side of the relationship is a container element.
        """
        from .utils import is_container_element

        element_1 = self.doc_structure.get_element_by_id(element_id_1)
        element_2 = self.doc_structure.get_element_by_id(element_id_2)

        if element_1 and element_2:
            return is_container_element(element_1) or is_container_element(element_2)

        return False

    def _is_caption_or_footnote_target(self, element_id: int) -> bool:
        """Check if element is a target of a VALID caption/footnote relationship.

        Only returns True if the element is a target AND at least one side of the
        relationship is a container element. This prevents non-container elements
        from being skipped when they're part of invalid caption/footnote paths.
        """
        # Check captions
        for path_id, (
            container_id,
            caption_id,
        ) in self.doc_structure.path_mappings.to_caption.items():
            if caption_id == element_id:
                if self._has_valid_container_relationship(container_id, caption_id):
                    return True

        # Check footnotes
        for path_id, (
            container_id,
            footnote_id,
        ) in self.doc_structure.path_mappings.to_footnote.items():
            if footnote_id == element_id:
                if self._has_valid_container_relationship(container_id, footnote_id):
                    return True

        return False

    def _get_merge_elements(self, element_id: int) -> List[CVATElement]:
        """Get all elements that should be merged with the given element.

        Auto-corrects backwards merge paths by sorting elements according to reading order.
        """
        for path_id, element_ids in self.doc_structure.path_mappings.merge.items():
            if element_id not in element_ids:
                continue

            # Auto-correct the order before collecting elements
            corrected_ids, _ = self.doc_structure.get_corrected_merge_elements(
                path_id, element_ids
            )

            # Collect unprocessed elements in corrected order
            merge_elements = []
            for el_id in corrected_ids:
                if el_id not in self.processed_elements:
                    element = self.doc_structure.get_element_by_id(el_id)
                    if element:
                        merge_elements.append(element)

            return merge_elements

        return []

    def _create_merged_item(
        self, elements: List[CVATElement], parent: Optional[NodeItem]
    ) -> Optional[NodeItem]:
        """Create a single DocItem from multiple merged elements."""
        if not elements:
            return None

        # Use first element as primary
        primary_element = elements[0]

        # Extract text and provenance from all elements
        all_texts = []
        all_provs = []

        for i, element in enumerate(elements):
            page_no, text, prov = self._process_element_bbox(element)
            all_texts.append(text)

            # Update character span for merged text
            start_char = sum(len(t) + 1 for t in all_texts[:-1]) if i > 0 else 0
            end_char = start_char + len(text)
            prov.charspan = (start_char, end_char)

            all_provs.append(prov)

        # Concatenate text
        merged_text = " ".join(all_texts)

        if isinstance(primary_element.label, DocItemLabel):
            # Create item based on label
            item = self._create_item_by_label(
                primary_element.label,
                merged_text,
                all_provs[0],
                primary_element,
                parent,
            )
        else:
            # Return None if label is not a DocItemLabel
            return None

        # Add additional provenances
        if item and len(all_provs) > 1:
            item.prov.extend(all_provs[1:])

        return item

    def _create_single_item(
        self, element: CVATElement, parent: Optional[NodeItem]
    ) -> Optional[NodeItem]:
        """Create a DocItem for a single element."""
        page_no, text, provenance = self._process_element_bbox(element)

        if isinstance(element.label, DocItemLabel):
            # Create item based on label
            return self._create_item_by_label(
                element.label, text, provenance, element, parent
            )
        else:
            return None

    def _get_page_number_from_bbox(self, bbox: BoundingBox) -> int:
        """Determine which page a bbox belongs to based on its x-coordinate."""
        x_center = (bbox.l + bbox.r) / 2

        for page_no in sorted(self.page_widths.keys(), reverse=True):
            if x_center >= self.page_widths[page_no]:
                return page_no

        return 1  # Default to first page

    def _adjust_bbox_for_page(self, bbox: BoundingBox, page_no: int) -> BoundingBox:
        """Adjust bbox coordinates to be relative to its page.

        The input bbox is in CVAT pixel coordinates across all pages.
        This method returns a bbox relative to the specific page, still in pixels.
        """
        if page_no == 1:
            return bbox

        # Subtract the cumulative width of previous pages (in pixels)
        offset = self.page_widths[page_no]
        return BoundingBox(
            l=bbox.l - offset,
            r=bbox.r - offset,
            t=bbox.t,
            b=bbox.b,
            coord_origin=bbox.coord_origin,
        )

    def _create_item_by_label(
        self,
        doc_label: DocItemLabel,
        text: str,
        prov: ProvenanceItem,
        element: CVATElement,
        parent: Optional[NodeItem],
    ) -> Optional[DocItem]:
        """Create appropriate DocItem based on element label."""
        content_layer = element.content_layer

        if doc_label == DocItemLabel.TITLE:
            return self.doc.add_title(
                text=text, prov=prov, parent=parent, content_layer=content_layer
            )

        elif doc_label == DocItemLabel.SECTION_HEADER:
            level = element.level or 1
            return self.doc.add_heading(
                text=text,
                level=level,
                prov=prov,
                parent=parent,
                content_layer=content_layer,
            )

        elif doc_label == DocItemLabel.LIST_ITEM:
            return self.doc.add_list_item(
                text=text, prov=prov, parent=parent, content_layer=content_layer
            )

        elif doc_label in [DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX]:
            # TODO: INSERT TABLE DATA PREP HERE
            pool_rows = self.doc_structure.get_elements_by_label(
                TableStructLabel.TABLE_ROW
            )
            pool_cols = self.doc_structure.get_elements_by_label(
                TableStructLabel.TABLE_COLUMN
            )
            pool_merges = self.doc_structure.get_elements_by_label(
                TableStructLabel.TABLE_MERGED_CELL
            )

            pool_col_headers = self.doc_structure.get_elements_by_label(
                TableStructLabel.COL_HEADER
            )
            pool_row_headers = self.doc_structure.get_elements_by_label(
                TableStructLabel.ROW_HEADER
            )
            pool_row_sections = self.doc_structure.get_elements_by_label(
                TableStructLabel.ROW_SECTION
            )

            pool_fillable_cells = self.doc_structure.get_elements_by_label(
                TableStructLabel.TABLE_FILLABLE_CELLS
            )

            tb = element.bbox

            pool_rows.extend(
                pool_row_sections
            )  # use row sections to compensate for missing rows
            # pool_rows.extend(pool_col_headers)  # use column headers to compensate for missing rows

            rows = dedupe_items_by_bbox(
                [
                    e
                    for e in pool_rows
                    if bbox_contains(e.bbox, tb, threshold=DEFAULT_CONTAINMENT_THRESH)
                ]
            )
            cols = dedupe_items_by_bbox(
                [
                    e
                    for e in pool_cols
                    if bbox_contains(e.bbox, tb, threshold=DEFAULT_CONTAINMENT_THRESH)
                ]
            )
            merges = dedupe_items_by_bbox(
                [
                    e
                    for e in pool_merges
                    if bbox_contains(e.bbox, tb, threshold=DEFAULT_CONTAINMENT_THRESH)
                ]
            )

            col_headers = dedupe_items_by_bbox(
                [
                    e
                    for e in pool_col_headers
                    if bbox_contains(e.bbox, tb, threshold=DEFAULT_CONTAINMENT_THRESH)
                ]
            )
            row_headers = dedupe_items_by_bbox(
                [
                    e
                    for e in pool_row_headers
                    if bbox_contains(e.bbox, tb, threshold=DEFAULT_CONTAINMENT_THRESH)
                ]
            )
            row_sections = dedupe_items_by_bbox(
                [
                    e
                    for e in pool_row_sections
                    if bbox_contains(e.bbox, tb, threshold=DEFAULT_CONTAINMENT_THRESH)
                ]
            )
            fillable_cells = dedupe_items_by_bbox(
                [
                    e
                    for e in pool_fillable_cells
                    if bbox_contains(e.bbox, tb, threshold=DEFAULT_CONTAINMENT_THRESH)
                ]
            )

            # Compute table cells from CVAT elements: rows, cols, merges
            computed_table_cells = compute_cells(
                rows,
                cols,
                merges,
                col_headers,
                row_headers,
                row_sections,
                fillable_cells,
            )

            # If no table structure found, create single fake cell for content
            if not rows or not cols:
                computed_table_cells = [
                    Cell(
                        start_row=0,
                        end_row=0,
                        start_column=0,
                        end_column=0,
                        row_span_length=1,
                        column_span_length=1,
                        bbox=tb,
                        column_header=False,
                        row_header=False,
                        row_section=False,
                        fillable_cell=False,
                    )
                ]
                table_data = TableData(num_rows=1, num_cols=1)
            else:
                table_data = TableData(num_rows=len(rows), num_cols=len(cols))
            # Store pre-computed table structure
            # to be used for re-construction of actual table cells in _process_table_data
            self.tabular_data.append(
                {
                    "computed_table_cells": computed_table_cells,
                    "bbox": tb,
                    "page_no": self._get_page_number_from_bbox(tb),
                }
            )
            return self.doc.add_table(
                data=table_data,
                prov=prov,
                parent=parent,
                label=doc_label,
                content_layer=content_layer,
            )

        elif doc_label in [DocItemLabel.PICTURE, DocItemLabel.HANDWRITTEN_TEXT]:
            pic_item = self.doc.add_picture(
                prov=prov, parent=parent, content_layer=content_layer
            )

            if element.type is not None:
                pic_class = element.type
                pic_class_harmonized = pic_classes[pic_class]

                pic_item.annotations.append(
                    PictureClassificationData(
                        provenance="human",
                        predicted_classes=[
                            PictureClassificationClass(
                                class_name=pic_class_harmonized, confidence=1.0
                            )
                        ],
                    )
                )

            return pic_item
        elif doc_label == DocItemLabel.FORM:
            # Create empty graph data for form
            graph_data = GraphData(nodes=[], edges=[])
            return self.doc.add_form(
                graph=graph_data,
                prov=prov,
                parent=parent,
            )
        elif doc_label == DocItemLabel.KEY_VALUE_REGION:
            _logger.debug(f"Untreatable label: {doc_label}, ignoring.")
            return None
        elif doc_label == DocItemLabel.CODE:
            return self.doc.add_code(
                text=text, prov=prov, parent=parent, content_layer=content_layer
            )

        elif doc_label == DocItemLabel.FORMULA:
            return self.doc.add_formula(
                text=text, prov=prov, parent=parent, content_layer=content_layer
            )
        elif doc_label == DocItemLabel.GRADING_SCALE:
            _logger.debug(f"Untreatable label: {doc_label}, ignoring.")
            return None
        # elif doc_label == DocItemLabel.HANDWRITTEN_TEXT:
        #     _logger.warning(f"Untreatable label: {doc_label}, ignoring.")
        #     return None
        else:
            return self.doc.add_text(
                label=doc_label,
                text=text,
                prov=prov,
                parent=parent,
                content_layer=content_layer,
            )

    def _process_captions_and_footnotes(self):
        """Process caption and footnote relationships.

        Skips invalid paths where neither side is a container element (these are
        already validated and reported as warnings).
        """
        # Process captions
        for path_id, (
            container_id,
            caption_id,
        ) in self.doc_structure.path_mappings.to_caption.items():
            # Skip if neither side is a container (invalid path)
            if not self._has_valid_container_relationship(container_id, caption_id):
                continue
            self._add_caption_or_footnote(container_id, caption_id, is_caption=True)

        # Process footnotes
        for path_id, (
            container_id,
            footnote_id,
        ) in self.doc_structure.path_mappings.to_footnote.items():
            # Skip if neither side is a container (invalid path)
            if not self._has_valid_container_relationship(container_id, footnote_id):
                continue
            self._add_caption_or_footnote(container_id, footnote_id, is_caption=False)

    def _process_to_value_relationships(self) -> None:  # noqa: C901
        """Convert CVAT *to_value* links into a single KeyValueItem graph."""

        if not self.doc_structure.path_mappings.to_value:
            return

        cell_by_element: dict[int, GraphCell] = {}
        links: list[GraphLink] = []
        cell_id_seq: int = 0

        def _make_cell(element_id: int, label: GraphCellLabel) -> GraphCell:
            nonlocal cell_id_seq

            if element_id in cell_by_element:
                return cell_by_element[element_id]

            element = self.doc_structure.get_element_by_id(element_id)
            if element is None:
                raise RuntimeError(
                    f"Element {element_id} referenced in to_value path is missing."
                )

            _, text, prov = self._process_element_bbox(element)

            item_ref = None
            node_item = self.element_to_item.get(element_id)
            if node_item is not None:
                item_ref = node_item.get_ref()

            cell = GraphCell(
                cell_id=cell_id_seq,
                label=label,
                text=text,
                orig=text,
                prov=prov,
                item_ref=item_ref,
            )
            cell_by_element[element_id] = cell
            cell_id_seq += 1
            return cell

        for path_id, (
            key_id,
            value_id,
        ) in self.doc_structure.path_mappings.to_value.items():
            try:
                key_cell = _make_cell(key_id, GraphCellLabel.KEY)
                value_cell = _make_cell(value_id, GraphCellLabel.VALUE)
                links.append(
                    GraphLink(
                        label=GraphLinkLabel.TO_VALUE,
                        source_cell_id=key_cell.cell_id,
                        target_cell_id=value_cell.cell_id,
                    )
                )
            except Exception as err:
                _logger.warning(f"Skipping malformed to_value path {path_id}: {err}")

        if not cell_by_element:
            return

        graph = GraphData(cells=list(cell_by_element.values()), links=links)

        try:
            classify_cells(graph=graph)
        except Exception as err:
            _logger.debug(f"classify_cells failed: {err}")

        # Overall provenance omitted – not needed for CVAT
        self.doc.add_key_values(graph=graph, prov=None)

        sort_cell_ids(self.doc)

    def _add_caption_or_footnote(
        self, container_id: int, target_id: int, is_caption: bool
    ):
        """Add caption or footnote to a container item.

        Handles both forward and backward relationships. Tolerates caption/footnote
        elements labeled as TEXT - always creates items with correct CAPTION/FOOTNOTE
        labels regardless of original label.
        """
        # Try forward direction first (container -> caption)
        if self._try_caption_direction(container_id, target_id, is_caption):
            return

        # Try backward direction (caption -> container)
        if self._try_caption_direction(target_id, container_id, is_caption):
            return

        _logger.warning(
            f"Could not establish {'caption' if is_caption else 'footnote'} relationship between {container_id} and {target_id}"
        )

    def _try_caption_direction(
        self, container_id: int, target_id: int, is_caption: bool
    ) -> bool:
        """Try to add caption/footnote in a specific direction. Returns True if successful."""
        # Get container item
        container_item = self.element_to_item.get(container_id)
        if not container_item or not isinstance(container_item, FloatingItem):
            return False

        # Get target element
        target_element = self.doc_structure.get_element_by_id(target_id)
        if not target_element:
            return False

        page_no, text, provenance = self._process_element_bbox(target_element)

        # Create caption/footnote item - always use correct label regardless of original element label
        label = DocItemLabel.CAPTION if is_caption else DocItemLabel.FOOTNOTE
        item = self.doc.add_text(
            label=label,
            text=text,
            prov=provenance,
            parent=container_item,
            content_layer=target_element.content_layer,
        )

        # Add reference to container
        if item:
            if is_caption:
                container_item.captions.append(item.get_ref())
            else:
                container_item.footnotes.append(item.get_ref())

            self.element_to_item[target_id] = item
            self.processed_elements.add(target_id)
            return True

        return False

    def _extract_text_from_bbox(self, bbox: BoundingBox, page_no: int) -> str:
        """Extract text from bounding box using SegmentedPage text cells.

        Both bbox and SegmentedPage are at cvat_input_scale, so coordinates match perfectly.

        Coordinate system invariant:
        - CVAT bboxes: Always TOP_LEFT origin
        - PDF parser cells: Always BOTTOM_LEFT origin
        - OCR cells: Always TOP_LEFT origin

        Args:
            bbox: Bounding box in CVAT coordinates (TOP_LEFT, at cvat_input_scale)
            page_no: Page number

        Returns:
            Extracted text from the bbox
        """
        try:
            if page_no not in self.segmented_pages:
                return ""

            seg_page = self.segmented_pages[page_no]

            # Adjust bbox for multi-page (still in CVAT TOP_LEFT coordinates at cvat_input_scale)
            adjusted_bbox = self._adjust_bbox_for_page(bbox, page_no)

            if isinstance(seg_page, SegmentedPdfPage):
                # PDF parser → BOTTOM_LEFT: Convert CVAT bbox from TOP_LEFT to BOTTOM_LEFT
                page_height = seg_page.dimension.height
                search_bbox = adjusted_bbox.to_bottom_left_origin(page_height)

                # Get LINE cells only
                cells = seg_page.get_cells_in_bbox(
                    TextCellUnit.LINE, search_bbox, ios=0.1
                )
            else:
                # OCR → TOP_LEFT: Both bbox and cells are TOP_LEFT, no conversion needed
                search_bbox = adjusted_bbox

                # Get WORD cells and filter by containment
                cells = []
                for cell in seg_page.iterate_cells(TextCellUnit.WORD):
                    cell_bbox = cell.rect.to_bounding_box()

                    if is_bbox_within(search_bbox, cell_bbox):
                        cells.append(cell)

            if cells:
                # Sort cells by their index/position
                cells.sort(key=lambda c: c.index if c.index >= 0 else float("inf"))
                text_parts = [cell.text for cell in cells]
                return " ".join(text_parts).strip()

            return ""

        except Exception as e:
            _logger.error(f"Error extracting text: {e}")
            return ""


def scale_segmented_pdf_page(
    seg_page: SegmentedPdfPage, scale: float
) -> SegmentedPdfPage:
    """Scale all coordinates in a SegmentedPdfPage by the given factor.

    Modifies the page in-place for efficiency (no deep copy).

    Args:
        seg_page: SegmentedPdfPage to scale (will be modified in-place)
        scale: Scale factor to apply (e.g., 2.0 for 144 DPI)

    Returns:
        The same SegmentedPdfPage with all coordinates scaled
    """
    if scale == 1.0:
        return seg_page

    # Scale page geometry rect in-place
    rect = seg_page.dimension.rect
    rect.r_x0 *= scale
    rect.r_y0 *= scale
    rect.r_x1 *= scale
    rect.r_y1 *= scale
    rect.r_x2 *= scale
    rect.r_y2 *= scale
    rect.r_x3 *= scale
    rect.r_y3 *= scale

    # Scale all bounding boxes in PdfPageGeometry in-place
    seg_page.dimension.art_bbox = seg_page.dimension.art_bbox.scaled(scale)
    seg_page.dimension.bleed_bbox = seg_page.dimension.bleed_bbox.scaled(scale)
    seg_page.dimension.crop_bbox = seg_page.dimension.crop_bbox.scaled(scale)
    seg_page.dimension.media_bbox = seg_page.dimension.media_bbox.scaled(scale)
    seg_page.dimension.trim_bbox = seg_page.dimension.trim_bbox.scaled(scale)

    # Scale all text cells in-place
    for cell_list in [
        seg_page.char_cells,
        seg_page.word_cells,
        seg_page.textline_cells,
    ]:
        for cell in cell_list:
            cell.rect.r_x0 *= scale
            cell.rect.r_y0 *= scale
            cell.rect.r_x1 *= scale
            cell.rect.r_y1 *= scale
            cell.rect.r_x2 *= scale
            cell.rect.r_y2 *= scale
            cell.rect.r_x3 *= scale
            cell.rect.r_y3 *= scale

    # Scale lines in-place
    for line in seg_page.lines:
        # Scale line points
        line.points = [Coord2D(p.x * scale, p.y * scale) for p in line.points]
        line.width *= scale

    # Scale bitmap resources in-place
    for bitmap in seg_page.bitmap_resources:
        bitmap.rect.r_x0 *= scale
        bitmap.rect.r_y0 *= scale
        bitmap.rect.r_x1 *= scale
        bitmap.rect.r_y1 *= scale
        bitmap.rect.r_x2 *= scale
        bitmap.rect.r_y2 *= scale
        bitmap.rect.r_x3 *= scale
        bitmap.rect.r_y3 *= scale

    return seg_page


def create_segmented_page_from_ocr(
    image: PILImage.Image,
    coordinate_scale: float = 1.0,
    target_width: int | None = None,
    target_height: int | None = None,
) -> SegmentedPage:
    """Create a SegmentedPage from OCR results.

    Args:
        image: PIL Image to run OCR on
        coordinate_scale: Scale to map OCR coordinates back (for PDFs: 2.0/ocr_scale)
        target_width: Target page width for geometry (defaults to image width)
        target_height: Target page height for geometry (defaults to image height)

    Returns:
        SegmentedPage object with coordinates mapped to target dimensions
    """
    from ocrmac import ocrmac

    ocr_results = ocrmac.OCR(
        image,
        framework="vision",
        recognition_level="fast",
        language_preference=["en-US"],
    ).recognize(px=True)

    # Use provided dimensions or fall back to image dimensions
    page_width = target_width if target_width is not None else image.width
    page_height = target_height if target_height is not None else image.height

    # Create page geometry (in pixels, TOP_LEFT origin) using target dimensions
    page_rect = BoundingRectangle(
        r_x0=0,
        r_y0=0,
        r_x1=page_width,
        r_y1=0,
        r_x2=page_width,
        r_y2=page_height,
        r_x3=0,
        r_y3=page_height,
        coord_origin=CoordOrigin.TOPLEFT,
    )

    page_geometry = PageGeometry(angle=0.0, rect=page_rect)

    # Convert OCR results to TextCells
    word_cells = []
    for idx, (text, confidence, coords) in enumerate(ocr_results):
        # coords are in pixels with TOP_LEFT origin: (x0, y0, x1, y1)
        x0, y0, x1, y1 = coords

        # Map coordinates to target space if needed
        if coordinate_scale != 1.0:
            x0 = x0 * coordinate_scale
            y0 = y0 * coordinate_scale
            x1 = x1 * coordinate_scale
            y1 = y1 * coordinate_scale

        # Create BoundingRectangle for the word (TOP_LEFT origin, pixels)
        rect = BoundingRectangle(
            r_x0=x0,
            r_y0=y0,
            r_x1=x1,
            r_y1=y0,
            r_x2=x1,
            r_y2=y1,
            r_x3=x0,
            r_y3=y1,
            coord_origin=CoordOrigin.TOPLEFT,
        )

        # Create TextCell
        word_cell = TextCell(
            index=idx,
            rect=rect,
            text=text,
            orig=text,
            confidence=confidence,
            from_ocr=True,
        )

        word_cells.append(word_cell)

    # Create SegmentedPage
    seg_page = SegmentedPage(
        dimension=page_geometry, word_cells=word_cells, has_words=True
    )

    return seg_page


def load_document_pages(
    input_path: Path,
    page_numbers: Optional[List[int]] = None,
    force_ocr: bool = False,
    ocr_scale: float = 1.0,
    cvat_input_scale: float = 2.0,
) -> Tuple[Dict[int, SegmentedPage], Dict[int, PILImage.Image]]:
    """Load document pages with text extraction at CVAT input scale.

    Args:
        input_path: Path to document (PDF or image)
        page_numbers: Specific page numbers to load (1-indexed). If None, loads all pages.
        force_ocr: Force OCR on PDFs instead of using native text layer
        ocr_scale: Scale factor for rendering PDFs for OCR (default: 1.0 = 72 DPI).
        cvat_input_scale: Scale at which CVAT annotations are provided (2.0 for PDFs, 1.0 for images).
                         All returned SegmentedPages and images will be at this scale.

    Returns:
        Tuple of (segmented_pages dict at cvat_input_scale, page_images dict at cvat_input_scale) with 1-indexed page numbers as keys
    """
    segmented_pages: Dict[int, SegmentedPage] = {}
    page_images: Dict[int, PILImage.Image] = {}

    is_pdf = input_path.suffix.lower() == ".pdf"

    # For images, cvat_input_scale must be 1.0 (native resolution)
    if not is_pdf and cvat_input_scale != 1.0:
        _logger.warning(
            f"cvat_input_scale {cvat_input_scale} ignored for image input; using 1.0 (native resolution)"
        )
        cvat_input_scale = 1.0

    if is_pdf and not force_ocr:
        # PDF with native text layer, fallback to OCR if no text found or text quality is poor
        from docling.backend.docling_parse_v4_backend import (
            DoclingParseV4DocumentBackend,
        )
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.document import InputDocument
        from docling.models.page_preprocessing_model import (
            PagePreprocessingModel,
            PagePreprocessingOptions,
        )

        # Create text quality checker
        quality_checker = PagePreprocessingModel(
            PagePreprocessingOptions(images_scale=1.0)
        )

        in_doc = InputDocument(
            path_or_stream=input_path,
            format=InputFormat.PDF,
            backend=DoclingParseV4DocumentBackend,
        )
        doc_backend: DoclingParseV4DocumentBackend = in_doc._backend  # type: ignore

        num_pages = doc_backend.page_count()

        # Determine which pages to load
        if page_numbers is None:
            pages_to_load = list(range(1, num_pages + 1))
        else:
            pages_to_load = page_numbers

        for page_no in pages_to_load:
            if page_no < 1 or page_no > num_pages:
                _logger.warning(
                    "Page %s out of bounds for document %s (has %s pages)",
                    page_no,
                    input_path.name,
                    num_pages,
                )
                continue

            page = doc_backend.load_page(page_no - 1)
            # Get native SegmentedPage and scale it to cvat_input_scale
            # IMPORTANT: Both seg_page and page_image must be at the same scale (cvat_input_scale)
            native_seg_page = page.get_segmented_page()
            page_image = page.get_page_image(scale=cvat_input_scale)

            # Check if native page has text content
            has_text = False
            has_good_quality = True
            if native_seg_page is not None:
                # Check if any text cells exist
                has_text = (
                    len(native_seg_page.word_cells) > 0
                    or len(native_seg_page.textline_cells) > 0
                    or len(native_seg_page.char_cells) > 0
                )

                # Check text quality if text exists
                if has_text:
                    # Use textline cells for quality check
                    cells_to_check = native_seg_page.textline_cells

                    if cells_to_check:
                        low_quality_count = 0
                        total_count = len(cells_to_check)

                        for cell in cells_to_check:
                            quality_score = quality_checker.rate_text_quality(cell.text)
                            if quality_score < 0.7:
                                low_quality_count += 1

                        low_quality_ratio = low_quality_count / total_count
                        has_good_quality = low_quality_ratio <= 0.05

                        if not has_good_quality:
                            _logger.info(
                                "Page %s of %s has poor text quality (%.1f%% low-quality cells), falling back to OCR",
                                page_no,
                                input_path.name,
                                low_quality_ratio * 100,
                            )

            if has_text and has_good_quality and native_seg_page is not None:
                # Use native text layer
                seg_page = scale_segmented_pdf_page(native_seg_page, cvat_input_scale)
                segmented_pages[page_no] = seg_page
            else:
                # Fallback to OCR
                if not has_text:
                    _logger.info(
                        "Page %s of %s has no text layer, falling back to OCR",
                        page_no,
                        input_path.name,
                    )
                ocr_image = page.get_page_image(scale=ocr_scale)
                if ocr_image is not None and page_image is not None:
                    coord_scale = cvat_input_scale / ocr_scale
                    seg_page = create_segmented_page_from_ocr(
                        ocr_image,
                        coordinate_scale=coord_scale,
                        target_width=page_image.width,
                        target_height=page_image.height,
                    )
                    segmented_pages[page_no] = seg_page

            if page_image is not None:
                page_images[page_no] = page_image

            page.unload()

        doc_backend.unload()

    elif is_pdf and force_ocr:
        # PDF with OCR
        from docling.backend.docling_parse_v4_backend import (
            DoclingParseV4DocumentBackend,
        )
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.document import InputDocument

        in_doc = InputDocument(
            path_or_stream=input_path,
            format=InputFormat.PDF,
            backend=DoclingParseV4DocumentBackend,
        )
        doc_backend: DoclingParseV4DocumentBackend = in_doc._backend  # type: ignore

        num_pages = doc_backend.page_count()

        # Determine which pages to load
        if page_numbers is None:
            pages_to_load = list(range(1, num_pages + 1))
        else:
            pages_to_load = page_numbers

        for page_no in pages_to_load:
            if page_no < 1 or page_no > num_pages:
                _logger.warning(
                    "Page %s out of bounds for document %s (has %s pages)",
                    page_no,
                    input_path.name,
                    num_pages,
                )
                continue

            page = doc_backend.load_page(page_no - 1)

            # Get high-res image for OCR at user-specified scale
            ocr_image = page.get_page_image(scale=ocr_scale)
            # Get image at cvat_input_scale
            cvat_image = page.get_page_image(scale=cvat_input_scale)

            if ocr_image is not None and cvat_image is not None:
                # Map OCR coordinates from ocr_scale to cvat_input_scale
                coord_scale = cvat_input_scale / ocr_scale
                seg_page = create_segmented_page_from_ocr(
                    ocr_image,
                    coordinate_scale=coord_scale,
                    target_width=cvat_image.width,
                    target_height=cvat_image.height,
                )
                segmented_pages[page_no] = seg_page
                page_images[page_no] = cvat_image

            page.unload()

        doc_backend.unload()

    else:
        # Image input
        image = PILImage.open(input_path)
        seg_page = create_segmented_page_from_ocr(image)

        # For images, use page number 1 if not specified
        page_no = page_numbers[0] if page_numbers else 1
        segmented_pages[page_no] = seg_page
        page_images[page_no] = image

    return segmented_pages, page_images


@dataclass
class CVATConversionResult:
    """Outcome of converting a CVAT document."""

    document: Optional[DoclingDocument]
    validation_report: Optional[CVATValidationReport]
    per_page_reports: Dict[str, CVATValidationReport] = field(default_factory=dict)
    error: Optional[str] = None


def convert_cvat_folder_to_docling(
    folder_path: Path,
    xml_pattern: str = "task_{xx}_set_A",
    output_dir: Optional[Path] = None,
    save_formats: Optional[List[str]] = None,
    folder_structure: Optional[CVATFolderStructure] = None,
    log_validation: bool = False,
    force_ocr: bool = False,
    ocr_scale: float = 1.0,
    cvat_input_scale: float = 2.0,
    storage_scale: float = 2.0,
) -> Dict[str, CVATConversionResult]:
    """Convert an entire CVAT folder into DoclingDocument objects grouped by document.

    Args:
        folder_path: Path to CVAT folder
        xml_pattern: Pattern for XML files to process
        output_dir: Directory for output files
        save_formats: List of output formats
        folder_structure: Pre-parsed folder structure
        log_validation: Whether to log validation reports
        force_ocr: Force OCR on PDFs instead of using native text layer
        ocr_scale: Scale factor for rendering PDFs for OCR (default: 1.0 = 72 DPI).
        cvat_input_scale: Scale at which CVAT annotations are provided (2.0 for PDFs, 1.0 for images)
        storage_scale: Scale for stored page images and coordinates (default: 2.0 for PDFs, 1.0 for images)

    Returns:
        Dictionary mapping doc_hash to CVATConversionResult
    """

    if save_formats is None:
        save_formats = ["json"]

    if output_dir is None:
        output_dir = folder_path / "json_predictions"

    if folder_structure is None:
        folder_structure = parse_cvat_folder(folder_path, xml_pattern)

    converter = CVATFolderConverter(
        folder_structure,
        log_validation=log_validation,
        force_ocr=force_ocr,
        ocr_scale=ocr_scale,
        cvat_input_scale=cvat_input_scale,
        storage_scale=storage_scale,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert and write documents one at a time to reduce memory usage
    results: Dict[str, CVATConversionResult] = {}
    total_docs = len(folder_structure.documents)
    for idx, doc_hash in enumerate(folder_structure.documents, start=1):
        cvat_doc = folder_structure.documents[doc_hash]
        _logger.info(f"[{idx}/{total_docs}] Converting {cvat_doc.doc_name}...")
        result = converter.convert_document(doc_hash)

        # Store result (without document to save memory)
        results[doc_hash] = result

        # Write immediately if conversion succeeded
        doc = result.document
        if doc is not None:
            cvat_doc = folder_structure.documents[doc_hash]
            base_filename = cvat_doc.doc_name

            for format_type in save_formats:
                if format_type == "json":
                    output_path = output_dir / f"{base_filename}.json"
                    doc.save_as_json(output_path)
                elif format_type == "html":
                    output_path = output_dir / f"{base_filename}.html"
                    doc.save_as_html(output_path, image_mode=ImageRefMode.EMBEDDED)
                elif format_type == "md":
                    output_path = output_dir / f"{base_filename}.md"
                    doc.save_as_markdown(output_path, image_mode=ImageRefMode.EMBEDDED)
                elif format_type == "txt":
                    output_path = output_dir / f"{base_filename}.txt"
                    with open(output_path, "w", encoding="utf-8") as fp:
                        fp.write(doc.export_to_element_tree())
                elif format_type == "viz":
                    viz_imgs = doc.get_visualization()
                    for page_no, img in viz_imgs.items():
                        if page_no is not None:
                            img.save(
                                output_dir / f"{base_filename}_docling_p{page_no}.png"
                            )

            # Free memory by discarding the document after writing
            result.document = None

    return results


class CVATFolderConverter:
    """Convert CVAT folder structures into DoclingDocument instances."""

    def __init__(
        self,
        folder_structure: CVATFolderStructure,
        log_validation: bool = False,
        force_ocr: bool = False,
        ocr_scale: float = 1.0,
        cvat_input_scale: float = 2.0,
        storage_scale: float = 2.0,
    ):
        self.folder_structure = folder_structure
        self.log_validation = log_validation
        self.force_ocr = force_ocr
        self.ocr_scale = ocr_scale
        self.cvat_input_scale = cvat_input_scale
        self.storage_scale = storage_scale

    def convert_document(self, doc_hash: str) -> CVATConversionResult:
        """Convert a single document identified by its hash."""

        if doc_hash not in self.folder_structure.documents:
            _logger.error("Document %s not found in folder structure", doc_hash)
            return CVATConversionResult(
                document=None,
                validation_report=None,
                error="Document missing from folder structure",
            )

        cvat_doc = self.folder_structure.documents[doc_hash]

        try:
            validator = Validator()
            per_page_reports: Dict[str, CVATValidationReport] = {}
            fatal_messages: List[str] = []

            parsed_cache: Dict[Path, ParsedCVATFile] = {}

            for page_info in sorted(cvat_doc.pages, key=lambda page: page.page_number):
                page_name = page_info.image_filename
                try:
                    parsed_file = parsed_cache.get(page_info.xml_path)
                    if parsed_file is None:
                        parsed_file = parse_cvat_file(page_info.xml_path)
                        parsed_cache[page_info.xml_path] = parsed_file
                    validated = validate_cvat_sample(
                        page_info.xml_path,
                        page_name,
                        validator=validator,
                        parsed_file=parsed_file,
                    )
                    page_report = validated.report
                except Exception as exc:  # noqa: BLE001
                    page_report = CVATValidationReport(
                        sample_name=page_name,
                        errors=[
                            CVATValidationError(
                                error_type="processing_error",
                                message=f"Validation failed: {exc}",
                                severity=ValidationSeverity.FATAL,
                            )
                        ],
                    )
                    fatal_messages.append(f"{page_name}: Validation failed: {exc}")
                    _logger.error(
                        "Validation failed for document %s page %s: %s",
                        cvat_doc.doc_name,
                        page_name,
                        exc,
                    )
                else:
                    if page_report.has_fatal_errors():
                        page_fatals = [
                            f"{err.error_type}: {err.message}"
                            for err in page_report.errors
                            if err.severity == ValidationSeverity.FATAL
                        ]
                        fatal_messages.append(
                            f"{page_name}: "
                            + (
                                "; ".join(page_fatals)
                                if page_fatals
                                else "Fatal errors"
                            )
                        )

                per_page_reports[page_name] = page_report

                if self.log_validation:
                    _logger.info(
                        "Validation report for %s (page %s):\n%s",
                        cvat_doc.doc_name,
                        page_name,
                        page_report.model_dump_json(indent=2),
                    )

            if fatal_messages:
                _logger.error(
                    "Fatal validation errors on document %s. Skipping conversion.",
                    cvat_doc.doc_name,
                )
                error_message = " | ".join(fatal_messages)
                return CVATConversionResult(
                    document=None,
                    validation_report=None,
                    per_page_reports=per_page_reports,
                    error=error_message,
                )

            doc_structure = DocumentStructure.from_cvat_folder_structure(
                self.folder_structure, doc_hash
            )

            # Determine which pages to load and the input path
            annotated_page_numbers = sorted(
                {page_info.page_number for page_info in cvat_doc.pages}
            )

            # Determine scales based on input type
            is_pdf = cvat_doc.mime_type == "application/pdf"
            actual_cvat_input_scale = self.cvat_input_scale if is_pdf else 1.0
            actual_storage_scale = self.storage_scale if is_pdf else 1.0

            # Use shared function to load document pages (always at cvat_input_scale)
            if is_pdf:
                segmented_pages, page_images = load_document_pages(
                    input_path=cvat_doc.bin_file,
                    page_numbers=annotated_page_numbers,
                    force_ocr=self.force_ocr,
                    ocr_scale=self.ocr_scale,
                    cvat_input_scale=actual_cvat_input_scale,
                )
            else:
                # For images, load each page individually (since they may be separate files)
                segmented_pages = {}
                page_images = {}
                for page_info in cvat_doc.pages:
                    seg_pages, p_images = load_document_pages(
                        input_path=page_info.image_path,
                        page_numbers=[page_info.page_number],
                        force_ocr=False,  # Images always use OCR
                        ocr_scale=1.0,
                        cvat_input_scale=1.0,  # Images always at native resolution
                    )
                    segmented_pages.update(seg_pages)
                    page_images.update(p_images)

            converter = CVATToDoclingConverter(
                doc_structure,
                segmented_pages,
                page_images,
                cvat_doc.doc_name,
                cvat_input_scale=actual_cvat_input_scale,
                storage_scale=actual_storage_scale,
            )

            docling_doc = converter.convert()
            return CVATConversionResult(
                document=docling_doc,
                validation_report=None,
                per_page_reports=per_page_reports,
                error=None,
            )

        except Exception as exc:  # pragma: no cover - logged error propagation
            _logger.error("Error converting document %s: %s", doc_hash, exc)
            return CVATConversionResult(
                document=None,
                validation_report=None,
                per_page_reports=(
                    per_page_reports if "per_page_reports" in locals() else {}
                ),
                error=str(exc),
            )

    def convert_all_documents(self) -> Dict[str, CVATConversionResult]:
        """Convert every document present in the folder structure."""

        results: Dict[str, CVATConversionResult] = {}
        for doc_hash in self.folder_structure.documents:
            results[doc_hash] = self.convert_document(doc_hash)

        return results


def convert_cvat_to_docling(
    xml_path: Path,
    input_path: Path,
    image_identifier: Optional[str] = None,
    force_ocr: bool = False,
    ocr_scale: float = 1.0,
    cvat_input_scale: float = 2.0,
    storage_scale: float = 2.0,
) -> Optional[DoclingDocument]:
    """Convert a CVAT annotation to DoclingDocument.

    This function handles both image and PDF inputs, with proper coordinate system conversion:
    - CVAT annotations use pixels with TOP_LEFT origin at cvat_input_scale
    - PDF SegmentedPages use points with BOTTOM_LEFT origin at storage_scale
    - Image/OCR SegmentedPages use pixels with TOP_LEFT origin (always at scale 1.0)

    Args:
        xml_path: Path to CVAT XML file
        input_path: Path to document (image or PDF)
        image_identifier: Image filename as it appears in the CVAT XML (optional, defaults to input_path.name)
        force_ocr: Force OCR on PDFs instead of using native text layer
        ocr_scale: Scale factor for rendering PDFs for OCR (default: 1.0 = 72 DPI).
        cvat_input_scale: Scale at which CVAT annotations are provided (2.0 for PDFs, 1.0 for images)
        storage_scale: Scale for stored page images and coordinates (default: 2.0 for PDFs, 1.0 for images)

    Returns:
        DoclingDocument or None if conversion fails
    """
    try:
        image_name = (
            image_identifier if image_identifier is not None else input_path.name
        )
        parsed_file = parse_cvat_file(xml_path)
        validated_sample = validate_cvat_sample(
            xml_path, image_name, parsed_file=parsed_file
        )
        doc_structure = validated_sample.structure
        validation_report = validated_sample.report

        print(validation_report.model_dump_json(indent=2))

        if validation_report.has_fatal_errors():
            _logger.error(
                f"Fatal validation errors on sample {input_path.name}. Skipping conversion."
            )
            return None

        # Determine scales based on input type
        is_pdf = input_path.suffix.lower() == ".pdf"
        actual_cvat_input_scale = cvat_input_scale if is_pdf else 1.0
        actual_storage_scale = storage_scale if is_pdf else 1.0

        # Use shared function to load document pages (at cvat_input_scale)
        segmented_pages, page_images = load_document_pages(
            input_path=input_path,
            page_numbers=None,  # Load all pages
            force_ocr=force_ocr,
            ocr_scale=ocr_scale,
            cvat_input_scale=actual_cvat_input_scale,
        )

        # Create converter
        converter = CVATToDoclingConverter(
            doc_structure,
            segmented_pages,
            page_images,
            input_path.name,
            cvat_input_scale=actual_cvat_input_scale,
            storage_scale=actual_storage_scale,
        )

        # Convert
        return converter.convert()

    except MissingImageInCVATXML:
        # Re-raise so that calling code can handle with appropriate messaging
        raise
    except Exception as e:
        _logger.error(f"Failed to convert CVAT to DoclingDocument: {e}")
        import traceback

        traceback.print_exc()
        return None
