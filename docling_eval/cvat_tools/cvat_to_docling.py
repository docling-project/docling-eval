"""Convert CVAT DocumentStructure to DoclingDocument.

This module provides functionality to convert a populated DocumentStructure
from the CVAT parser into a DoclingDocument, handling text extraction via OCR
or PDF parsing, reading order, containment hierarchy, groups, merges, and 
caption/footnote relationships.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.document import (
    ContentLayer,
    DocItem,
    DocItemLabel,
    DoclingDocument,
    FloatingItem,
    GraphData,
    GroupItem,
    GroupLabel,
    ImageRef,
    ListItem,
    NodeItem,
    OrderedList,
    PictureClassificationClass,
    PictureClassificationData,
    ProvenanceItem,
    Size,
    TableData,
    UnorderedList,
)
from docling_core.types.doc.page import (
    BoundingRectangle,
    PageGeometry,
    SegmentedPage,
    SegmentedPdfPage,
    TextCell,
    TextCellUnit,
)
from PIL import Image as PILImage

from docling_eval.cvat_tools.document import DocumentStructure
from docling_eval.cvat_tools.models import CVATElement
from docling_eval.cvat_tools.tree import (
    TreeNode,
    apply_reading_order_to_tree,
    build_global_reading_order,
    find_node_by_element_id,
)
from docling_eval.cvat_tools.validator import Validator

_logger = logging.getLogger(__name__)


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
    """Converts CVAT DocumentStructure to DoclingDocument."""

    def __init__(
        self,
        doc_structure: DocumentStructure,
        segmented_pages: Dict[int, SegmentedPage],
        page_images: Dict[int, PILImage.Image],
        document_filename: Optional[str] = None,
    ):
        """Initialize the converter.

        Args:
            doc_structure: The populated DocumentStructure from CVAT parser
            segmented_pages: Dictionary mapping page numbers to SegmentedPage objects
            page_images: Dictionary mapping page numbers to PIL images
            document_filename: Optional filename for the document
        """
        self.doc_structure = doc_structure
        self.segmented_pages = segmented_pages
        self.page_images = page_images
        self.document_filename = document_filename or "document"
        self.num_pages = len(segmented_pages)

        # Initialize empty DoclingDocument
        self.doc = DoclingDocument(name=Path(self.document_filename).stem)

        # Maps for tracking created items
        self.element_to_item: Dict[int, Optional[NodeItem]] = {}
        self.processed_elements: Set[int] = set()

        # Track which groups have been created
        self.created_groups: Dict[int, GroupItem] = {}  # path_id -> GroupItem

        # Centralized list hierarchy management
        self.list_manager = ListHierarchyManager(self.doc)

        # Calculate single scaling factor for all pages
        self._calculate_scaling_factor()

        # Calculate page widths for multi-page handling (in CVAT pixel coordinates)
        self.page_widths = {}
        cumulative_width = 0.0

        # Use CVAT image info dimensions if available, otherwise calculate from pages
        if self.doc_structure.image_info:
            # For multi-page documents, CVAT concatenates pages horizontally
            total_cvat_width = self.doc_structure.image_info.width
            if self.num_pages > 1:
                # Distribute width proportionally based on original page widths
                total_original_width = sum(
                    p.dimension.width for p in segmented_pages.values()
                )
                for page_no in sorted(segmented_pages.keys()):
                    self.page_widths[page_no] = cumulative_width
                    page_width_ratio = (
                        segmented_pages[page_no].dimension.width / total_original_width
                    )
                    page_width_pixels = total_cvat_width * page_width_ratio
                    cumulative_width += page_width_pixels
            else:
                # Single page
                self.page_widths[1] = 0.0
        else:
            # Fallback: use page dimensions directly
            for page_no in sorted(segmented_pages.keys()):
                self.page_widths[page_no] = cumulative_width
                cumulative_width += (
                    segmented_pages[page_no].dimension.width * self.scale_factor
                )

    def _calculate_scaling_factor(self):
        """Calculate single scaling factor between PDF points and CVAT pixels."""
        if not self.doc_structure.image_info or not self.segmented_pages:
            self.scale_factor = 1.0
            return

        # Get CVAT image dimensions (in pixels)
        cvat_height = self.doc_structure.image_info.height

        # For PDFs, calculate scaling based on height comparison
        first_page = self.segmented_pages[1]
        if isinstance(first_page, SegmentedPdfPage):
            pdf_height_points = first_page.dimension.height
            # Scale factor: pixels per point
            self.scale_factor = cvat_height / pdf_height_points
        else:
            # For images, no scaling needed
            self.scale_factor = 1.0

    def _process_element_bbox(
        self, element: CVATElement
    ) -> Tuple[int, str, ProvenanceItem]:
        """Process element bbox to extract page, text, and create provenance.

        Returns:
            Tuple of (page_no, text, provenance_item)
        """
        # Get page number from bbox position
        page_no = self._get_page_number_from_bbox(element.bbox)

        # Extract text
        text = self._extract_text_from_bbox(element.bbox, page_no)

        # Adjust bbox for multi-page (still in pixels)
        adjusted_bbox = self._adjust_bbox_for_page(element.bbox, page_no)

        # Convert to page-native units and ensure BOTTOM_LEFT coordinates
        seg_page = self.segmented_pages[page_no]
        if isinstance(seg_page, SegmentedPdfPage):
            # Convert pixels to points and ensure BOTTOM_LEFT
            prov_bbox = BoundingBox(
                l=adjusted_bbox.l / self.scale_factor,
                r=adjusted_bbox.r / self.scale_factor,
                t=adjusted_bbox.t / self.scale_factor,
                b=adjusted_bbox.b / self.scale_factor,
                coord_origin=CoordOrigin.TOPLEFT,
            )
            # Convert to BOTTOM_LEFT for consistency
            page_height = seg_page.dimension.height
            prov_bbox = prov_bbox.to_bottom_left_origin(page_height)
        else:
            # For images, convert pixels to BOTTOM_LEFT for consistency
            page_height = seg_page.dimension.height
            prov_bbox = adjusted_bbox.to_bottom_left_origin(page_height)

        # Create provenance
        provenance = ProvenanceItem(
            page_no=page_no, bbox=prov_bbox, charspan=(0, len(text))
        )

        return page_no, text, provenance

    def convert(self) -> DoclingDocument:
        """Convert the DocumentStructure to DoclingDocument.

        Returns:
            The converted DoclingDocument
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

        # Process captions and footnotes
        self._process_captions_and_footnotes()

        return self.doc

    def _reset_list_state(self):
        """Reset list processing state for clean conversion."""
        self.list_manager.clear()

    def _add_pages(self):
        """Add page information to the document."""
        for page_no, seg_page in self.segmented_pages.items():
            if isinstance(seg_page, SegmentedPdfPage):
                # For PDFs, use the page geometry directly
                page_size = Size(
                    width=seg_page.dimension.width,
                    height=seg_page.dimension.height,
                )
            else:
                # For images, dimensions are already in pixels
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
        self, element: CVATElement, parent_item: Optional[NodeItem]
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

    def _process_elements_in_order(self, global_order: List[int]):
        """Process elements in reading order."""
        # Process elements in global reading order
        for element_id in global_order:
            # Skip if already processed
            if element_id in self.processed_elements:
                continue

            # Find the node containing this element
            node = find_node_by_element_id(self.doc_structure.tree_roots, element_id)
            if node:
                self._process_node(node, parent_item=None, global_order=global_order)

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
        self, node: TreeNode, parent_item: Optional[NodeItem], global_order: List[int]
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
            list_item = self._process_list_item_with_hierarchy(element, parent_item)
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
                self._process_node(child, item, global_order)

    def _is_caption_or_footnote_target(self, element_id: int) -> bool:
        """Check if element is a target of caption/footnote relationship."""
        # Check captions
        for path_id, (
            container_id,
            target_id,
        ) in self.doc_structure.path_mappings.to_caption.items():
            if target_id == element_id:
                return True

        # Check footnotes
        for path_id, (
            container_id,
            target_id,
        ) in self.doc_structure.path_mappings.to_footnote.items():
            if target_id == element_id:
                return True

        return False

    def _get_merge_elements(self, element_id: int) -> List[CVATElement]:
        """Get all elements that should be merged with the given element."""
        merge_elements = []

        for path_id, element_ids in self.doc_structure.path_mappings.merge.items():
            if element_id in element_ids:
                # Get all elements in this merge that haven't been processed
                for el_id in element_ids:
                    if el_id not in self.processed_elements:
                        element = self.doc_structure.get_element_by_id(el_id)
                        if element:
                            merge_elements.append(element)
                break

        return merge_elements

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

        # Create item based on label
        item = self._create_item_by_label(
            primary_element.label, merged_text, all_provs[0], primary_element, parent
        )

        # Add additional provenances
        if item and len(all_provs) > 1:
            item.prov.extend(all_provs[1:])

        return item

    def _create_single_item(
        self, element: CVATElement, parent: Optional[NodeItem]
    ) -> Optional[NodeItem]:
        """Create a DocItem for a single element."""
        page_no, text, provenance = self._process_element_bbox(element)

        # Create item based on label
        return self._create_item_by_label(
            element.label, text, provenance, element, parent
        )

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
        label: str,
        text: str,
        prov: ProvenanceItem,
        element: CVATElement,
        parent: Optional[NodeItem],
    ) -> Optional[DocItem]:
        """Create appropriate DocItem based on element label."""
        content_layer = ContentLayer(element.content_layer.lower())

        try:
            doc_label = DocItemLabel(label)
        except ValueError:
            _logger.warning(f"Unknown label: {label}, using TEXT")
            doc_label = DocItemLabel.TEXT

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
            # Create empty table data
            table_data = TableData(num_rows=0, num_cols=0, table_cells=[])
            return self.doc.add_table(
                data=table_data,
                prov=prov,
                parent=parent,
                label=doc_label,
                content_layer=content_layer,
            )

        elif doc_label == DocItemLabel.PICTURE:
            pic_item = self.doc.add_picture(
                prov=prov, parent=parent, content_layer=content_layer
            )

            if element.type is not None:
                pic_class = element.type
                pic_item.annotations.append(
                    PictureClassificationData(
                        provenance="human",
                        predicted_classes=[
                            PictureClassificationClass(
                                class_name=pic_class, confidence=1.0
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
            _logger.warning(f"Untreatable label: {doc_label}, ignoring.")
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
            _logger.warning(f"Untreatable label: {doc_label}, ignoring.")
            return None
        elif doc_label == DocItemLabel.HANDWRITTEN_TEXT:
            _logger.warning(f"Untreatable label: {doc_label}, ignoring.")
            return None
        else:
            return self.doc.add_text(
                label=doc_label,
                text=text,
                prov=prov,
                parent=parent,
                content_layer=content_layer,
            )

    def _process_captions_and_footnotes(self):
        """Process caption and footnote relationships."""
        # Process captions
        for path_id, (
            container_id,
            caption_id,
        ) in self.doc_structure.path_mappings.to_caption.items():
            self._add_caption_or_footnote(container_id, caption_id, is_caption=True)

        # Process footnotes
        for path_id, (
            container_id,
            footnote_id,
        ) in self.doc_structure.path_mappings.to_footnote.items():
            self._add_caption_or_footnote(container_id, footnote_id, is_caption=False)

    def _add_caption_or_footnote(
        self, container_id: int, target_id: int, is_caption: bool
    ):
        """Add caption or footnote to a container item."""
        # Get container item
        container_item = self.element_to_item.get(container_id)
        if not container_item:
            _logger.warning(f"Container {container_id} not found.")
            return

        # Check if container supports captions/footnotes
        if not isinstance(container_item, FloatingItem):
            _logger.warning(
                f"Container {container_id} does not support {'captions' if is_caption else 'footnotes'}"
            )
            return

        # Get target element
        target_element = self.doc_structure.get_element_by_id(target_id)
        if not target_element:
            return

        page_no, text, provenance = self._process_element_bbox(target_element)

        # Create caption/footnote item
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

    def _extract_text_from_bbox(self, bbox: BoundingBox, page_no: int) -> str:
        """Extract text from bounding box using SegmentedPage text cells."""
        try:
            if page_no not in self.segmented_pages:
                return ""

            seg_page = self.segmented_pages[page_no]

            # Adjust bbox for multi-page (this gives us page-relative coordinates)
            adjusted_bbox = self._adjust_bbox_for_page(bbox, page_no)

            # CVAT bboxes are in pixels with TOP_LEFT origin
            # Convert to the coordinate system of the segmented page

            if isinstance(seg_page, SegmentedPdfPage):
                # Convert from pixels to points
                scaled_bbox = BoundingBox(
                    l=adjusted_bbox.l / self.scale_factor,
                    r=adjusted_bbox.r / self.scale_factor,
                    t=adjusted_bbox.t / self.scale_factor,
                    b=adjusted_bbox.b / self.scale_factor,
                    coord_origin=CoordOrigin.TOPLEFT,
                )
                # Convert to BOTTOM_LEFT for PDF comparison
                page_height_points = seg_page.dimension.height
                search_bbox = scaled_bbox.to_bottom_left_origin(page_height_points)

                # Get LINE cells only
                cells = seg_page.get_cells_in_bbox(
                    TextCellUnit.LINE, search_bbox, ios=0.1
                )
            else:
                # For OCR/image pages, keep TOP_LEFT and use pixels
                search_bbox = adjusted_bbox

                # Get LINE cells only
                cells = []
                for cell in seg_page.iterate_cells(TextCellUnit.WORD):
                    cell_bbox = cell.rect.to_bounding_box()

                    # Ensure we're comparing in the same coordinate system (TOP_LEFT)
                    if cell_bbox.coord_origin != search_bbox.coord_origin:
                        cell_bbox = cell_bbox.to_top_left_origin(
                            seg_page.dimension.height
                        )

                    if search_bbox.intersection_over_union(cell_bbox) > 0.1:
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


def create_segmented_page_from_ocr(image):
    """Create a SegmentedPage from OCR results.

    Args:
        ocr_results: List of (text, confidence, coords) tuples from ocrmac
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels

    Returns:
        SegmentedPage object
    """
    from ocrmac import ocrmac

    ocr_results = ocrmac.OCR(image, framework="vision").recognize(px=True)

    # Create page geometry (in pixels, TOP_LEFT origin)
    page_rect = BoundingRectangle(
        r_x0=0,
        r_y0=0,
        r_x1=image.width,
        r_y1=0,
        r_x2=image.width,
        r_y2=image.height,
        r_x3=0,
        r_y3=image.height,
        coord_origin=CoordOrigin.TOPLEFT,
    )

    page_geometry = PageGeometry(angle=0.0, rect=page_rect)

    # Convert OCR results to TextCells
    word_cells = []
    for idx, (text, confidence, coords) in enumerate(ocr_results):
        # coords are in pixels with TOP_LEFT origin: (x0, y0, x1, y1)
        x0, y0, x1, y1 = coords

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


def convert_cvat_to_docling(
    xml_path: Path,
    input_path: Path,
) -> Optional[DoclingDocument]:
    """Convert a CVAT annotation to DoclingDocument.

    This function handles both image and PDF inputs, with proper coordinate system conversion:
    - CVAT annotations use pixels with TOP_LEFT origin
    - PDF SegmentedPages use points with BOTTOM_LEFT origin
    - Image/OCR SegmentedPages use pixels with TOP_LEFT origin

    The converter automatically calculates scaling factors between PDF points and CVAT pixels
    by comparing the CVAT image height with the PDF page height.

    Args:
        xml_path: Path to CVAT XML file
        input_path: Path to document (image or PDF)
        ocr_framework: OCR framework to use for images ("vision" or "livetext")

    Returns:
        DoclingDocument or None if conversion fails
    """
    try:

        # Create DocumentStructure
        doc_structure = DocumentStructure.from_cvat_xml(xml_path, input_path.name)

        validator = Validator()
        validation_report = validator.validate_sample(input_path.name, doc_structure)

        print(validation_report.model_dump_json(indent=2))

        if validation_report.has_fatal_errors():
            _logger.error(
                f"Fatal validation errors on sample {input_path.name}. Skipping conversion."
            )
            return None

        is_pdf = input_path.suffix.lower() == ".pdf"

        # Prepare segmented pages and images
        segmented_pages = {}
        page_images = {}

        if is_pdf:
            # Handle PDF input
            from docling.backend.docling_parse_v4_backend import (
                DoclingParseV4DocumentBackend,
            )
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.document import InputDocument

            # Create input document
            in_doc = InputDocument(
                path_or_stream=input_path,
                format=InputFormat.PDF,
                backend=DoclingParseV4DocumentBackend,
            )
            doc_backend: DoclingParseV4DocumentBackend = in_doc._backend  # type: ignore

            # Get number of pages
            num_pages = doc_backend.page_count()

            # Parse each page
            for page_no in range(1, num_pages + 1):
                page = doc_backend.load_page(page_no - 1)
                seg_page = page.get_segmented_page()
                page_image = page.get_page_image()

                if seg_page is not None:
                    segmented_pages[page_no] = seg_page
                if page_image is not None:
                    page_images[page_no] = page_image

                page.unload()

            doc_backend.unload()

        else:
            # Handle image input
            image = PILImage.open(input_path)

            # Run OCR

            # Create SegmentedPage from OCR results
            seg_page = create_segmented_page_from_ocr(image)

            segmented_pages[1] = seg_page
            page_images[1] = image

        # Create converter
        converter = CVATToDoclingConverter(
            doc_structure, segmented_pages, page_images, input_path.name  # type: ignore
        )

        # Convert
        return converter.convert()

    except Exception as e:
        _logger.error(f"Failed to convert CVAT to DoclingDocument: {e}")
        import traceback

        traceback.print_exc()
        return None
