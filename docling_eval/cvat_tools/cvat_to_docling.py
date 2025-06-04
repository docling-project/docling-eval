"""Convert CVAT DocumentStructure to DoclingDocument.

This module provides functionality to convert a populated DocumentStructure
from the CVAT parser into a DoclingDocument, handling text extraction via OCR,
reading order, containment hierarchy, groups, merges, and caption/footnote relationships.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.document import (
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    GraphData,
    GroupLabel,
    ImageRef,
    NodeItem,
    PictureClassificationClass,
    PictureClassificationData,
    ProvenanceItem,
    Size,
    TableData,
)
from ocrmac import ocrmac
from PIL import Image as PILImage

from docling_eval.cvat_tools.analysis import apply_reading_order_to_tree
from docling_eval.cvat_tools.document import DocumentStructure
from docling_eval.cvat_tools.models import CVATElement
from docling_eval.cvat_tools.tree import TreeNode, build_global_reading_order

_logger = logging.getLogger(__name__)


class CVATToDoclingConverter:
    """Converts CVAT DocumentStructure to DoclingDocument."""

    def __init__(
        self,
        doc_structure: DocumentStructure,
        image: PILImage.Image,
        ocr_framework: str = "vision",
        image_filename: Optional[str] = None,
    ):
        """Initialize the converter.

        Args:
            doc_structure: The populated DocumentStructure from CVAT parser
            image: The document image
            ocr_framework: OCR framework to use ("vision" or "livetext")
            image_filename: Optional filename for the document
        """
        self.doc_structure = doc_structure
        self.image = image
        self.ocr_framework = ocr_framework
        self.image_filename = image_filename or "document"

        # Initialize empty DoclingDocument
        self.doc = DoclingDocument(name=Path(self.image_filename).stem)

        # Maps for tracking created items
        self.element_to_item: Dict[int, NodeItem] = {}
        self.processed_elements: Set[int] = set()

        # Track which groups have been created
        self.created_groups: Dict[int, NodeItem] = {}  # path_id -> GroupItem

        # Initialize OCR
        _logger.info(f"Initializing OCR with framework: {ocr_framework}")
        self.ocr_results = ocrmac.OCR(
            self.image, framework=self.ocr_framework
        ).recognize(px=True)

    def convert(self) -> DoclingDocument:
        """Convert the DocumentStructure to DoclingDocument.

        Returns:
            The converted DoclingDocument
        """
        # Add page to document
        self._add_page()

        # Apply reading order to tree
        self._apply_reading_order()

        # Build global reading order
        global_order = self._build_global_reading_order()

        # Process elements in reading order (groups will be created on-demand)
        self._process_elements_in_order(global_order)

        # Process captions and footnotes
        self._process_captions_and_footnotes()

        return self.doc

    def _add_page(self):
        """Add page information to the document."""
        if self.doc_structure.image_info:
            page_size = Size(
                width=self.doc_structure.image_info.width,
                height=self.doc_structure.image_info.height,
            )
        else:
            page_size = Size(width=self.image.width, height=self.image.height)

        # Create image reference
        image_ref = ImageRef.from_pil(self.image, dpi=72)

        # Add page
        self.doc.add_page(page_no=1, size=page_size, image=image_ref)

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

    def _process_elements_in_order(self, global_order: List[int]):
        """Process elements in reading order."""
        # Process elements in global reading order
        for element_id in global_order:
            # Skip if already processed
            if element_id in self.processed_elements:
                continue

            # Find the node containing this element
            node = self._find_node_by_element_id(element_id)
            if node:
                self._process_node(node, parent_item=None, global_order=global_order)

    def _find_node_by_element_id(self, element_id: int) -> Optional[TreeNode]:
        """Find a tree node by its element ID."""

        def search_node(node: TreeNode) -> Optional[TreeNode]:
            if node.element.id == element_id:
                return node
            for child in node.children:
                result = search_node(child)
                if result:
                    return result
            return None

        for root in self.doc_structure.tree_roots:
            result = search_node(root)
            if result:
                return result
        return None

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

        # Determine the actual parent based on containment tree
        if parent_item is None:
            parent_node = self._find_parent_node(node)
            if parent_node and parent_node.element.id in self.element_to_item:
                parent_item = self.element_to_item[parent_node.element.id]

        # Check if this element is part of a group
        group_info = self._get_group_for_element(element.id)

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

        # Extract text from all elements
        all_texts = []
        all_provs = []

        for i, element in enumerate(elements):
            text = self._extract_text_from_bbox(element.bbox)
            all_texts.append(text)

            # Calculate character span
            start_char = sum(len(t) + 1 for t in all_texts[:-1]) if i > 0 else 0
            end_char = start_char + len(text)

            prov = ProvenanceItem(
                page_no=1, bbox=element.bbox, charspan=(start_char, end_char)
            )
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
        # Extract text
        text = self._extract_text_from_bbox(element.bbox)

        # Create provenance
        prov = ProvenanceItem(page_no=1, bbox=element.bbox, charspan=(0, len(text)))

        # Create item based on label
        return self._create_item_by_label(element.label, text, prov, element, parent)

    def _create_item_by_label(
        self,
        label: str,
        text: str,
        prov: ProvenanceItem,
        element: CVATElement,
        parent: Optional[NodeItem],
    ) -> Optional[NodeItem]:
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
            _logger.warning(
                f"Container {container_id} not found for {'caption' if is_caption else 'footnote'}"
            )
            return

        # Check if container supports captions/footnotes
        if not hasattr(container_item, "captions" if is_caption else "footnotes"):
            _logger.warning(
                f"Container {container_id} does not support {'captions' if is_caption else 'footnotes'}"
            )
            return

        # Get target element
        target_element = self.doc_structure.get_element_by_id(target_id)
        if not target_element:
            return

        # Extract text
        text = self._extract_text_from_bbox(target_element.bbox)

        # Create provenance
        prov = ProvenanceItem(
            page_no=1, bbox=target_element.bbox, charspan=(0, len(text))
        )

        # Create caption/footnote item
        label = DocItemLabel.CAPTION if is_caption else DocItemLabel.FOOTNOTE
        item = self.doc.add_text(
            label=label,
            text=text,
            prov=prov,
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

    def _extract_text_from_bbox(self, bbox: BoundingBox) -> str:
        """Extract text from bounding box using OCR results."""
        try:
            text_parts = []

            for text, confidence, coords in self.ocr_results:
                # coords are in pixels: (x0, y0, x1, y1)
                ocr_x0, ocr_y0, ocr_x1, ocr_y1 = coords

                # Create OCR bounding box
                ocr_bbox = BoundingBox(
                    l=ocr_x0,
                    r=ocr_x1,
                    t=ocr_y0,
                    b=ocr_y1,
                    coord_origin=CoordOrigin.TOPLEFT,
                )

                # Check intersection
                if bbox.intersection_over_union(ocr_bbox) > 0.1:
                    text_parts.append(text)

            return " ".join(text_parts).strip()

        except Exception as e:
            _logger.error(f"Error extracting text: {e}")
            return ""


def convert_cvat_to_docling(
    xml_path: Path, image_path: Path, ocr_framework: str = "vision"
) -> Optional[DoclingDocument]:
    """Convert a CVAT annotation to DoclingDocument.

    Args:
        xml_path: Path to CVAT XML file
        image_path: Path to document image
        ocr_framework: OCR framework to use ("vision" or "livetext")

    Returns:
        DoclingDocument or None if conversion fails
    """
    try:
        # Load image
        image = PILImage.open(image_path)

        # Create DocumentStructure
        doc_structure = DocumentStructure.from_cvat_xml(xml_path, image_path.name)

        # Create converter
        converter = CVATToDoclingConverter(
            doc_structure, image, ocr_framework, image_path.name
        )

        # Convert
        return converter.convert()

    except Exception as e:
        _logger.error(f"Failed to convert CVAT to DoclingDocument: {e}")
        return None
