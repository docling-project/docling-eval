"""Core document structure representation for CVAT annotations.

This module provides the DocumentStructure class which encapsulates all core data structures
(elements, paths, containment tree, and path mappings) and their construction.
"""

import logging
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from docling_core.types.doc.base import BoundingBox

from .folder_models import CVATDocument, CVATFolderStructure
from .models import CVATAnnotationPath, CVATElement, CVATImageInfo
from .parser import ParsedCVATFile, ParsedCVATImage, parse_cvat_file
from .path_mappings import (
    PathMappings,
    associate_paths_to_containers,
    map_path_points_to_elements,
)
from .tree import (
    TreeNode,
    build_containment_tree,
    find_node_by_element_id,
    index_tree_by_element_id,
)
from .utils import DEFAULT_PROXIMITY_THRESHOLD

_logger = logging.getLogger(__name__)


@dataclass
class DocumentStructure:
    """Core document structure containing all first-level data structures.

    This class encapsulates the core data structures needed to represent a document's
    structure from CVAT annotations. It handles the construction of these structures
    and provides a clean interface for downstream use cases (validation, analysis, etc.).
    """

    elements: List[CVATElement]
    paths: List[CVATAnnotationPath]
    tree_roots: List[TreeNode]
    path_mappings: PathMappings
    path_to_container: Dict[int, TreeNode]
    image_info: CVATImageInfo
    _node_index: Dict[int, TreeNode] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        self._node_index = index_tree_by_element_id(self.tree_roots)

    @classmethod
    def from_cvat_xml(
        cls,
        xml_path: Path,
        image_filename: str,
        proximity_thresh: float = DEFAULT_PROXIMITY_THRESHOLD,
    ) -> "DocumentStructure":
        """Construct a DocumentStructure from a CVAT XML file.

        Args:
            xml_path: Path to the CVAT XML file
            image_filename: Name of the image file to process
            proximity_thresh: Distance threshold for point-to-element mapping

        Returns:
            DocumentStructure containing all core data structures
        """
        parsed_file = parse_cvat_file(xml_path)
        parsed_image = parsed_file.get_image(image_filename)
        return cls.from_parsed_image(parsed_image, proximity_thresh=proximity_thresh)

    @classmethod
    def from_parsed_image(
        cls,
        parsed_image: ParsedCVATImage,
        proximity_thresh: float = DEFAULT_PROXIMITY_THRESHOLD,
    ) -> "DocumentStructure":
        """Construct a DocumentStructure from a parsed CVAT image."""
        elements = [element.model_copy(deep=True) for element in parsed_image.elements]
        paths = [path.model_copy(deep=True) for path in parsed_image.paths]
        image_info = parsed_image.image_info.model_copy(deep=True)

        # Build containment tree
        tree_roots = build_containment_tree(elements)

        # Create path mappings
        path_mappings = map_path_points_to_elements(
            paths, elements, proximity_thresh=proximity_thresh
        )
        path_mappings, path_to_container = associate_paths_to_containers(
            path_mappings, tree_roots, paths
        )

        return cls(
            elements=elements,
            paths=paths,
            tree_roots=tree_roots,
            path_mappings=path_mappings,
            path_to_container=path_to_container,
            image_info=image_info,
        )

    @classmethod
    def from_cvat_document(
        cls,
        cvat_document: CVATDocument,
        proximity_thresh: float = DEFAULT_PROXIMITY_THRESHOLD,
    ) -> "DocumentStructure":
        """Construct a DocumentStructure from a reconstructed CVAT document."""

        all_elements: List[CVATElement] = []
        all_paths: List[CVATAnnotationPath] = []
        image_infos: List[CVATImageInfo] = []

        cumulative_width = 0.0

        sorted_pages = sorted(cvat_document.pages, key=lambda p: p.page_number)

        parsed_cache: Dict[Path, ParsedCVATFile] = {}

        for page_info in sorted_pages:
            parsed_file = parsed_cache.get(page_info.xml_path)
            if parsed_file is None:
                parsed_file = parse_cvat_file(page_info.xml_path)
                parsed_cache[page_info.xml_path] = parsed_file

            parsed_image = parsed_file.get_image(page_info.image_filename)

            elements = [
                element.model_copy(deep=True) for element in parsed_image.elements
            ]
            paths = [path.model_copy(deep=True) for path in parsed_image.paths]
            image_info = parsed_image.image_info.model_copy(deep=True)
            image_infos.append(image_info)

            element_offset = (
                (max(e.id for e in all_elements) + 1) if all_elements else 0
            )
            path_offset = (max(p.id for p in all_paths) + 1) if all_paths else 0

            page_offset = cumulative_width
            cumulative_width += image_info.width

            adjusted_elements: List[CVATElement] = []
            for element in elements:
                bbox = element.bbox
                if page_offset:
                    bbox = BoundingBox(
                        l=bbox.l + page_offset,
                        r=bbox.r + page_offset,
                        t=bbox.t,
                        b=bbox.b,
                        coord_origin=bbox.coord_origin,
                    )

                adjusted_elements.append(
                    element.model_copy(
                        update={
                            "id": element.id + element_offset,
                            "bbox": bbox,
                        }
                    )
                )

            adjusted_paths: List[CVATAnnotationPath] = []
            for path in paths:
                points = path.points
                if page_offset:
                    points = [(x + page_offset, y) for x, y in points]

                adjusted_paths.append(
                    path.model_copy(
                        update={
                            "id": path.id + path_offset,
                            "points": points,
                        }
                    )
                )

            all_elements.extend(adjusted_elements)
            all_paths.extend(adjusted_paths)

        if not image_infos:
            raise ValueError(f"No pages found for document {cvat_document.doc_hash}")

        total_width = sum(info.width for info in image_infos)
        max_height = max(info.height for info in image_infos)

        combined_image_info = CVATImageInfo(
            width=total_width,
            height=max_height,
            name=cvat_document.doc_name,
        )

        tree_roots = build_containment_tree(all_elements)

        path_mappings = map_path_points_to_elements(
            all_paths, all_elements, proximity_thresh=proximity_thresh
        )
        path_mappings, path_to_container = associate_paths_to_containers(
            path_mappings, tree_roots, all_paths
        )

        return cls(
            elements=all_elements,
            paths=all_paths,
            tree_roots=tree_roots,
            path_mappings=path_mappings,
            path_to_container=path_to_container,
            image_info=combined_image_info,
        )

    @classmethod
    def from_cvat_folder_structure(
        cls,
        folder_structure: CVATFolderStructure,
        doc_hash: str,
        proximity_thresh: float = DEFAULT_PROXIMITY_THRESHOLD,
    ) -> "DocumentStructure":
        """Construct a DocumentStructure for a document contained in a CVAT folder."""

        if doc_hash not in folder_structure.documents:
            raise ValueError(f"Document {doc_hash} not found in folder structure")

        cvat_document = folder_structure.documents[doc_hash]
        return cls.from_cvat_document(cvat_document, proximity_thresh)

    def get_elements_by_label(self, label: object) -> list[CVATElement]:
        return [e for e in self.elements if e.label == label]

    def get_element_by_id(self, element_id: int) -> Optional[CVATElement]:
        """Get an element by its ID."""
        return next((el for el in self.elements if el.id == element_id), None)

    def get_path_by_id(self, path_id: int) -> Optional[CVATAnnotationPath]:
        """Get a path by its ID."""
        return next((p for p in self.paths if p.id == path_id), None)

    def get_node_by_element_id(self, element_id: int) -> Optional[TreeNode]:
        """Get a tree node by its element ID."""
        node = self._node_index.get(element_id)
        if node is not None:
            return node

        node = find_node_by_element_id(self.tree_roots, element_id)
        if node is not None:
            self._node_index[element_id] = node
        return node

    @cached_property
    def _global_reading_order_positions(self) -> Dict[int, int]:
        """Build global reading order position map (element_id -> position).

        Cached for performance since this is used by both validation and conversion.
        """
        positions: Dict[int, int] = {}
        for ro_path_elements in self.path_mappings.reading_order.values():
            for pos, element_id in enumerate(ro_path_elements):
                # Use first occurrence if element appears multiple times
                if element_id not in positions:
                    positions[element_id] = pos
        return positions

    def get_corrected_merge_elements(
        self, merge_path_id: int, element_ids: List[int]
    ) -> Tuple[List[int], bool]:
        """Get merge elements in correct reading order.

        Args:
            merge_path_id: The merge path ID (for logging)
            element_ids: List of element IDs in the merge path

        Returns:
            Tuple of (corrected_element_ids, was_corrected)
        """
        if len(element_ids) < 2:
            return element_ids, False

        # Get reading order positions
        positions = self._global_reading_order_positions
        elements_with_pos = [
            (eid, positions.get(eid, float("inf"))) for eid in element_ids
        ]

        # If at least 2 elements are in reading order, sort by it
        elements_in_ro = [eid for eid, pos in elements_with_pos if pos != float("inf")]
        if len(elements_in_ro) < 2:
            return element_ids, False

        # Sort by reading order position
        sorted_ids = [eid for eid, _ in sorted(elements_with_pos, key=lambda x: x[1])]

        # Check if order changed
        if element_ids != sorted_ids:
            _logger.debug(
                f"Merge path {merge_path_id}: Auto-correcting backwards merge "
                f"(was {element_ids}, now {sorted_ids})"
            )
            return sorted_ids, True

        return element_ids, False
