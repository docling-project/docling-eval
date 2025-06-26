"""Core document structure representation for CVAT annotations.

This module provides the DocumentStructure class which encapsulates all core data structures
(elements, paths, containment tree, and path mappings) and their construction.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import CVATAnnotationPath, CVATElement, CVATImageInfo
from .parser import parse_cvat_xml_for_image
from .path_mappings import (
    PathMappings,
    associate_paths_to_containers,
    map_path_points_to_elements,
)
from .tree import TreeNode, build_containment_tree, find_node_by_element_id
from .utils import DEFAULT_PROXIMITY_THRESHOLD


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
        # Parse XML
        elements, paths, image_info = parse_cvat_xml_for_image(xml_path, image_filename)

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

    def get_element_by_id(self, element_id: int) -> Optional[CVATElement]:
        """Get an element by its ID."""
        return next((el for el in self.elements if el.id == element_id), None)

    def get_path_by_id(self, path_id: int) -> Optional[CVATAnnotationPath]:
        """Get a path by its ID."""
        return next((p for p in self.paths if p.id == path_id), None)

    def get_node_by_element_id(self, element_id: int) -> Optional[TreeNode]:
        """Get a tree node by its element ID."""
        return find_node_by_element_id(self.tree_roots, element_id)
