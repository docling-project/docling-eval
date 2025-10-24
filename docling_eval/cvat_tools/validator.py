import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Type

from docling_core.types.doc.document import ContentLayer, DocItemLabel

from .document import DocumentStructure
from .folder_models import CVATDocument
from .models import (
    CVATElement,
    CVATValidationError,
    CVATValidationReport,
    TableStructLabel,
    ValidationSeverity,
)
from .parser import ParsedCVATFile, parse_cvat_file
from .tree import find_ancestor
from .utils import DEFAULT_PROXIMITY_THRESHOLD, find_elements_containing_point

logger = logging.getLogger("docling_eval.cvat_tools.validator")


class ValidationRule(ABC):
    """Base class for validation rules."""

    @abstractmethod
    def validate(self, doc: DocumentStructure) -> List[CVATValidationError]:
        """Validate the context and return a list of errors."""
        pass


class ValidLabelsRule(ValidationRule):
    """Validate that all element labels are valid DocItemLabels."""

    def validate(self, doc: DocumentStructure) -> List[CVATValidationError]:
        errors: list[CVATValidationError] = []
        for el in doc.elements:
            try:
                _ = el.label  # This will raise ValueError if invalid
            except ValueError:
                errors.append(
                    CVATValidationError(
                        error_type="invalid_label",
                        message=f"Element {el.id} has invalid label '{el.label}'",
                        severity=ValidationSeverity.WARNING,
                        element_id=el.id,
                    )
                )
        return errors


class ReadingOrderRule(ValidationRule):
    """Validate reading order requirements - FATAL level."""

    def validate(self, doc: DocumentStructure) -> List[CVATValidationError]:
        errors: list[CVATValidationError] = []

        # Find all first-level reading order paths
        level1_paths = [
            p
            for p in doc.paths
            if p.label.startswith("reading_order") and (p.level == 1 or p.level is None)
        ]

        # Skip reading order validation for samples with very few elements
        # (0 or 1 rectangle elements don't need reading order)
        if len(doc.elements) <= 1:
            return errors

        # Check missing first-level reading order
        if len(level1_paths) == 0:
            errors.append(
                CVATValidationError(
                    error_type="missing_first_level_reading_order",
                    message="No first-level reading-order path found.",
                    severity=ValidationSeverity.FATAL,
                )
            )
        # Check multiple first-level reading order
        elif len(level1_paths) > 1:
            errors.append(
                CVATValidationError(
                    error_type="multiple_first_level_reading_order",
                    message=f"Found {len(level1_paths)} first-level reading-order paths. Only one is allowed.",
                    severity=ValidationSeverity.FATAL,
                    path_ids=[p.id for p in level1_paths],
                )
            )

        return errors


class SecondLevelReadingOrderParentRule(ValidationRule):
    """Validate that second-level reading order paths have parent containers - WARNING level."""

    def validate(self, doc: DocumentStructure) -> List[CVATValidationError]:
        errors: list[CVATValidationError] = []
        for p in doc.paths:
            if p.label.startswith("reading_order") and p.level and p.level > 1:
                container = doc.path_to_container.get(p.id)
                if container is None or container.parent is None:
                    errors.append(
                        CVATValidationError(
                            error_type="second_level_reading_order_no_parent",
                            message=f"Second-level reading-order path {p.id} has no parent container.",
                            severity=ValidationSeverity.WARNING,
                            path_id=p.id,
                        )
                    )
        return errors


class ElementTouchedByReadingOrderRule(ValidationRule):
    """Validate that every non-background element is touched by a reading order path.

    Severity levels:
    - ERROR: Elements not touched by reading order (default case)
    - ERROR: Elements inside tables not touched by second-level (level 2+) reading order
    - WARNING: Elements inside regular picture containers (not chart/infographic/illustration)

    Table-specific validation:
    - Elements inside table containers MUST be in second-level reading order
    - Exceptions: checkbox elements, picture elements, key/value elements (GraphCellLabel), table structure elements (TableStructLabel)

    Exemptions (no error reported):
    - GraphCellLabel elements (key/value) - don't need reading order anywhere
    - TableStructLabel elements (table structure) - don't need reading order anywhere
    - Checkbox elements inside tables - can be standalone
    - Picture elements inside tables - can be standalone
    - Picture elements with descendants in level 1 reading order
    - Elements inside completely untouched CHART/INFOGRAPHIC/ILLUSTRATION pictures
      (when NO descendants are in reading order, treated as atomic visual units)

    Special handling:
    - Elements inside partially annotated CHART/INFOGRAPHIC/ILLUSTRATION pictures
      (where SOME descendants are in reading order) still get ERROR if untouched
    """

    # Picture types that get special exemption handling (atomic visual units when completely untouched)
    ATOMIC_PICTURE_TYPES = frozenset(["CHART", "INFOGRAPHIC", "ILLUSTRATION"])

    def _is_merge_group_inside_container(
        self,
        element: CVATElement,
        container_element: CVATElement,
        doc: DocumentStructure,
        threshold: float = 0.7,
    ) -> bool:
        """Check if an element's merge group is substantially inside a container.

        For merged elements, ALL parts of the merge group must be inside the container.
        If any element in a merge group is outside, the entire group is considered outside.

        Args:
            element: Element to check
            container_element: Container element to check against
            doc: Document structure
            threshold: Fraction of element that must be inside (default 0.7 = 70%)

        Returns:
            True if all elements in the merge group are >threshold inside the container
        """
        from .geometry import bbox_fraction_inside

        # Get all elements in the merge group (or just this element if not merged)
        element_ids_to_check = {element.id}

        # Find merge group containing this element
        for merge_elements in doc.path_mappings.merge.values():
            if element.id in merge_elements:
                element_ids_to_check = set(merge_elements)
                break

        # Check all elements in the group
        for el_id in element_ids_to_check:
            el = next((e for e in doc.elements if e.id == el_id), None)
            if not el:
                continue

            # Check spatial containment for each element in the group
            fraction = bbox_fraction_inside(el.bbox, container_element.bbox)

            # If any element in the merge group is not substantially inside,
            # treat the entire group as outside the container
            if fraction <= threshold:
                return False

        return True

    def _is_element_inside_table(
        self, element: CVATElement, doc: DocumentStructure
    ) -> bool:
        """Check if an element is inside a table container.

        For merged elements, ALL parts of the merge group must be inside the table.
        If any element in a merge group is outside, the entire group is considered outside.
        """
        node = doc.get_node_by_element_id(element.id)
        if not node:
            return False

        # Check if element has a TABLE ancestor in containment tree
        table_ancestor = find_ancestor(
            node, lambda ancestor: ancestor.element.label == DocItemLabel.TABLE
        )

        if not table_ancestor:
            return False

        # Check if the merge group is substantially inside the table
        return self._is_merge_group_inside_container(
            element, table_ancestor.element, doc, threshold=0.7
        )

    def _is_element_inside_regular_picture(
        self, element: CVATElement, doc: DocumentStructure
    ) -> bool:
        """Check if an element is inside a regular picture container.

        Regular pictures are those that are NOT chart/infographic/illustration types.
        Chart/infographic/illustration pictures get special exemption handling.

        For merged elements, ALL parts of the merge group must be inside the same
        regular picture. If any element is outside or in a different container,
        the entire group is considered outside.
        """
        node = doc.get_node_by_element_id(element.id)
        if not node:
            return False

        # Find if element has a picture ancestor that is NOT chart/infographic/illustration
        picture_ancestor = find_ancestor(
            node, lambda ancestor: ancestor.element.label == DocItemLabel.PICTURE
        )

        if not picture_ancestor:
            return False

        # Check if the picture is an atomic type (chart/infographic/illustration)
        picture_type = picture_ancestor.element.type
        if picture_type and picture_type.upper() in self.ATOMIC_PICTURE_TYPES:
            return False

        # For merged elements, check that ALL are inside the same regular picture
        element_ids_to_check = {element.id}

        # Find merge group containing this element
        for merge_elements in doc.path_mappings.merge.values():
            if element.id in merge_elements:
                element_ids_to_check = set(merge_elements)
                break

        # Check all elements in the merge group
        for el_id in element_ids_to_check:
            if el_id == element.id:
                continue  # Already checked above

            el_node = doc.get_node_by_element_id(el_id)
            if not el_node:
                # Element not in tree - treat as outside
                return False

            # Check if this element has the same picture ancestor
            el_picture_ancestor = find_ancestor(
                el_node, lambda ancestor: ancestor.element.label == DocItemLabel.PICTURE
            )

            # If element has no picture ancestor or different picture, treat group as outside
            if (
                not el_picture_ancestor
                or el_picture_ancestor.element.id != picture_ancestor.element.id
            ):
                return False

            # Check if this element's picture is also non-atomic
            el_picture_type = el_picture_ancestor.element.type
            if el_picture_type and el_picture_type.upper() in self.ATOMIC_PICTURE_TYPES:
                return False

        return True

    def validate(self, doc: DocumentStructure) -> List[CVATValidationError]:
        from .models import GraphCellLabel

        errors: list[CVATValidationError] = []

        # Skip reading order validation for samples with very few elements
        # (0 or 1 elements don't need reading order)
        if len(doc.elements) <= 1:
            return errors

        touched = set()
        for elist in doc.path_mappings.reading_order.values():
            touched.update(elist)

        logger.debug(f"Total elements: {len(doc.elements)}")
        logger.debug(f"Total touched elements: {len(touched)}")
        logger.debug(f"Touched element IDs: {sorted(touched)}")

        # Print containment tree structure
        logger.debug("\nCONTAINMENT TREE STRUCTURE:")
        for i, root in enumerate(doc.tree_roots):
            logger.debug(f"Tree root {i}:")
            self._print_tree_node(root, 0)

        # Print all elements with their details
        logger.debug("\nALL ELEMENTS:")
        for el in doc.elements:
            node = doc.get_node_by_element_id(el.id)
            parent_id = node.parent.element.id if node and node.parent else None
            descendants = node.get_descendant_ids() if node else set()
            logger.debug(
                f"Element {el.id} ({el.label}) - type: {el.type}, parent: {parent_id}, descendants: {sorted(descendants)}"
            )

        # Pre-compute level 1 reading order touched elements for efficiency
        level1_touched = set()
        level2_plus_touched = set()
        for path_id, element_ids in doc.path_mappings.reading_order.items():
            path = doc.get_path_by_id(path_id)
            if path and path.label.startswith("reading_order"):
                if path.level == 1 or path.level is None:
                    level1_touched.update(element_ids)
                elif path.level and path.level > 1:
                    level2_plus_touched.update(element_ids)

        logger.debug(f"Level 1 touched elements: {len(level1_touched)}")
        logger.debug(f"Level 1 touched element IDs: {sorted(level1_touched)}")
        logger.debug(f"Level 2+ touched elements: {len(level2_plus_touched)}")
        logger.debug(f"Level 2+ touched element IDs: {sorted(level2_plus_touched)}")

        # Pre-compute all descendant IDs for efficiency (avoid repeated tree traversal)
        element_descendants = {}
        for el in doc.elements:
            node = doc.get_node_by_element_id(el.id)
            if node:
                element_descendants[el.id] = node.get_descendant_ids()

        # Collect elements that would fail reading order validation
        # Split into table elements and non-table elements for different validation rules
        untouched_elements = []
        untouched_table_elements = []

        for el in doc.elements:
            if el.content_layer == ContentLayer.BACKGROUND:
                continue

            # Skip validation for GraphCellLabel elements (key/value) - they don't need reading order
            if isinstance(el.label, GraphCellLabel):
                continue

            # Skip validation for TableStructLabel elements - they don't need reading order
            if isinstance(el.label, TableStructLabel):
                continue

            # Handle elements inside table containers separately
            if self._is_element_inside_table(el, doc):
                # Checkboxes and pictures inside tables don't need reading order
                if el.label in [
                    DocItemLabel.CHECKBOX_SELECTED,
                    DocItemLabel.CHECKBOX_UNSELECTED,
                    DocItemLabel.PICTURE,
                ]:
                    continue

                # Check if touched by level 2+ reading order
                descendant_ids = element_descendants.get(el.id, set())
                if not (descendant_ids & level2_plus_touched):
                    untouched_table_elements.append(el)
                continue

            # Skip validation for picture elements that have descendants touched by level 1 reading order
            if (
                el.label == DocItemLabel.PICTURE
                and self._has_descendants_touched_by_level1_reading_order_fast(
                    el.id, element_descendants, level1_touched
                )
            ):
                logger.debug(
                    f"Skipping picture element {el.id} (has descendants touched by level 1)"
                )
                continue

            descendant_ids = element_descendants.get(el.id, set())
            if not (descendant_ids & touched):
                untouched_elements.append(el)

        logger.debug(
            f"Untouched elements before CHART/INFOGRAPHIC logic: {len(untouched_elements)}"
        )
        logger.debug(f"Untouched element IDs: {[el.id for el in untouched_elements]}")

        # Apply special condition for atomic picture elements (CHART/INFOGRAPHIC/ILLUSTRATION)
        if untouched_elements:
            # Find atomic pictures that have NO touched descendants (completely untouched)
            # Pre-compute mapping of element_id -> parent atomic picture_id for efficiency
            element_to_untouched_atomic_picture: Dict[int, int] = {}

            for el in doc.elements:
                if (
                    el.label == DocItemLabel.PICTURE
                    and el.type
                    and el.type.upper() in self.ATOMIC_PICTURE_TYPES
                ):
                    logger.debug(
                        f"Found atomic picture element {el.id} with type '{el.type}'"
                    )
                    picture_descendants = element_descendants.get(el.id, set())
                    # Exclude the picture element itself from touched elements inside
                    touched_inside = (picture_descendants & touched) - {el.id}
                    logger.debug(
                        f"Picture {el.id} descendants: {sorted(picture_descendants)}"
                    )
                    logger.debug(
                        f"Picture {el.id} touched_inside: {sorted(touched_inside)}"
                    )
                    # Only add if the picture has NO touched elements (excluding the picture itself)
                    if not touched_inside:
                        # Map all descendants to this untouched atomic picture
                        for desc_id in picture_descendants:
                            element_to_untouched_atomic_picture[desc_id] = el.id
                        logger.debug(
                            f"Picture {el.id} is completely untouched, mapped {len(picture_descendants)} descendants"
                        )
                    else:
                        logger.debug(
                            f"Picture {el.id} has touched descendants, not exempt"
                        )

            logger.debug(
                f"Elements in untouched atomic pictures: {len(element_to_untouched_atomic_picture)}"
            )

            # Report errors for elements not in completely untouched atomic pictures
            for el in untouched_elements:
                # Skip if element is in a completely untouched atomic picture
                if el.id in element_to_untouched_atomic_picture:
                    picture_id = element_to_untouched_atomic_picture[el.id]
                    logger.debug(
                        f"Skipping element {el.id} - inside completely untouched atomic picture {picture_id}"
                    )
                    continue

                logger.debug(f"Adding error for element {el.id} ({el.label})")

                # Check if element is inside any NON-chart/infographic/illustration picture container
                is_inside_regular_picture = self._is_element_inside_regular_picture(
                    el, doc
                )

                if is_inside_regular_picture:
                    # WARNING level for elements inside regular picture containers (not chart/infographic/illustration)
                    errors.append(
                        CVATValidationError(
                            error_type="element_not_touched_by_reading_order_inside_picture",
                            message=f"Element {el.id} ({el.label}) inside picture container not touched by any reading-order path.",
                            severity=ValidationSeverity.WARNING,
                            element_id=el.id,
                        )
                    )
                else:
                    # ERROR level for all other elements
                    errors.append(
                        CVATValidationError(
                            error_type="element_not_touched_by_reading_order",
                            message=f"Element {el.id} ({el.label}) not touched by any reading-order path.",
                            severity=ValidationSeverity.ERROR,
                            element_id=el.id,
                        )
                    )

        # Report errors for elements inside tables not touched by level 2+ reading order
        logger.debug(f"Untouched table elements: {len(untouched_table_elements)}")
        logger.debug(
            f"Untouched table element IDs: {[el.id for el in untouched_table_elements]}"
        )

        for el in untouched_table_elements:
            errors.append(
                CVATValidationError(
                    error_type="element_not_touched_by_reading_order_inside_table",
                    message=f"Element {el.id} ({el.label}) inside table container not touched by second-level reading-order path.",
                    severity=ValidationSeverity.ERROR,
                    element_id=el.id,
                )
            )

        return errors

    def _has_descendants_touched_by_level1_reading_order_fast(
        self,
        element_id: int,
        element_descendants: Dict[int, Set[int]],
        level1_touched: Set[int],
    ) -> bool:
        """Fast check if a picture element has descendants touched by level 1 reading order."""
        descendant_ids = element_descendants.get(element_id, set())
        return bool(descendant_ids & level1_touched)

    def _print_tree_node(self, node, depth: int):
        """Helper method to print tree structure recursively."""
        indent = "  " * depth
        el = node.element
        logger.debug(f"{indent}Element {el.id} ({el.label}) - type: {el.type}")
        for child in node.children:
            self._print_tree_node(child, depth + 1)


class MergeGroupPathsRule(ValidationRule):
    """Validate merge and group paths - ERROR level."""

    def validate(self, doc: DocumentStructure) -> List[CVATValidationError]:
        errors: list[CVATValidationError] = []

        # Validate merge paths
        errors.extend(
            self._validate_path_mappings(doc.elements, doc.path_mappings.merge, "merge")
        )

        # Validate group paths
        errors.extend(
            self._validate_path_mappings(doc.elements, doc.path_mappings.group, "group")
        )

        return errors

    def _validate_path_mappings(
        self,
        elements: List[CVATElement],
        path_mappings: Dict[int, List[int]],
        path_type: str,
    ) -> List[CVATValidationError]:
        """Validate that elements in path mappings have same label and content_layer."""
        if not elements or not path_mappings:
            return []

        errors: list[CVATValidationError] = []
        id_to_element = {el.id: el for el in elements}

        for path_id, el_ids in path_mappings.items():
            if len(el_ids) < 2:
                continue

            # Check that all elements exist
            elements_in_group = []
            missing = False
            for el_id in el_ids:
                el = id_to_element.get(el_id)
                if not el:
                    errors.append(
                        CVATValidationError(
                            error_type=f"{path_type}_path_missing_element",
                            message=f"{path_type.capitalize()} path {path_id}: Element {el_id} not found",
                            severity=ValidationSeverity.ERROR,
                            path_id=path_id,
                        )
                    )
                    missing = True
            if missing:
                continue

            elements_in_group = [id_to_element[el_id] for el_id in el_ids]
            labels = {el.label for el in elements_in_group}
            first_el = elements_in_group[0]

            # Check label consistency based on path type and content
            # A group path is a "list group" if the first element is a list_item
            # List groups can have mixed labels (list items grouped with their child content)
            # Other groups must have uniform labels (except checkbox special case)
            is_list_group = (
                path_type == "group" and first_el.label == DocItemLabel.LIST_ITEM
            )

            if is_list_group:
                # List groups can have mixed labels - skip label check
                pass
            elif labels == {"checkbox_selected", "checkbox_unselected"}:
                # Special case for checkboxes - mixed labels allowed
                pass
            elif len(labels) > 1:
                errors.append(
                    CVATValidationError(
                        error_type=f"{path_type}_path_different_labels",
                        message=f"{path_type.capitalize()} path {path_id}: Elements have different labels: {sorted(labels)}",
                        severity=ValidationSeverity.ERROR,
                        path_id=path_id,
                    )
                )

            # Check same content_layer
            content_layers = {el.content_layer for el in elements_in_group}
            if len(content_layers) > 1:
                errors.append(
                    CVATValidationError(
                        error_type=f"{path_type}_path_different_content_layers",
                        message=f"{path_type.capitalize()} path {path_id}: Elements have different content layers: {sorted(content_layers)}",
                        severity=ValidationSeverity.ERROR,
                        path_id=path_id,
                    )
                )

            # For list groups, check that all list_item elements have same level
            if is_list_group:
                levels = {
                    el.level
                    for el in elements_in_group
                    if el.label == DocItemLabel.LIST_ITEM
                }
                if None in levels:
                    errors.append(
                        CVATValidationError(
                            error_type="group_path_list_items_missing_level",
                            message=f"Group path {path_id}: Some list_item elements missing level attribute",
                            severity=ValidationSeverity.ERROR,
                            path_id=path_id,
                        )
                    )
                elif len(levels) > 1:
                    # Filter out None values for sorting (we know None is not in levels due to elif)
                    non_none_levels = [lv for lv in levels if lv is not None]
                    errors.append(
                        CVATValidationError(
                            error_type="group_path_list_items_different_levels",
                            message=f"Group path {path_id}: list_item elements have different levels: {sorted(non_none_levels)}",
                            severity=ValidationSeverity.ERROR,
                            path_id=path_id,
                        )
                    )

        return errors


class MergePathDirectionRule(ValidationRule):
    """Validate that merge path direction matches reading order - ERROR level."""

    def validate(self, doc: DocumentStructure) -> List[CVATValidationError]:
        errors: list[CVATValidationError] = []

        # Check each merge path using the centralized helper
        for path_id, element_ids in doc.path_mappings.merge.items():
            corrected_ids, was_backwards = doc.get_corrected_merge_elements(
                path_id, element_ids
            )

            if was_backwards:
                errors.append(
                    CVATValidationError(
                        error_type="merge_path_backwards",
                        message=f"Merge path {path_id}: Direction is backwards relative to reading order "
                        f"(merge: {element_ids}, reading order: {corrected_ids}). ",
                        severity=ValidationSeverity.WARNING,
                        path_id=path_id,
                    )
                )

        return errors


class CaptionFootnotePathsRule(ValidationRule):
    """Validate caption and footnote paths - ERROR level."""

    def _validate_basic_requirements(
        self,
        elements: List[CVATElement],
        to_caption: Dict[int, Tuple[int, int]],
        to_footnote: Dict[int, Tuple[int, int]],
    ) -> List[CVATValidationError]:
        """Validate basic caption and footnote path requirements.

        Rules:
        1. Starting point of to_caption and to_footnote paths must be on a container element
        2. End point must be on a caption or footnote element, respectively
        """
        from .utils import is_caption_element, is_container_element, is_footnote_element

        errors: List[CVATValidationError] = []
        id_to_element = {el.id: el for el in elements}

        # Validate to_caption paths
        for path_id, (container_id, caption_id) in to_caption.items():
            container = id_to_element.get(container_id)
            caption = id_to_element.get(caption_id)

            if not container or not is_container_element(container):
                errors.append(
                    CVATValidationError(
                        error_type="caption_footnote_path_error",
                        message=f"Caption path {path_id}: Starting point is not a container element",
                        severity=ValidationSeverity.ERROR,
                        path_id=path_id,
                    )
                )
            if not caption or not is_caption_element(caption):
                errors.append(
                    CVATValidationError(
                        error_type="caption_footnote_path_error",
                        message=f"Caption path {path_id}: End point is not a caption element",
                        severity=ValidationSeverity.ERROR,
                        path_id=path_id,
                    )
                )

        # Validate to_footnote paths
        for path_id, (container_id, footnote_id) in to_footnote.items():
            container = id_to_element.get(container_id)
            footnote = id_to_element.get(footnote_id)

            if not container or not is_container_element(container):
                errors.append(
                    CVATValidationError(
                        error_type="caption_footnote_path_error",
                        message=f"Footnote path {path_id}: Starting point is not a container element",
                        severity=ValidationSeverity.ERROR,
                        path_id=path_id,
                    )
                )
            if not footnote or not is_footnote_element(footnote):
                errors.append(
                    CVATValidationError(
                        error_type="caption_footnote_path_error",
                        message=f"Footnote path {path_id}: End point is not a footnote element",
                        severity=ValidationSeverity.ERROR,
                        path_id=path_id,
                    )
                )

        return errors

    def validate(self, doc: DocumentStructure) -> List[CVATValidationError]:
        from .utils import is_container_element

        errors: list[CVATValidationError] = []
        id_to_element = {el.id: el for el in doc.elements}

        # First check for the special case where neither side is a container (ERROR)
        caption_paths_to_skip = set()
        for path_id, (container_id, caption_id) in doc.path_mappings.to_caption.items():
            container_el = id_to_element.get(container_id)
            caption_el = id_to_element.get(caption_id)

            if container_el and caption_el:
                if not is_container_element(container_el) and not is_container_element(
                    caption_el
                ):
                    errors.append(
                        CVATValidationError(
                            error_type="caption_path_no_container",
                            message=f"Caption path {path_id}: Neither element {container_id} ({container_el.label}) nor element {caption_id} ({caption_el.label}) is a container type",
                            severity=ValidationSeverity.ERROR,
                            path_id=path_id,
                        )
                    )
                    caption_paths_to_skip.add(path_id)

        footnote_paths_to_skip = set()
        for path_id, (
            container_id,
            footnote_id,
        ) in doc.path_mappings.to_footnote.items():
            container_el = id_to_element.get(container_id)
            footnote_el = id_to_element.get(footnote_id)

            if container_el and footnote_el:
                if not is_container_element(container_el) and not is_container_element(
                    footnote_el
                ):
                    errors.append(
                        CVATValidationError(
                            error_type="footnote_path_no_container",
                            message=f"Footnote path {path_id}: Neither element {container_id} ({footnote_el.label}) nor element {footnote_id} ({footnote_el.label}) is a container type",
                            severity=ValidationSeverity.ERROR,
                            path_id=path_id,
                        )
                    )
                    footnote_paths_to_skip.add(path_id)

        # Now validate other caption/footnote path requirements (ERROR level)
        # Skip paths already reported as ERROR above to avoid duplicate errors
        errors.extend(
            self._validate_basic_requirements(
                doc.elements,
                {
                    k: v
                    for k, v in doc.path_mappings.to_caption.items()
                    if k not in caption_paths_to_skip
                },
                {
                    k: v
                    for k, v in doc.path_mappings.to_footnote.items()
                    if k not in footnote_paths_to_skip
                },
            )
        )

        # Validate caption uniqueness: each caption must be referenced by exactly one to_caption path
        caption_to_paths: Dict[int, List[int]] = {}
        for path_id, (_, caption_id) in doc.path_mappings.to_caption.items():
            caption_to_paths.setdefault(caption_id, []).append(path_id)

        # Check for captions with multiple references
        for caption_id, path_ids in caption_to_paths.items():
            if len(path_ids) > 1:
                errors.append(
                    CVATValidationError(
                        error_type="caption_multiple_references",
                        message=f"Caption element {caption_id} is referenced by multiple to_caption paths: {path_ids}",
                        severity=ValidationSeverity.ERROR,
                        element_id=caption_id,
                        path_ids=path_ids,
                    )
                )

        # Check for caption elements with no references
        for el in doc.elements:
            if el.label == DocItemLabel.CAPTION and el.id not in caption_to_paths:
                errors.append(
                    CVATValidationError(
                        error_type="caption_no_reference",
                        message=f"Caption element {el.id} is not referenced by any to_caption path",
                        severity=ValidationSeverity.ERROR,
                        element_id=el.id,
                    )
                )

        return errors


class TableStructLocationRule(ValidationRule):
    """Validate that TableStructLabel elements are inside table or document_index containers - ERROR level."""

    def validate(self, doc: DocumentStructure) -> List[CVATValidationError]:
        from .models import TableStructLabel
        from .tree import contains

        errors: list[CVATValidationError] = []

        # Find all table and document_index container elements
        containers = [
            el
            for el in doc.elements
            if el.label in [DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX]
        ]

        # Check each TableStructLabel element
        for el in doc.elements:
            if isinstance(el.label, TableStructLabel):
                # Check if contained in any table or document_index
                is_contained = any(contains(container, el) for container in containers)

                if not is_contained:
                    errors.append(
                        CVATValidationError(
                            error_type="table_struct_outside_container",
                            message=f"TableStructLabel element {el.id} ({el.label}) is not contained within a table or document_index",
                            severity=ValidationSeverity.ERROR,
                            element_id=el.id,
                        )
                    )

        return errors


class GraphCellLocationRule(ValidationRule):
    """Validate that GraphCellLabel (key/value) elements are contained in valid text-item parents - ERROR level."""

    # Valid parent labels for key/value elements
    VALID_TEXT_ITEM_LABELS = {
        DocItemLabel.TEXT,
        DocItemLabel.SECTION_HEADER,
        DocItemLabel.LIST_ITEM,
        DocItemLabel.FOOTNOTE,
        DocItemLabel.CAPTION,
        DocItemLabel.CHECKBOX_SELECTED,
        DocItemLabel.CHECKBOX_UNSELECTED,
        DocItemLabel.HANDWRITTEN_TEXT,
    }

    def validate(self, doc: DocumentStructure) -> List[CVATValidationError]:
        from .models import GraphCellLabel

        errors: list[CVATValidationError] = []

        # Check each GraphCellLabel element (key/value)
        for el in doc.elements:
            if isinstance(el.label, GraphCellLabel):
                node = doc.get_node_by_element_id(el.id)

                if not node:
                    # Element not in tree - should not happen but handle gracefully
                    errors.append(
                        CVATValidationError(
                            error_type="graph_cell_not_in_tree",
                            message=f"GraphCellLabel element {el.id} ({el.label}) not found in containment tree",
                            severity=ValidationSeverity.ERROR,
                            element_id=el.id,
                        )
                    )
                    continue

                if not node.parent:
                    # No parent - at root level
                    errors.append(
                        CVATValidationError(
                            error_type="graph_cell_no_parent",
                            message=f"GraphCellLabel element {el.id} ({el.label}) has no parent container",
                            severity=ValidationSeverity.ERROR,
                            element_id=el.id,
                        )
                    )
                    continue

                parent_label = node.parent.element.label
                if parent_label not in self.VALID_TEXT_ITEM_LABELS:
                    errors.append(
                        CVATValidationError(
                            error_type="graph_cell_invalid_parent",
                            message=f"GraphCellLabel element {el.id} ({el.label}) has invalid parent {node.parent.element.id} ({parent_label}). Must be contained in text-item labels: {sorted(str(lbl) for lbl in self.VALID_TEXT_ITEM_LABELS)}",
                            severity=ValidationSeverity.ERROR,
                            element_id=el.id,
                        )
                    )

        return errors


class ToValuePathStructureRule(ValidationRule):
    """Validate that to_value paths connect exactly 2 logical elements (after merges) - ERROR level."""

    def validate(self, doc: DocumentStructure) -> List[CVATValidationError]:
        from .utils import get_deepest_element_at_point

        errors: list[CVATValidationError] = []

        # Build element-to-merge-group mapping
        element_to_merge_group: Dict[int, Set[int]] = {}
        for merge_elements in doc.path_mappings.merge.values():
            group = set(merge_elements)
            for el_id in merge_elements:
                element_to_merge_group[el_id] = group

        # Check each to_value path
        for path in doc.paths:
            if path.label != "to_value":
                continue

            # Find which elements this path touches
            touched_elements: List[int] = []
            for pt in path.points:
                deepest = get_deepest_element_at_point(
                    pt, doc.elements, DEFAULT_PROXIMITY_THRESHOLD
                )
                if deepest:
                    if not touched_elements or touched_elements[-1] != deepest.id:
                        touched_elements.append(deepest.id)

            if len(touched_elements) < 2:
                errors.append(
                    CVATValidationError(
                        error_type="to_value_path_insufficient_elements",
                        message=f"to_value path {path.id} touches only {len(touched_elements)} element(s), expected 2",
                        severity=ValidationSeverity.ERROR,
                        path_id=path.id,
                    )
                )
                continue

            # Group elements by merge relationships to count logical elements
            logical_groups: List[Set[int]] = []
            seen_elements = set()

            for el_id in touched_elements:
                if el_id in seen_elements:
                    continue

                if el_id in element_to_merge_group:
                    group = element_to_merge_group[el_id]
                    logical_groups.append(group)
                    seen_elements.update(group)
                else:
                    logical_groups.append({el_id})
                    seen_elements.add(el_id)

            if len(logical_groups) != 2:
                errors.append(
                    CVATValidationError(
                        error_type="to_value_path_invalid_element_count",
                        message=f"to_value path {path.id} touches {len(touched_elements)} element(s) "
                        f"which resolve to {len(logical_groups)} logical group(s) after merges. Expected exactly 2 logical elements (key and value).",
                        severity=ValidationSeverity.ERROR,
                        path_id=path.id,
                    )
                )

        return errors


class GraphCellConnectionRule(ValidationRule):
    """Validate that all GraphCellLabel (key/value) elements are connected by to_value paths - ERROR level."""

    def validate(self, doc: DocumentStructure) -> List[CVATValidationError]:
        from .models import GraphCellLabel

        errors: list[CVATValidationError] = []

        # Build set of all element IDs touched by to_value paths
        connected_elements: Set[int] = set()
        for key_id, value_id in doc.path_mappings.to_value.values():
            connected_elements.add(key_id)
            connected_elements.add(value_id)

        # Build element-to-merge-group mapping to handle merged elements
        element_to_merge_group: Dict[int, Set[int]] = {}
        for merge_elements in doc.path_mappings.merge.values():
            group = set(merge_elements)
            for el_id in merge_elements:
                element_to_merge_group[el_id] = group

        # Check each GraphCellLabel element has a connection (directly or via merge)
        for el in doc.elements:
            if isinstance(el.label, GraphCellLabel):
                is_connected = el.id in connected_elements

                # Also check if element is merged with a connected element
                if not is_connected and el.id in element_to_merge_group:
                    merge_group = element_to_merge_group[el.id]
                    is_connected = bool(merge_group & connected_elements)

                if not is_connected:
                    errors.append(
                        CVATValidationError(
                            error_type="graph_cell_no_connection",
                            message=f"GraphCellLabel element {el.id} ({el.label}) is not connected by any to_value path",
                            severity=ValidationSeverity.ERROR,
                            element_id=el.id,
                        )
                    )

        return errors


class ControlPointsHitElementsRule(ValidationRule):
    """Validate that all control points of paths hit some element - WARNING level."""

    def validate(self, doc: DocumentStructure) -> List[CVATValidationError]:
        errors: list[CVATValidationError] = []

        for path in doc.paths:
            for i, pt in enumerate(path.points):
                candidates = find_elements_containing_point(
                    pt, doc.elements, DEFAULT_PROXIMITY_THRESHOLD
                )

                if not candidates:
                    errors.append(
                        CVATValidationError(
                            error_type="control_point_no_element",
                            message=f"Control point {i} of path {path.id} ({path.label}) at coordinates {pt} does not hit any element",
                            severity=ValidationSeverity.WARNING,
                            path_id=path.id,
                            point_index=i,
                            point_coords=pt,
                        )
                    )

        return errors


class MissingAttributesRule(ValidationRule):
    """Validate required element attributes are present."""

    def validate(self, doc: DocumentStructure) -> List[CVATValidationError]:
        errors: list[CVATValidationError] = []

        for el in doc.elements:
            # Check content_layer
            if el.content_layer is None:
                errors.append(
                    CVATValidationError(
                        error_type="missing_content_layer",
                        message=f"Element {el.id} missing content_layer attribute",
                        severity=ValidationSeverity.WARNING,
                        element_id=el.id,
                    )
                )

            # Check level for specific labels
            if el.label in [DocItemLabel.SECTION_HEADER, DocItemLabel.LIST_ITEM]:
                if el.level is None:
                    errors.append(
                        CVATValidationError(
                            error_type="missing_level",
                            message=f"Element {el.id} ({el.label}) missing required level attribute",
                            severity=ValidationSeverity.WARNING,
                            element_id=el.id,
                        )
                    )

        return errors


class UnrecognizedAttributesRule(ValidationRule):
    """Validate element attributes are recognized."""

    KNOWN_ATTRIBUTES = {"content_layer", "type", "level", "json"}  # Add more as needed

    def validate(self, doc: DocumentStructure) -> List[CVATValidationError]:
        errors: list[CVATValidationError] = []

        for el in doc.elements:
            for attr_name in el.attributes.keys():
                if attr_name not in self.KNOWN_ATTRIBUTES:
                    errors.append(
                        CVATValidationError(
                            error_type="unrecognized_attribute",
                            message=f"Element {el.id} has unrecognized attribute '{attr_name}'",
                            severity=ValidationSeverity.WARNING,
                            element_id=el.id,
                        )
                    )

        return errors


class GroupConsecutiveReadingOrderRule(ValidationRule):
    """Validate that group paths with list items don't have ungrouped elements sandwiched in reading order.

    This validator flags cases where elements appear in reading order between grouped list items
    but are not included in the group path. This typically indicates that the group path is
    incomplete and may cause conversion issues where list content is not properly nested.
    """

    def validate(self, doc: DocumentStructure) -> List[CVATValidationError]:
        errors: List[CVATValidationError] = []

        id_to_element = {el.id: el for el in doc.elements}

        list_group_elements: Dict[int, Set[int]] = {}

        for path_id, group_element_ids in doc.path_mappings.group.items():
            if len(group_element_ids) < 2:
                continue

            resolved_ids = {eid for eid in group_element_ids if eid in id_to_element}

            if not resolved_ids:
                continue

            if any(
                id_to_element[eid].label == DocItemLabel.LIST_ITEM
                for eid in resolved_ids
            ):
                list_group_elements[path_id] = resolved_ids

        if not list_group_elements:
            return errors

        element_to_groups: Dict[int, Set[int]] = {}
        for path_id, element_ids in list_group_elements.items():
            for element_id in element_ids:
                element_to_groups.setdefault(element_id, set()).add(path_id)

        group_positions: Dict[int, Dict[int, List[int]]] = {}
        ro_to_groups: Dict[int, Set[int]] = {}

        for ro_path_id, ro_element_ids in doc.path_mappings.reading_order.items():
            for index, element_id in enumerate(ro_element_ids):
                for group_path_id in element_to_groups.get(element_id, ()):
                    positions = group_positions.setdefault(
                        group_path_id, {}
                    ).setdefault(ro_path_id, [])
                    positions.append(index)
                    ro_to_groups.setdefault(ro_path_id, set()).add(group_path_id)

        def _ranges_overlap(positions_a: List[int], positions_b: List[int]) -> bool:
            if not positions_a or not positions_b:
                return False
            return min(positions_a) <= max(positions_b) and min(positions_b) <= max(
                positions_a
            )

        adjacency: Dict[int, Set[int]] = {
            path_id: set() for path_id in list_group_elements.keys()
        }

        for ro_path_id, groups_in_ro in ro_to_groups.items():
            group_list = list(groups_in_ro)
            for idx_a in range(len(group_list)):
                path_a = group_list[idx_a]
                positions_a = group_positions.get(path_a, {}).get(ro_path_id, [])

                for idx_b in range(idx_a + 1, len(group_list)):
                    path_b = group_list[idx_b]
                    positions_b = group_positions.get(path_b, {}).get(ro_path_id, [])

                    if _ranges_overlap(positions_a, positions_b):
                        adjacency[path_a].add(path_b)
                        adjacency[path_b].add(path_a)

        unresolved_paths = set(list_group_elements.keys())
        clusters: List[Tuple[Set[int], Set[int]]] = []

        while unresolved_paths:
            start_path = unresolved_paths.pop()
            stack: List[int] = [start_path]
            cluster_paths: Set[int] = set()
            cluster_elements: Set[int] = set()

            while stack:
                current_path = stack.pop()
                if current_path in cluster_paths:
                    continue

                cluster_paths.add(current_path)
                cluster_elements.update(list_group_elements[current_path])

                for neighbor in adjacency[current_path]:
                    if neighbor in unresolved_paths:
                        stack.append(neighbor)
                        unresolved_paths.remove(neighbor)

            if not cluster_paths:
                continue

            clusters.append((cluster_paths, cluster_elements))

        for cluster_paths, cluster_elements in clusters:
            if len(cluster_elements) < 2:
                continue

            sorted_cluster_paths = sorted(cluster_paths)
            cluster_content_layers: Set[ContentLayer] = {
                id_to_element[element_id].content_layer
                for element_id in cluster_elements
                if element_id in id_to_element
            }

            for ro_path_id, ro_element_ids in doc.path_mappings.reading_order.items():
                cluster_positions: List[Tuple[int, int]] = [
                    (index, element_id)
                    for index, element_id in enumerate(ro_element_ids)
                    if element_id in cluster_elements
                ]

                if len(cluster_positions) < 2:
                    continue

                cluster_positions.sort(key=lambda item: item[0])

                for idx in range(len(cluster_positions) - 1):
                    start_pos, start_id = cluster_positions[idx]
                    end_pos, end_id = cluster_positions[idx + 1]

                    sandwiched_elements = [
                        ro_element_ids[position]
                        for position in range(start_pos + 1, end_pos)
                        if ro_element_ids[position] not in cluster_elements
                        # Ignore elements that belong to a different content layer (e.g. furniture)
                        and (
                            ro_element_ids[position] not in id_to_element
                            or id_to_element[ro_element_ids[position]].content_layer
                            in cluster_content_layers
                        )
                    ]

                    if not sandwiched_elements:
                        continue

                    sandwiched_labels = [
                        (
                            id_to_element[element_id].label.value
                            if element_id in id_to_element
                            else "unknown"
                        )
                        for element_id in sandwiched_elements
                    ]

                    errors.append(
                        CVATValidationError(
                            error_type="list_group_reading_order_impurity",
                            message=(
                                f"Group cluster {sorted_cluster_paths} (list groups): Found "
                                f"{len(sandwiched_elements)} non-grouped element(s) in reading order "
                                f"path {ro_path_id} between grouped list elements {start_id} and {end_id}. "
                                f"Sandwiched elements: {sandwiched_elements} (labels: {sandwiched_labels}). "
                                "These elements may not be properly nested in the converted document structure."
                            ),
                            severity=ValidationSeverity.WARNING,
                            path_id=sorted_cluster_paths[0],
                            path_ids=sorted_cluster_paths,
                        )
                    )

        return errors


class Validator:
    """Main validator class that runs all validation rules."""

    def __init__(self, rules: Optional[List[Type[ValidationRule]]] = None):
        """Initialize with optional list of validation rules."""
        self.rules = rules or [
            # FATAL
            ReadingOrderRule,
            # ERROR
            ElementTouchedByReadingOrderRule,
            MergeGroupPathsRule,
            MergePathDirectionRule,
            CaptionFootnotePathsRule,
            TableStructLocationRule,
            GraphCellLocationRule,
            ToValuePathStructureRule,
            GraphCellConnectionRule,
            GroupConsecutiveReadingOrderRule,
            # WARNING
            ValidLabelsRule,
            # SecondLevelReadingOrderParentRule,  # Temporarily excluded
            # ControlPointsHitElementsRule,  # Temporarily excluded
            MissingAttributesRule,
            UnrecognizedAttributesRule,
        ]

    def validate_sample(
        self,
        sample_name: str,
        doc: DocumentStructure,
    ) -> CVATValidationReport:
        """Validate a single sample and return a validation report."""
        errors: List[CVATValidationError] = []
        for rule_class in self.rules:
            rule = rule_class()
            errors.extend(rule.validate(doc))

        return CVATValidationReport(sample_name=sample_name, errors=errors)


@dataclass
class ValidatedSample:
    """DocumentStructure paired with its validation report."""

    sample_name: str
    structure: DocumentStructure
    report: CVATValidationReport


def validate_cvat_sample(
    xml_path: Path,
    image_filename: str,
    *,
    validator: Optional[Validator] = None,
    parsed_file: Optional[ParsedCVATFile] = None,
    proximity_thresh: float = DEFAULT_PROXIMITY_THRESHOLD,
) -> ValidatedSample:
    """Load a CVAT sample, build its structure, and validate it."""

    active_parsed = (
        parsed_file if parsed_file is not None else parse_cvat_file(xml_path)
    )
    parsed_image = active_parsed.get_image(image_filename)
    structure = DocumentStructure.from_parsed_image(
        parsed_image, proximity_thresh=proximity_thresh
    )
    active_validator = validator if validator is not None else Validator()
    report = active_validator.validate_sample(image_filename, structure)
    return ValidatedSample(
        sample_name=image_filename, structure=structure, report=report
    )


def validate_cvat_document(
    cvat_document: CVATDocument,
    *,
    validator: Optional[Validator] = None,
    proximity_thresh: float = DEFAULT_PROXIMITY_THRESHOLD,
) -> Dict[str, ValidatedSample]:
    """Validate each page contained in ``cvat_document``."""

    active_validator = validator if validator is not None else Validator()
    results: Dict[str, ValidatedSample] = {}
    parsed_cache: Dict[Path, ParsedCVATFile] = {}

    for page_info in sorted(cvat_document.pages, key=lambda page: page.page_number):
        parsed_file = parsed_cache.get(page_info.xml_path)
        if parsed_file is None:
            parsed_file = parse_cvat_file(page_info.xml_path)
            parsed_cache[page_info.xml_path] = parsed_file
        sample = validate_cvat_sample(
            page_info.xml_path,
            page_info.image_filename,
            validator=active_validator,
            parsed_file=parsed_file,
            proximity_thresh=proximity_thresh,
        )
        results[page_info.image_filename] = sample

    return results
