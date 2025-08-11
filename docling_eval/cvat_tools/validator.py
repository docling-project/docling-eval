import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Type

from docling_core.types.doc.document import ContentLayer

from .document import DocumentStructure
from .models import (
    CVATElement,
    CVATValidationError,
    CVATValidationReport,
    ValidationSeverity,
)
from .path_mappings import validate_caption_footnote_paths
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
    """Validate that every non-background element is touched by a reading order path - ERROR level."""

    def _is_element_inside_table(
        self, element: CVATElement, doc: DocumentStructure
    ) -> bool:
        """Check if an element is inside a table container."""
        node = doc.get_node_by_element_id(element.id)
        if not node:
            return False

        # Walk up the containment tree to check if any ancestor is a table
        current = node.parent
        while current:
            if current.element.label == "table":
                return True
            current = current.parent

        return False

    def validate(self, doc: DocumentStructure) -> List[CVATValidationError]:
        errors: list[CVATValidationError] = []
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
        for path_id, element_ids in doc.path_mappings.reading_order.items():
            path = doc.get_path_by_id(path_id)
            if (
                path
                and path.label.startswith("reading_order")
                and (path.level == 1 or path.level is None)
            ):
                level1_touched.update(element_ids)

        logger.debug(f"Level 1 touched elements: {len(level1_touched)}")
        logger.debug(f"Level 1 touched element IDs: {sorted(level1_touched)}")

        # Pre-compute all descendant IDs for efficiency (avoid repeated tree traversal)
        element_descendants = {}
        for el in doc.elements:
            node = doc.get_node_by_element_id(el.id)
            if node:
                element_descendants[el.id] = node.get_descendant_ids()

        # Collect all elements that would fail the reading order validation
        untouched_elements = []
        for el in doc.elements:
            if el.content_layer == ContentLayer.BACKGROUND:
                continue

            # Skip validation for elements inside table containers
            if self._is_element_inside_table(el, doc):
                continue

            # Skip validation for picture elements that have descendants touched by level 1 reading order
            if (
                el.label == "picture"
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

        # Apply special condition for CHART/INFOGRAPHIC picture elements
        if untouched_elements:
            # Find CHART/INFOGRAPHIC pictures that have NO touched elements (completely untouched)
            chart_infographic_untouched = set()
            for el in doc.elements:
                if (
                    el.label == "picture"
                    and el.type
                    and el.type.upper() in ["CHART", "INFOGRAPHIC"]
                ):
                    logger.debug(
                        f"Found CHART/INFOGRAPHIC picture element {el.id} with type '{el.type}'"
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
                        chart_infographic_untouched.add(el.id)
                        logger.debug(
                            f"Added picture {el.id} to chart_infographic_untouched"
                        )
                    else:
                        logger.debug(
                            f"Picture {el.id} has touched descendants, NOT added to chart_infographic_untouched"
                        )

            logger.debug(
                f"chart_infographic_untouched: {sorted(chart_infographic_untouched)}"
            )

            # Report errors for elements not in completely untouched CHART/INFOGRAPHIC pictures
            for el in untouched_elements:
                # Check if element is in a completely untouched CHART/INFOGRAPHIC picture
                in_untouched_chart_infographic = False
                for picture_el in doc.elements:
                    if (
                        picture_el.label == "picture"
                        and picture_el.type
                        and picture_el.type.upper() in ["CHART", "INFOGRAPHIC"]
                        and picture_el.id in chart_infographic_untouched
                        and el.id in element_descendants.get(picture_el.id, set())
                    ):
                        in_untouched_chart_infographic = True
                        logger.debug(
                            f"Element {el.id} is inside untouched CHART/INFOGRAPHIC picture {picture_el.id}"
                        )
                        break

                if not in_untouched_chart_infographic:
                    logger.debug(f"Adding error for element {el.id} ({el.label})")
                    errors.append(
                        CVATValidationError(
                            error_type="element_not_touched_by_reading_order",
                            message=f"Element {el.id} ({el.label}) not touched by any reading-order path.",
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

            # Allow only the special case for checkboxes
            if labels == {"checkbox_selected", "checkbox_unselected"}:
                pass  # legal
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
            first_el = elements_in_group[0]
            for el in elements_in_group[1:]:
                if el.content_layer != first_el.content_layer:
                    errors.append(
                        CVATValidationError(
                            error_type=f"{path_type}_path_different_content_layers",
                            message=f"{path_type.capitalize()} path {path_id}: Elements have different content layers: {first_el.content_layer} vs {el.content_layer}",
                            severity=ValidationSeverity.ERROR,
                            path_id=path_id,
                        )
                    )

        return errors


class CaptionFootnotePathsRule(ValidationRule):
    """Validate caption and footnote paths - ERROR level."""

    def validate(self, doc: DocumentStructure) -> List[CVATValidationError]:
        errors: list[CVATValidationError] = []
        for error_msg in validate_caption_footnote_paths(
            doc.elements,
            doc.path_mappings.to_caption,
            doc.path_mappings.to_footnote,
        ):
            errors.append(
                CVATValidationError(
                    error_type="caption_footnote_path_error",
                    message=error_msg,
                    severity=ValidationSeverity.ERROR,
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
            if el.label in ["section_header", "list_item"]:
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
    """Validate that group paths connect consecutive elements in reading order."""

    def validate(self, doc: DocumentStructure) -> List[CVATValidationError]:
        errors: List[CVATValidationError] = []

        # Build global reading order - this needs proper implementation
        # For now, skipping this validation
        # global_order = build_global_reading_order(...)

        for path_id, element_ids in doc.path_mappings.group.items():
            if len(element_ids) < 2:
                continue

            # TODO: Implement proper consecutive reading order validation
            # This requires building the global reading order first
            pass

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
            CaptionFootnotePathsRule,
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
