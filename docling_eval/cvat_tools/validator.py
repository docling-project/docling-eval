from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from .document import DocumentStructure
from .models import (
    CVATElement,
    CVATValidationError,
    CVATValidationReport,
    ValidationSeverity,
)
from .path_mappings import validate_caption_footnote_paths
from .utils import DEFAULT_PROXIMITY_THRESHOLD, find_elements_containing_point


class ValidationRule(ABC):
    """Base class for validation rules."""

    @abstractmethod
    def validate(self, doc: DocumentStructure) -> List[CVATValidationError]:
        """Validate the context and return a list of errors."""
        pass


class ValidLabelsRule(ValidationRule):
    """Validate that all element labels are valid DocItemLabels."""

    def validate(self, doc: DocumentStructure) -> List[CVATValidationError]:
        errors = []
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
        errors = []

        # Find all first-level reading order paths
        level1_paths = [
            p
            for p in doc.paths
            if p.label.startswith("reading_order") and (p.level == 1 or p.level is None)
        ]

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
        errors = []
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
        errors = []
        touched = set()
        for elist in doc.path_mappings.reading_order.values():
            touched.update(elist)

        for el in doc.elements:
            if el.content_layer.upper() == "BACKGROUND":
                continue

            # Skip validation for elements inside table containers
            if self._is_element_inside_table(el, doc):
                continue

            node = doc.get_node_by_element_id(el.id)
            if not node:
                continue

            descendant_ids = node.get_descendant_ids()
            if not (descendant_ids & touched):
                errors.append(
                    CVATValidationError(
                        error_type="element_not_touched_by_reading_order",
                        message=f"Element {el.id} ({el.label}) not touched by any reading-order path.",
                        severity=ValidationSeverity.ERROR,
                        element_id=el.id,
                    )
                )
        return errors


class MergeGroupPathsRule(ValidationRule):
    """Validate merge and group paths - ERROR level."""

    def validate(self, doc: DocumentStructure) -> List[CVATValidationError]:
        errors = []

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

        errors = []
        id_to_element = {el.id: el for el in elements}

        for path_id, el_ids in path_mappings.items():
            if len(el_ids) < 2:
                continue

            # Check that all elements exist
            first_el = id_to_element.get(el_ids[0])
            if not first_el:
                errors.append(
                    CVATValidationError(
                        error_type=f"{path_type}_path_missing_element",
                        message=f"{path_type.capitalize()} path {path_id}: Element {el_ids[0]} not found",
                        severity=ValidationSeverity.ERROR,
                        path_id=path_id,
                    )
                )
                continue

            # Check same label and content_layer
            for el_id in el_ids[1:]:
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
                    continue
                if el.label != first_el.label:
                    errors.append(
                        CVATValidationError(
                            error_type=f"{path_type}_path_different_labels",
                            message=f"{path_type.capitalize()} path {path_id}: Elements have different labels: {first_el.label} vs {el.label}",
                            severity=ValidationSeverity.ERROR,
                            path_id=path_id,
                        )
                    )
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
        errors = []
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
        errors = []

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
        errors = []

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
        errors = []

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
            SecondLevelReadingOrderParentRule,
            ControlPointsHitElementsRule,
            MissingAttributesRule,
            UnrecognizedAttributesRule,
        ]

    def validate_sample(
        self,
        sample_name: str,
        doc: DocumentStructure,
    ) -> CVATValidationReport:
        """Validate a single sample and return a validation report."""
        errors = []
        for rule_class in self.rules:
            rule = rule_class()
            errors.extend(rule.validate(doc))

        return CVATValidationReport(sample_name=sample_name, errors=errors)
