from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from .document import DocumentStructure
from .models import CVATValidationError, CVATValidationReport
from .path_mappings import (
    validate_caption_footnote_paths,
    validate_group_paths,
    validate_merge_paths,
)


@dataclass
class ValidationContext:
    """Context object passed to validation rules."""

    doc: DocumentStructure


class ValidationRule(ABC):
    """Base class for validation rules."""

    @abstractmethod
    def validate(self, context: ValidationContext) -> List[CVATValidationError]:
        """Validate the context and return a list of errors."""
        pass


class ValidLabelsRule(ValidationRule):
    """Validate that all element labels are valid DocItemLabels."""

    def validate(self, context: ValidationContext) -> List[CVATValidationError]:
        errors = []
        for el in context.doc.elements:
            try:
                _ = el.label  # This will raise ValueError if invalid
            except ValueError:
                errors.append(
                    CVATValidationError(
                        error_type="invalid_label",
                        message=f"Element {el.id} has invalid label '{el.label}'",
                        element_id=el.id,
                    )
                )
        return errors


class FirstLevelReadingOrderRule(ValidationRule):
    """Validate that there is a first-level reading order path."""

    def validate(self, context: ValidationContext) -> List[CVATValidationError]:
        found = any(
            p.label.startswith("reading_order") and (p.level == 1 or p.level is None)
            for p in context.doc.paths
        )
        if not found:
            return [
                CVATValidationError(
                    error_type="missing_first_level_reading_order",
                    message="No first-level reading-order path found.",
                )
            ]
        return []


class SingleFirstLevelReadingOrderRule(ValidationRule):
    """Validate that there is exactly one first-level reading order path."""

    def validate(self, context: ValidationContext) -> List[CVATValidationError]:
        level1_paths = [
            p
            for p in context.doc.paths
            if p.label.startswith("reading_order") and (p.level == 1 or p.level is None)
        ]
        if len(level1_paths) > 1:
            return [
                CVATValidationError(
                    error_type="multiple_first_level_reading_order",
                    message=f"Found {len(level1_paths)} first-level reading-order paths. Only one is allowed.",
                    path_ids=[p.id for p in level1_paths],
                )
            ]
        return []


class SecondLevelReadingOrderParentRule(ValidationRule):
    """Validate that second-level reading order paths have parent containers."""

    def validate(self, context: ValidationContext) -> List[CVATValidationError]:
        errors = []
        for p in context.doc.paths:
            if p.label.startswith("reading_order") and p.level and p.level > 1:
                container = context.doc.path_to_container.get(p.id)
                if container is None or container.parent is None:
                    errors.append(
                        CVATValidationError(
                            error_type="second_level_reading_order_no_parent",
                            message=f"Second-level reading-order path {p.id} has no parent container.",
                            path_id=p.id,
                        )
                    )
        return errors


class ElementTouchedByReadingOrderRule(ValidationRule):
    """Validate that every non-background element is touched by a reading order path."""

    def validate(self, context: ValidationContext) -> List[CVATValidationError]:
        errors = []
        touched = set()
        for elist in context.doc.path_mappings.reading_order.values():
            touched.update(elist)

        for el in context.doc.elements:
            if el.content_layer.upper() == "BACKGROUND":
                continue

            node = context.doc.get_node_by_element_id(el.id)
            if not node:
                continue

            descendant_ids = node.get_descendant_ids()
            if not (descendant_ids & touched):
                errors.append(
                    CVATValidationError(
                        error_type="element_not_touched_by_reading_order",
                        message=f"Element {el.id} ({el.label}) not touched by any reading-order path.",
                        element_id=el.id,
                    )
                )
        return errors


class MergePathsRule(ValidationRule):
    """Validate merge paths."""

    def validate(self, context: ValidationContext) -> List[CVATValidationError]:
        errors = []
        for error_msg in validate_merge_paths(
            context.doc.elements, context.doc.path_mappings.merge
        ):
            errors.append(
                CVATValidationError(
                    error_type="merge_path_error",
                    message=error_msg,
                )
            )
        return errors


class GroupPathsRule(ValidationRule):
    """Validate group paths."""

    def validate(self, context: ValidationContext) -> List[CVATValidationError]:
        errors = []
        for error_msg in validate_group_paths(
            context.doc.elements, context.doc.path_mappings.group
        ):
            errors.append(
                CVATValidationError(
                    error_type="group_path_error",
                    message=error_msg,
                )
            )
        return errors


class CaptionFootnotePathsRule(ValidationRule):
    """Validate caption and footnote paths."""

    def validate(self, context: ValidationContext) -> List[CVATValidationError]:
        errors = []
        for error_msg in validate_caption_footnote_paths(
            context.doc.elements,
            context.doc.path_mappings.to_caption,
            context.doc.path_mappings.to_footnote,
        ):
            errors.append(
                CVATValidationError(
                    error_type="caption_footnote_path_error",
                    message=error_msg,
                )
            )
        return errors


class Validator:
    """Main validator class that runs all validation rules."""

    def __init__(self, rules: Optional[List[Type[ValidationRule]]] = None):
        """Initialize with optional list of validation rules."""
        self.rules = rules or [
            ValidLabelsRule,
            FirstLevelReadingOrderRule,
            SingleFirstLevelReadingOrderRule,
            SecondLevelReadingOrderParentRule,
            ElementTouchedByReadingOrderRule,
            MergePathsRule,
            GroupPathsRule,
            CaptionFootnotePathsRule,
        ]

    def validate_sample(
        self,
        sample_name: str,
        doc: DocumentStructure,
    ) -> CVATValidationReport:
        """Validate a single sample and return a validation report."""
        context = ValidationContext(doc=doc)

        errors = []
        for rule_class in self.rules:
            rule = rule_class()
            errors.extend(rule.validate(context))

        return CVATValidationReport(sample_name=sample_name, errors=errors)
