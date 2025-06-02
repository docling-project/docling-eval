from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from .models import CVATAnnotationPath, Element, ValidationError, ValidationReport
from .tree import TreeNode


@dataclass
class ValidationContext:
    """Context object passed to validation rules."""

    elements: List[Element]
    paths: List[CVATAnnotationPath]
    tree_roots: List[TreeNode]
    path_to_elements: Dict[int, List[int]]
    path_to_container: Dict[int, TreeNode]


class ValidationRule(ABC):
    """Base class for validation rules."""

    @abstractmethod
    def validate(self, context: ValidationContext) -> List[ValidationError]:
        """Validate the context and return a list of errors."""
        pass


class ValidLabelsRule(ValidationRule):
    """Validate that all element labels are valid DocItemLabels."""

    def validate(self, context: ValidationContext) -> List[ValidationError]:
        errors = []
        for el in context.elements:
            try:
                _ = el.label  # This will raise ValueError if invalid
            except ValueError:
                errors.append(
                    ValidationError(
                        error_type="invalid_label",
                        message=f"Element {el.id} has invalid label '{el.label}'",
                        element_id=el.id,
                    )
                )
        return errors


class FirstLevelReadingOrderRule(ValidationRule):
    """Validate that there is a first-level reading order path."""

    def validate(self, context: ValidationContext) -> List[ValidationError]:
        found = any(
            p.label.startswith("reading_order") and (p.level == 1 or p.level is None)
            for p in context.paths
        )
        if not found:
            return [
                ValidationError(
                    error_type="missing_first_level_reading_order",
                    message="No first-level reading-order path found.",
                )
            ]
        return []


class SecondLevelReadingOrderParentRule(ValidationRule):
    """Validate that second-level reading order paths have parent containers."""

    def validate(self, context: ValidationContext) -> List[ValidationError]:
        errors = []
        for p in context.paths:
            if p.label.startswith("reading_order") and p.level and p.level > 1:
                container = context.path_to_container.get(p.id)
                if container is None or container.parent is None:
                    errors.append(
                        ValidationError(
                            error_type="second_level_reading_order_no_parent",
                            message=f"Second-level reading-order path {p.id} has no parent container.",
                            path_id=p.id,
                        )
                    )
        return errors


class ElementTouchedByReadingOrderRule(ValidationRule):
    """Validate that every non-background element is touched by a reading order path."""

    def validate(self, context: ValidationContext) -> List[ValidationError]:
        errors = []
        touched = set()
        for elist in context.path_to_elements.values():
            touched.update(elist)

        # Build id->node mapping
        id_to_node = {}
        stack = list(context.tree_roots)
        while stack:
            node = stack.pop()
            id_to_node[node.element.id] = node
            stack.extend(node.children)

        for el in context.elements:
            if el.content_layer.upper() == "BACKGROUND":
                continue

            node = id_to_node.get(el.id)
            if not node:
                continue

            descendant_ids = node.get_descendant_ids()
            if not (descendant_ids & touched):
                errors.append(
                    ValidationError(
                        error_type="element_not_touched_by_reading_order",
                        message=f"Element {el.id} ({el.label}) not touched by any reading-order path.",
                        element_id=el.id,
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
            SecondLevelReadingOrderParentRule,
            ElementTouchedByReadingOrderRule,
        ]

    def validate_sample(
        self,
        sample_name: str,
        elements: List[Element],
        paths: List[CVATAnnotationPath],
        tree_roots: List[TreeNode],
        path_to_elements: Dict[int, List[int]],
        path_to_container: Dict[int, TreeNode],
    ) -> ValidationReport:
        """Validate a single sample and return a validation report."""
        context = ValidationContext(
            elements=elements,
            paths=paths,
            tree_roots=tree_roots,
            path_to_elements=path_to_elements,
            path_to_container=path_to_container,
        )

        errors = []
        for rule_class in self.rules:
            rule = rule_class()
            errors.extend(rule.validate(context))

        return ValidationReport(sample_name=sample_name, errors=errors)
