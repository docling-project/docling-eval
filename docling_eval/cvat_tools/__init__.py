"""
CVAT Annotation Validation Tool

- Parses CVAT XML annotation files and corresponding images
- Builds containment trees and parses reading-order/group/merge paths
- Validates annotation structure according to project rules
- Outputs element tree and validation report per sample

Usage:
    python -m docling_eval.cvat_tools.cli <input_root_dir>
"""

from .models import (
    CVATAnnotationPath,
    Element,
    ImageInfo,
    ValidationError,
    ValidationReport,
    ValidationRunReport,
)
from .parser import find_samples_in_directory, parse_cvat_xml_for_image
from .tree import (
    TreeNode,
    associate_reading_order_paths_to_containers,
    build_containment_tree,
    map_path_points_to_elements,
)
from .validator import (
    ElementTouchedByReadingOrderRule,
    FirstLevelReadingOrderRule,
    SecondLevelReadingOrderParentRule,
    ValidationContext,
    ValidationRule,
    Validator,
    ValidLabelsRule,
)

__all__ = [
    # Models
    "Element",
    "CVATAnnotationPath",
    "ValidationError",
    "ValidationReport",
    "ValidationRunReport",
    "ImageInfo",
    # Parser
    "parse_cvat_xml_for_image",
    "find_samples_in_directory",
    # Tree
    "TreeNode",
    "build_containment_tree",
    "map_path_points_to_elements",
    "associate_reading_order_paths_to_containers",
    # Validator
    "Validator",
    "ValidationRule",
    "ValidationContext",
    "ValidLabelsRule",
    "FirstLevelReadingOrderRule",
    "SecondLevelReadingOrderParentRule",
    "ElementTouchedByReadingOrderRule",
]
