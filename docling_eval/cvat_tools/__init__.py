"""
CVAT Annotation Validation Tool

- Parses CVAT XML annotation files and corresponding images
- Builds containment trees and parses reading-order/group/merge paths
- Validates annotation structure according to project rules
- Outputs element tree and validation report per sample
- Provides analysis tools for visualizing annotations and reading order

Usage:
    python -m docling_eval.cvat_tools.cli <input_root_dir>
"""

from .analysis import (
    apply_reading_order_to_tree,
    print_containment_tree,
    print_elements_and_paths,
)
from .models import (
    CVATAnnotationPath,
    CVATElement,
    CVATImageInfo,
    CVATValidationError,
    CVATValidationReport,
    CVATValidationRunReport,
)
from .parser import find_samples_in_directory, parse_cvat_xml_for_image
from .path_mappings import (
    PathMappings,
    associate_paths_to_containers,
    map_path_points_to_elements,
    validate_caption_footnote_paths,
    validate_group_paths,
    validate_merge_paths,
)
from .tree import TreeNode, build_containment_tree, build_global_reading_order
from .validator import (
    CaptionFootnotePathsRule,
    ElementTouchedByReadingOrderRule,
    FirstLevelReadingOrderRule,
    GroupPathsRule,
    MergePathsRule,
    SecondLevelReadingOrderParentRule,
    ValidationContext,
    ValidationRule,
    Validator,
    ValidLabelsRule,
)

__all__ = [
    # Models
    "CVATElement",
    "CVATAnnotationPath",
    "CVATValidationError",
    "CVATValidationReport",
    "CVATValidationRunReport",
    "CVATImageInfo",
    # Parser
    "parse_cvat_xml_for_image",
    "find_samples_in_directory",
    # Tree
    "TreeNode",
    "build_containment_tree",
    "build_global_reading_order",
    # Path Mappings
    "PathMappings",
    "map_path_points_to_elements",
    "associate_paths_to_containers",
    "validate_merge_paths",
    "validate_group_paths",
    "validate_caption_footnote_paths",
    # Validator
    "Validator",
    "ValidationRule",
    "ValidationContext",
    "ValidLabelsRule",
    "FirstLevelReadingOrderRule",
    "SecondLevelReadingOrderParentRule",
    "ElementTouchedByReadingOrderRule",
    "MergePathsRule",
    "GroupPathsRule",
    "CaptionFootnotePathsRule",
    # Analysis
    "print_elements_and_paths",
    "print_containment_tree",
    "apply_reading_order_to_tree",
]
