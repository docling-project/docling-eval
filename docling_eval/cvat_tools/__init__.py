"""
CVAT Annotation Validation Tool

- Parses CVAT XML annotation files and corresponding images
- Builds containment trees and parses reading-order/group/merge paths
- Validates annotation structure according to project rules
- Outputs element tree and validation report per sample
- Provides analysis tools for visualizing annotations and reading order
- Converts CVAT annotations to DoclingDocuments in batch

Usage:
    from docling_eval.cvat_tools import validate_cvat_sample

    # Load and validate a CVAT XML sample
    validated = validate_cvat_sample(xml_path, image_filename)

    # Inspect the validation report
    report = validated.report
"""

from .analysis import print_containment_tree, print_elements_and_paths
from .document import DocumentStructure
from .geometry import (
    bbox_contains,
    bbox_fraction_inside,
    bbox_intersection,
    bbox_iou,
    dedupe_items_by_bbox,
    iter_unique_by_bbox,
)
from .models import (
    CVATAnnotationPath,
    CVATElement,
    CVATImageInfo,
    CVATValidationError,
    CVATValidationReport,
    CVATValidationRunReport,
    ValidationSeverity,
)
from .parser import find_samples_in_directory
from .path_mappings import (
    PathMappings,
    associate_paths_to_containers,
    map_path_points_to_elements,
    validate_caption_footnote_paths,
)
from .tree import (
    TreeNode,
    apply_reading_order_to_tree,
    build_containment_tree,
    build_global_reading_order,
)
from .utils import (
    DEFAULT_PROXIMITY_THRESHOLD,
    find_elements_containing_point,
    get_deepest_element_at_point,
    is_caption_element,
    is_container_element,
    is_footnote_element,
    validate_element_types,
)
from .validator import (
    CaptionFootnotePathsRule,
    ControlPointsHitElementsRule,
    ElementTouchedByReadingOrderRule,
    GroupConsecutiveReadingOrderRule,
    MergeGroupPathsRule,
    MissingAttributesRule,
    ReadingOrderRule,
    SecondLevelReadingOrderParentRule,
    UnrecognizedAttributesRule,
    ValidatedSample,
    ValidationRule,
    Validator,
    ValidLabelsRule,
    validate_cvat_document,
    validate_cvat_sample,
)

__all__ = [
    # Document structure
    "DocumentStructure",
    # Models
    "CVATElement",
    "CVATAnnotationPath",
    "CVATValidationError",
    "CVATValidationReport",
    "CVATValidationRunReport",
    "CVATImageInfo",
    "ValidationSeverity",
    # Parser
    "find_samples_in_directory",
    # Tree
    "TreeNode",
    "build_containment_tree",
    "build_global_reading_order",
    "apply_reading_order_to_tree",
    # Geometry
    "bbox_contains",
    "bbox_fraction_inside",
    "bbox_intersection",
    "bbox_iou",
    "dedupe_items_by_bbox",
    "iter_unique_by_bbox",
    # Path Mappings
    "PathMappings",
    "map_path_points_to_elements",
    "associate_paths_to_containers",
    "validate_caption_footnote_paths",
    # Validator
    "Validator",
    "ValidationRule",
    "ValidLabelsRule",
    "ReadingOrderRule",
    "SecondLevelReadingOrderParentRule",
    "ElementTouchedByReadingOrderRule",
    "MergeGroupPathsRule",
    "CaptionFootnotePathsRule",
    "ControlPointsHitElementsRule",
    "MissingAttributesRule",
    "UnrecognizedAttributesRule",
    "GroupConsecutiveReadingOrderRule",
    "ValidatedSample",
    "validate_cvat_sample",
    "validate_cvat_document",
    # Analysis
    "print_elements_and_paths",
    "print_containment_tree",
    # Utils
    "DEFAULT_PROXIMITY_THRESHOLD",
    "find_elements_containing_point",
    "get_deepest_element_at_point",
    "is_caption_element",
    "is_container_element",
    "is_footnote_element",
    "validate_element_types",
]
