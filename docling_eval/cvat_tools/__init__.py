"""
CVAT Annotation Validation Tool

- Parses CVAT XML annotation files and corresponding images
- Builds containment trees and parses reading-order/group/merge paths
- Validates annotation structure according to project rules
- Outputs element tree and validation report per sample
- Provides analysis tools for visualizing annotations and reading order
- Converts CVAT annotations to DoclingDocuments in batch

Usage:
    from docling_eval.cvat_tools import Validator, DocumentStructure
    
    # Create document structure from CVAT XML
    doc = DocumentStructure.from_cvat_xml(xml_path, image_filename)
    
    # Validate the structure
    validator = Validator()
    report = validator.validate_sample("sample_name", doc)
"""

from .analysis import print_containment_tree, print_elements_and_paths
from .document import DocumentStructure
from .models import (
    CVATAnnotationPath,
    CVATElement,
    CVATImageInfo,
    CVATValidationError,
    CVATValidationReport,
    CVATValidationRunReport,
    ValidationSeverity,
)
from .parser import find_samples_in_directory, parse_cvat_xml_for_image
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
    ValidationRule,
    Validator,
    ValidLabelsRule,
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
    "parse_cvat_xml_for_image",
    "find_samples_in_directory",
    # Tree
    "TreeNode",
    "build_containment_tree",
    "build_global_reading_order",
    "apply_reading_order_to_tree",
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
