"""Test module for CVAT to DoclingDocument conversion."""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import pytest
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import DoclingDocument
from dotenv import load_dotenv
from PIL import Image as PILImage

from docling_eval.cvat_tools.analysis import (
    print_containment_tree,
    print_elements_and_paths,
)
from docling_eval.cvat_tools.cvat_to_docling import convert_cvat_to_docling
from docling_eval.cvat_tools.document import DocumentStructure
from docling_eval.cvat_tools.models import CVATValidationReport, ValidationSeverity
from docling_eval.cvat_tools.tree import (
    apply_reading_order_to_tree,
    build_global_reading_order,
)
from docling_eval.cvat_tools.validator import Validator

IS_CI = bool(os.getenv("CI"))
load_dotenv()


def _find_case_directories(root_dir: Path) -> List[Path]:
    """Find all case directories in the root directory.

    Args:
        root_dir: Root directory to search for case directories

    Returns:
        List of case directory paths
    """
    return sorted(
        [d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("case")]
    )


def _get_sample_paths_for_case(case_dir: Path) -> List[Path]:
    """Get all image file paths for a given case directory.

    Args:
        case_dir: Case directory to search for image files

    Returns:
        List of image file paths
    """
    sample_paths = []

    # Find all image files in case directory
    for ext in [
        "*.pdf",
        "*.PDF",
        "*.png",
        "*.PNG",
        "*.jpg",
        "*.JPG",
        "*.jpeg",
        "*.JPEG",
        "*.tif",
        "*.TIF",
        "*.tiff",
        "*.TIFF",
        "*.bmp",
        "*.BMP",
    ]:
        sample_paths.extend(sorted(case_dir.glob(ext)))

    return sorted(sample_paths)


def _test_conversion_with_sample_data(
    xml_path: Path,
    image_path: Path,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[CVATValidationReport, Optional[DoclingDocument]]:
    """Test the conversion with sample data.

    Args:
        xml_path: Path to CVAT XML file
        image_path: Path to document image
        output_dir: Optional output directory for saving results
        verbose: Whether to print detailed information
    """
    if output_dir is None:
        output_dir = image_path.parent

    print(f"Testing conversion for: {image_path.name}")
    print("=" * 60)

    # Create DocumentStructure first for validation
    doc_structure = DocumentStructure.from_cvat_xml(xml_path, image_path.name)
    print(f"✓ Created DocumentStructure:")
    print(f"  - Elements: {len(doc_structure.elements)}")
    print(f"  - Paths: {len(doc_structure.paths)}")
    print(f"  - Tree roots: {len(doc_structure.tree_roots)}")

    # Validate the document structure before conversion
    validator = Validator()
    validation_report = validator.validate_sample(image_path.name, doc_structure)

    print(f"\n--- Validation Report ---")
    print(f"Total validation errors: {len(validation_report.errors)}")

    # Count errors by severity
    fatal_errors = [
        e for e in validation_report.errors if e.severity == ValidationSeverity.FATAL
    ]
    error_errors = [
        e for e in validation_report.errors if e.severity == ValidationSeverity.ERROR
    ]
    warning_errors = [
        e for e in validation_report.errors if e.severity == ValidationSeverity.WARNING
    ]

    print(f"  - FATAL: {len(fatal_errors)}")
    print(f"  - ERROR: {len(error_errors)}")
    print(f"  - WARNING: {len(warning_errors)}")

    # Print validation errors if any
    if validation_report.errors:
        print("\nValidation Issues:")
        for error in validation_report.errors:
            print(f"  {error.severity.value}: {error.message}")

    # Check for fatal errors - do not proceed with conversion if found
    if fatal_errors:
        print(
            f"\n✗ Cannot proceed with conversion due to {len(fatal_errors)} FATAL validation error(s)"
        )
        return validation_report, None

    print(f"\n✓ Validation passed - proceeding with conversion")

    if verbose:
        print("\n--- Elements and Paths ---")
        print_elements_and_paths(
            doc_structure.elements, doc_structure.paths, doc_structure.image_info
        )

        print("\n--- Original Containment Tree ---")
        print_containment_tree(doc_structure.tree_roots, doc_structure.image_info)

        global_ro = build_global_reading_order(
            doc_structure.paths,
            doc_structure.path_mappings.reading_order,
            doc_structure.path_to_container,
            doc_structure.tree_roots,
        )
        print("\n--- Ordered Containment Tree ---")
        # Apply reading order to tree before printing
        apply_reading_order_to_tree(doc_structure.tree_roots, global_ro)
        print_containment_tree(doc_structure.tree_roots, doc_structure.image_info)

    # Convert to DoclingDocument (only if validation passed)
    doc = convert_cvat_to_docling(xml_path, image_path)

    # Only proceed with document processing if conversion was successful
    if doc is None:
        print(f"\n✗ Conversion failed for {image_path.name}")
        return validation_report, None

    print(f"\n✓ Converted to DoclingDocument: {image_path.name}")
    print(f"  - Pages: {len(doc.pages)}")
    print(f"  - Groups: {len(doc.groups)}")
    print(f"  - Texts: {len(doc.texts)}")
    print(f"  - Pictures: {len(doc.pictures)}")
    print(f"  - Tables: {len(doc.tables)}")

    # Print element tree
    if verbose:
        print("\n--- DoclingDocument Element Tree ---")
        doc.print_element_tree()

    # Save outputs
    json_output = output_dir / f"{image_path.stem}_docling.json"
    html_output = output_dir / f"{image_path.stem}_docling.html"
    md_output = output_dir / f"{image_path.stem}_docling.md"

    doc.save_as_json(json_output)
    doc.save_as_html(
        html_output, image_mode=ImageRefMode.EMBEDDED, split_page_view=True
    )
    doc.save_as_markdown(md_output, image_mode=ImageRefMode.EMBEDDED)

    viz_imgs = doc.get_visualization()
    for page_no, img in viz_imgs.items():
        if page_no is not None:
            img.save(output_dir / f"{image_path.stem}_docling_p{page_no}.png")

    print(f"\n✓ Saved outputs:")
    print(f"  - JSON: {json_output.name}")
    print(f"  - HTML: {html_output.name}")
    print(f"  - Markdown: {md_output.name}")

    return validation_report, doc


@pytest.mark.skipif(IS_CI, reason="Skipping test in CI because the test is too heavy.")
def test_cvat_to_docling_conversion():
    """Test CVAT to DoclingDocument conversion for all available cases."""
    # Find all case directories
    root_dir = Path("tests/data/cvat_pdfs_dataset_e2e")
    case_dirs = _find_case_directories(root_dir)
    output_dir = Path("scratch/cvat_to_docling_converter")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each case directory
    for case_dir in case_dirs:
        # Extract case number and basename from directory name
        case_basename = case_dir.name
        annotations_xml = case_dir.parent / f"{case_basename}_annotations.xml"

        # Find all image files in case directory
        sample_paths = _get_sample_paths_for_case(case_dir)

        for image_path in sample_paths:
            print(f"\nProcessing {image_path.name}...")

            if annotations_xml.exists() and image_path.exists():
                validation_report, result = _test_conversion_with_sample_data(
                    annotations_xml, image_path, output_dir=output_dir, verbose=True
                )
                if validation_report.has_errors():
                    print(f"✗ Validation errors: {validation_report.errors}")
                if result is None:
                    print(f"✗ Conversion failed for {image_path.name}")
                else:
                    print(f"✓ Conversion successful for {image_path.name}")
            else:
                print(f"⚠ Missing files for {image_path.name}:")
                if not annotations_xml.exists():
                    print(f"  - Missing annotations.xml at {annotations_xml}")
                if not image_path.exists():
                    print(f"  - Missing image at {image_path}")
                # Skip this test case if files are missing
                continue


@pytest.mark.skipif(IS_CI, reason="Skipping test in CI because the test is too heavy.")
def test_case_02_specific_sample():
    """Test CVAT to DoclingDocument conversion for a specific sample in case_02."""
    # Define paths
    root_dir = Path("tests/data/cvat_pdfs_dataset_e2e")
    pdf_path = (
        root_dir
        / "case_02"
        / "6b18af59b633f89b96a64aa435e0f7616eb1813d884c4c3da5e4cea9a8f9316b.pdf"
    )
    xml_path = root_dir / "case_02_annotations.xml"
    output_dir = Path("scratch/cvat_to_docling_converter/case_02")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify files exist
    assert pdf_path.exists(), f"PDF file not found: {pdf_path}"
    assert xml_path.exists(), f"XML file not found: {xml_path}"

    # Run conversion test
    validation_report, result = _test_conversion_with_sample_data(
        xml_path=xml_path, image_path=pdf_path, output_dir=output_dir, verbose=True
    )

    # Assertions
    assert (
        not validation_report.has_errors()
    ), f"Validation errors found: {validation_report.errors}"
    assert result is not None, "Conversion failed - result is None"
    assert len(result.pages) > 0, "No pages found in converted document"
    assert len(result.texts) > 0, "No text elements found in converted document"

    # Verify output files were created
    expected_files = [
        output_dir / f"{pdf_path.stem}_docling.json",
        output_dir / f"{pdf_path.stem}_docling.html",
        output_dir / f"{pdf_path.stem}_docling.md",
    ]
    for file_path in expected_files:
        assert file_path.exists(), f"Expected output file not found: {file_path}"
