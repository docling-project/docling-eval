"""Test module for CVAT to DoclingDocument conversion."""

import os
from pathlib import Path
from typing import Optional, List

from docling_core.types.doc.base import ImageRefMode
from PIL import Image as PILImage

from docling_eval.cvat_tools.analysis import (
    apply_reading_order_to_tree,
    print_containment_tree,
    print_elements_and_paths,
)
from docling_eval.cvat_tools.cvat_to_docling import (
    CVATToDoclingConverter,
    convert_cvat_to_docling,
)
from docling_eval.cvat_tools.document import DocumentStructure
from docling_eval.cvat_tools.tree import build_global_reading_order


def test_conversion_with_sample_data(
    xml_path: Path,
    image_path: Path,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
):
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

    try:
        # Load image
        image = PILImage.open(image_path)
        print(f"✓ Loaded image: {image.size}")

        # Create DocumentStructure
        doc_structure = DocumentStructure.from_cvat_xml(xml_path, image_path.name)
        print(f"✓ Created DocumentStructure:")
        print(f"  - Elements: {len(doc_structure.elements)}")
        print(f"  - Paths: {len(doc_structure.paths)}")
        print(f"  - Tree roots: {len(doc_structure.tree_roots)}")

        if verbose:
            print("\n--- Elements and Paths ---")
            print_elements_and_paths(
                doc_structure.elements, doc_structure.paths, doc_structure.image_info
            )

            print("\n--- Original Containment Tree ---")
            if doc_structure.image_info is not None:
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
            if doc_structure.image_info is not None:
                print_containment_tree(doc_structure.tree_roots, doc_structure.image_info)

        # Create converter
        converter = CVATToDoclingConverter(
            doc_structure, image, ocr_framework="vision", image_filename=image_path.name
        )

        # Convert to DoclingDocument
        doc = converter.convert()
        print(f"\n✓ Converted to DoclingDocument: {doc.name}")
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

        print(f"\n✓ Saved outputs:")
        print(f"  - JSON: {json_output.name}")
        print(f"  - HTML: {html_output.name}")
        print(f"  - Markdown: {md_output.name}")

        return doc

    except Exception as e:
        print(f"\n✗ Error during conversion: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_batch_conversion(
    root_dir: Path, output_dir: Optional[Path] = None, verbose: bool = False
):
    """Test batch conversion of multiple documents.

    Args:
        root_dir: Root directory containing images and annotations.xml files
        output_dir: Optional output directory for results
        verbose: Whether to print detailed information
    """
    if output_dir is None:
        output_dir = root_dir / "docling_output"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all image files
    image_extensions = {".png", ".jpg", ".jpeg"}
    image_files: List[Path] = []
    for ext in image_extensions:
        image_files.extend(root_dir.rglob(f"*{ext}"))
        image_files.extend(root_dir.rglob(f"*{ext.upper()}"))

    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))

    processed_count = 0
    failed_count = 0

    print(f"Found {len(image_files)} image files in {root_dir}")
    print("=" * 60)

    for image_path in image_files:
        # Check if annotations.xml exists in the same directory
        xml_path = image_path.parent / "annotations.xml"

        if not xml_path.exists():
            print(f"⚠ No annotations.xml found for {image_path.name}, skipping")
            continue

        print(f"\nProcessing {image_path.relative_to(root_dir)}...")

        try:
            doc = convert_cvat_to_docling(xml_path, image_path, ocr_framework="vision")

            if doc:
                # Save outputs
                doc_output_dir = output_dir / image_path.parent.relative_to(root_dir)
                doc_output_dir.mkdir(parents=True, exist_ok=True)

                doc.save_as_json(doc_output_dir / f"{image_path.stem}.json")
                doc.save_as_html(
                    doc_output_dir / f"{image_path.stem}.html",
                    image_mode=ImageRefMode.EMBEDDED,
                    split_page_view=True,
                )

                with open(doc_output_dir / f"{image_path.stem}.txt", "w") as fp:
                    fp.write(doc.export_to_element_tree())

                if verbose:
                    print(f"  ✓ Converted successfully")
                    print(f"    - Texts: {len(doc.texts)}")
                    print(f"    - Pictures: {len(doc.pictures)}")
                    print(f"    - Tables: {len(doc.tables)}")
                else:
                    print(f"  ✓ Converted successfully")

                processed_count += 1
            else:
                print(f"  ✗ Conversion failed")
                failed_count += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed_count += 1

    print("\n" + "=" * 60)
    print(f"Batch processing complete:")
    print(f"  ✓ Successfully processed: {processed_count}")
    print(f"  ✗ Failed: {failed_count}")


def main():
    """Main test function."""
    # Example usage - update these paths for your test data

    # Sample paths for testing
    case_03_dir = Path("tests/data/cvat_pdfs_dataset_e2e/case_03")
    annotations_xml = case_03_dir.parent / "case_03_annotations.xml"

    # Find all image files in case_03 directory
    sample_paths = sorted(case_03_dir.glob("*.png"))

    for image_path in sample_paths:
        print(f"\nProcessing {image_path.name}...")

        if annotations_xml.exists() and image_path.exists():
            test_conversion_with_sample_data(annotations_xml, image_path, verbose=True)
        else:
            print(f"⚠ Missing files for {image_path.name}:")
            if not annotations_xml.exists():
                print(f"  - Missing annotations.xml at {annotations_xml}")
            if not image_path.exists():
                print(f"  - Missing image at {image_path}")

    # Batch test
    test_batch = False
    if test_batch:
        root_dir = Path("tests/data/cvat_pdfs_dataset_e2e/")  # Update this path

        if root_dir.exists():
            test_batch_conversion(root_dir, verbose=True)
        else:
            print(f"Please update the root directory path: {root_dir}")


if __name__ == "__main__":
    main()
