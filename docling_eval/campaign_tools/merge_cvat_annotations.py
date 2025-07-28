#!/usr/bin/env python3
"""
Script to merge CVAT annotation XML files based on available image files.

This script:
1. Scans all image files recursively in a given folder (optional)
2. Scans all XML files named 'annotations.xml' recursively in another folder
3. Creates a single CVAT annotation XML with image tags that match available images
   (if image folder provided) or all image tags (if no image folder provided)
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Set


def scan_image_files(image_input_folder: Path) -> Set[str]:
    """
    Scan for all image files recursively in the given folder.

    Args:
        image_input_folder: Path to folder containing images

    Returns:
        Set of image filenames (without path)
    """
    image_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
    }
    image_files: Set[str] = set()

    for ext in image_extensions:
        image_files.update(file.name for file in image_input_folder.rglob(f"*{ext}"))
        image_files.update(
            file.name for file in image_input_folder.rglob(f"*{ext.upper()}")
        )

    return image_files


def scan_annotation_files(xml_input_folder: Path, xml_pattern: str) -> List[Path]:
    """
    Scan for all XML files matching the given pattern recursively in the given folder.

    Args:
        xml_input_folder: Path to folder containing XML files
        xml_pattern: Pattern for XML filenames (e.g., 'annotations.xml', '*set_A.xml')

    Returns:
        List of paths to matching XML files
    """
    return list(xml_input_folder.rglob(xml_pattern))


def extract_image_tags(
    xml_files: List[Path], available_images: Optional[Set[str]] = None
) -> List[ET.Element]:
    """
    Extract image tags from XML files.

    Args:
        xml_files: List of paths to XML annotation files
        available_images: Optional set of available image filenames. If None, all image tags are included.

    Returns:
        List of XML image elements that match available images (if provided) or all image elements
    """
    matching_images = []

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Find all image elements
            for image_elem in root.findall("image"):
                image_name = image_elem.get("name")
                if image_name:
                    # If no available_images filter provided, include all images
                    if available_images is None or image_name in available_images:
                        matching_images.append(image_elem)

        except ET.ParseError as e:
            print(f"Warning: Failed to parse {xml_file}: {e}")
        except Exception as e:
            print(f"Warning: Error processing {xml_file}: {e}")

    return matching_images


def create_merged_annotation_xml(
    image_elements: List[ET.Element], output_path: Path
) -> None:
    """
    Create a merged CVAT annotation XML file with the given image elements.

    Args:
        image_elements: List of XML image elements to include
        output_path: Path where to save the merged XML file
    """
    # Create root annotations element
    root = ET.Element("annotations")

    # Add all matching image elements
    for image_elem in image_elements:
        root.append(image_elem)

    # Create the tree and write to file
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)  # Pretty formatting

    # Write XML with declaration
    with open(output_path, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)

    print(f"Merged annotation XML written to: {output_path}")
    print(f"Total images included: {len(image_elements)}")


def main() -> None:
    """Main function to parse arguments and execute the merging process."""
    parser = argparse.ArgumentParser(
        description="Merge CVAT annotation XML files based on available image files"
    )
    parser.add_argument(
        "xml_input_folder",
        type=Path,
        help="Folder containing XML files to scan recursively",
    )
    parser.add_argument(
        "-i",
        "--image-input-folder",
        type=Path,
        help="Optional folder containing image files to scan recursively. If provided, only annotations for available images are included.",
    )
    parser.add_argument(
        "-p",
        "--xml-pattern",
        type=str,
        default="annotations.xml",
        help="Pattern for XML filenames to include (default: annotations.xml, e.g. '*set_A.xml')",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="merged_annotations.xml",
        help="Output path for merged annotation XML (default: merged_annotations.xml)",
    )

    args = parser.parse_args()

    # Validate input paths
    if args.image_input_folder and not args.image_input_folder.exists():
        print(f"Error: Image folder does not exist: {args.image_input_folder}")
        return

    if not args.xml_input_folder.exists():
        print(f"Error: XML folder does not exist: {args.xml_input_folder}")
        return

    # Determine available images if image folder provided
    available_images: Optional[Set[str]] = None
    if args.image_input_folder:
        print(f"Scanning image files in: {args.image_input_folder}")
        available_images = scan_image_files(args.image_input_folder)
        print(f"Found {len(available_images)} image files")
    else:
        print(
            "No image folder provided - will include all image annotations from XML files"
        )

    print(
        f"Scanning annotation files in: {args.xml_input_folder} with pattern: {args.xml_pattern}"
    )
    xml_files = scan_annotation_files(args.xml_input_folder, args.xml_pattern)
    print(f"Found {len(xml_files)} XML files matching pattern '{args.xml_pattern}'")

    if not xml_files:
        print(f"No XML files found matching pattern '{args.xml_pattern}'!")
        return

    print("Extracting image annotations...")
    matching_images = extract_image_tags(xml_files, available_images)

    if not matching_images:
        if available_images:
            print("No matching image annotations found!")
        else:
            print("No image annotations found in XML files!")
        return

    create_merged_annotation_xml(matching_images, args.output)


if __name__ == "__main__":
    main()
