#!/usr/bin/env python3
"""
Script to collect images from CVAT XML annotation file.

This script:
1. Parses a CVAT XML annotation file to extract image filenames
2. Searches for these images in subdirectories containing cvat_tasks folders
3. Only considers subdirectories that contain a 'cvat_tasks' folder
4. Copies found images to an output directory
"""

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Set


def extract_image_filenames(xml_path: Path) -> Set[str]:
    """Extract image filenames from CVAT XML file."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Find all image elements and extract their 'name' attributes
        image_filenames = set()
        for image_elem in root.findall(".//image"):
            name_attr = image_elem.get("name")
            if name_attr:
                image_filenames.add(name_attr)

        return image_filenames
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error reading XML file: {e}", file=sys.stderr)
        sys.exit(1)


def find_images_in_subdirectories(
    root_dir: Path, image_filenames: Set[str]
) -> dict[str, Path]:
    """Find images in subdirectories that contain 'cvat_tasks' folder."""
    found_images = {}

    # Walk through all subdirectories
    for subdir in root_dir.rglob("*"):
        if not subdir.is_dir():
            continue

        # Check if this subdirectory contains a 'cvat_tasks' folder
        cvat_tasks_path = subdir / "cvat_tasks"
        if not cvat_tasks_path.exists() or not cvat_tasks_path.is_dir():
            continue

        # Search recursively within this subdirectory for images
        for image_filename in image_filenames:
            # Look for the image in this directory and all its subdirectories
            for potential_image_path in subdir.rglob(image_filename):
                if potential_image_path.is_file():
                    found_images[image_filename] = potential_image_path
                    break  # Found this image, move to next filename

    return found_images


def copy_images_to_output(found_images: dict[str, Path], output_dir: Path) -> None:
    """Copy found images to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    for image_filename, source_path in found_images.items():
        dest_path = output_dir / image_filename

        try:
            shutil.copy2(source_path, dest_path)
            print(f"Copied: {source_path} -> {dest_path}")
            copied_count += 1
        except Exception as e:
            print(f"Error copying {source_path}: {e}", file=sys.stderr)

    print(f"\nSuccessfully copied {copied_count} images to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect images from CVAT XML annotation file"
    )
    parser.add_argument("xml_file", type=Path, help="Path to CVAT XML annotation file")
    parser.add_argument(
        "root_dir", type=Path, help="Root directory to search for images"
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for collected images"
    )

    args = parser.parse_args()

    # Validate input file exists
    if not args.xml_file.exists():
        print(f"Error: XML file '{args.xml_file}' does not exist", file=sys.stderr)
        sys.exit(1)

    if not args.root_dir.exists():
        print(
            f"Error: Root directory '{args.root_dir}' does not exist", file=sys.stderr
        )
        sys.exit(1)

    print(f"Parsing XML file: {args.xml_file}")
    image_filenames = extract_image_filenames(args.xml_file)
    print(f"Found {len(image_filenames)} image filenames in XML")

    print(f"Searching for images in: {args.root_dir}")
    found_images = find_images_in_subdirectories(args.root_dir, image_filenames)
    print(
        f"Found {len(found_images)} images in subdirectories with 'cvat_tasks' folders"
    )

    if not found_images:
        print("No images found. Exiting.")
        return

    # Show which images were found
    print("\nFound images:")
    for filename, path in found_images.items():
        print(f"  {filename} -> {path}")

    # Show missing images
    missing_images = image_filenames - set(found_images.keys())
    if missing_images:
        print(f"\nMissing images ({len(missing_images)}):")
        for filename in sorted(missing_images):
            print(f"  {filename}")

    print(f"\nCopying images to: {args.output_dir}")
    copy_images_to_output(found_images, args.output_dir)


if __name__ == "__main__":
    main()
