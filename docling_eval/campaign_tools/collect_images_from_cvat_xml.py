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
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Set

from docling_eval.cvat_tools.parser import get_all_images_from_cvat_xml


def extract_image_filenames(xml_path: Path) -> Set[str]:
    """Extract image filenames from CVAT XML file."""
    try:
        return set(get_all_images_from_cvat_xml(xml_path))
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error reading XML file: {e}", file=sys.stderr)
        sys.exit(1)


def find_images_in_subdirectories(
    root_dir: Path, image_filenames: Set[str]
) -> dict[str, Path]:
    """Find requested images using CVAT folder metadata when available."""

    found_images: dict[str, Path] = {}
    remaining = set(image_filenames)
    processed_folders: set[Path] = set()

    for overview_path in root_dir.rglob("cvat_overview.json"):
        folder_path = overview_path.parent
        resolved_folder = folder_path.resolve()
        if resolved_folder in processed_folders:
            continue
        processed_folders.add(resolved_folder)

        try:
            from docling_eval.cvat_tools.folder_parser import parse_cvat_folder

            folder_structure = parse_cvat_folder(folder_path)
        except Exception:
            continue

        for document in folder_structure.documents.values():
            for page in document.pages:
                if page.image_filename in remaining:
                    found_images[page.image_filename] = page.image_path
        remaining -= found_images.keys()
        if not remaining:
            return found_images

    if not remaining:
        return found_images

    # Fallback search: look for images next to annotations.xml outside processed folders
    for dirpath, _, _ in os.walk(root_dir):
        current_dir = Path(dirpath).resolve()
        if any(
            current_dir == folder or current_dir.is_relative_to(folder)
            for folder in processed_folders
        ):
            continue

        xml_path = Path(dirpath) / "annotations.xml"
        if not xml_path.exists():
            continue

        for filename in list(remaining):
            candidate = Path(dirpath) / filename
            if candidate.exists():
                found_images[filename] = candidate
                remaining.remove(filename)

        if not remaining:
            break

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
