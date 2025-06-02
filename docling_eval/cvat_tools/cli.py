import argparse
import json
from pathlib import Path
from typing import List, Tuple

from .models import ValidationRunReport
from .parser import find_samples_in_directory, parse_cvat_xml_for_image
from .tree import (
    associate_reading_order_paths_to_containers,
    build_containment_tree,
    map_path_points_to_elements,
)
from .validator import Validator


def process_samples(samples: List[Tuple[str, Path, str]]) -> ValidationRunReport:
    """Process a list of samples and return a validation report."""
    validator = Validator()
    reports = []

    for sample_name, xml_path, image_filename in samples:
        try:
            elements, paths, _ = parse_cvat_xml_for_image(xml_path, image_filename)
            tree_roots = build_containment_tree(elements)
            path_to_elements = map_path_points_to_elements(paths, elements)
            path_to_container = associate_reading_order_paths_to_containers(
                path_to_elements, tree_roots
            )

            report = validator.validate_sample(
                sample_name,
                elements,
                paths,
                tree_roots,
                path_to_elements,
                path_to_container,
            )
            # Only include reports that have errors
            if report.errors:
                reports.append(report)

        except Exception as e:
            # Create error report for failed samples
            reports.append(
                ValidationReport(
                    sample_name=sample_name,
                    errors=[
                        {
                            "error_type": "processing_error",
                            "message": f"Failed to process sample: {str(e)}",
                        }
                    ],
                )
            )

    return ValidationRunReport(samples=reports)


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="CVAT annotation validation tool")
    parser.add_argument(
        "input_path", type=str, help="Path to root directory or annotations.xml file"
    )
    parser.add_argument(
        "--image-name",
        type=str,
        default=None,
        help="If input is XML, only process <image> with this name",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="validation_report.json",
        help="Path to output JSON file (default: validation_report.json in current directory)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    samples = []

    if input_path.is_dir():
        samples = find_samples_in_directory(input_path)
    else:
        if args.image_name:
            # Only process the specified image
            samples = [(args.image_name, input_path, args.image_name)]
        else:
            # Process all images in the XML file
            samples = [(input_path.name, input_path, input_path.name)]

    report = process_samples(samples)

    # Write report to JSON file
    output_path = Path(args.output)
    with output_path.open("w") as f:
        json.dump(report.model_dump(), f, indent=2)

    print(f"Validation report written to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
