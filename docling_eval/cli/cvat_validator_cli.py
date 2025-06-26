import argparse
import json
from pathlib import Path
from typing import List, Tuple

from ..cvat_tools.document import DocumentStructure
from ..cvat_tools.models import CVATValidationReport, CVATValidationRunReport
from ..cvat_tools.parser import find_samples_in_directory, get_all_images_from_cvat_xml
from ..cvat_tools.validator import Validator


def process_samples(samples: List[Tuple[str, Path, str]]) -> CVATValidationRunReport:
    """Process a list of samples and return a validation report."""
    validator = Validator()
    reports = []

    for sample_name, xml_path, image_filename in samples:
        try:
            doc = DocumentStructure.from_cvat_xml(xml_path, image_filename)
            report = validator.validate_sample(sample_name, doc)
            # Only include reports that have errors
            if report.errors:
                reports.append(report)

        except Exception as e:
            # Create error report for failed samples
            reports.append(
                CVATValidationReport(
                    sample_name=sample_name,
                    errors=[
                        {
                            "error_type": "processing_error",
                            "message": f"Failed to process sample: {str(e)}",
                        }
                    ],
                )
            )

    return CVATValidationRunReport(samples=reports)


def main():
    """Main CLI for CVAT validation: accepts a directory or XML file, and optional image name."""
    parser = argparse.ArgumentParser(
        description="Validate CVAT annotations for document images."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input directory or XML file",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Image filename to process (if input is a directory or to filter specific image from XML)",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        help="Path to save the validation report JSON file (default: cvat_validation_report.json)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist")
        return

    if input_path.is_dir():
        # Find all samples in directory
        samples = find_samples_in_directory(input_path)
        # Filter by image name if specified
        if args.image:
            samples = [s for s in samples if s[0] == args.image]
            if not samples:
                print(f"Error: No matching image '{args.image}' found in {input_path}")
                return
    else:
        # Single file mode
        if args.image:
            # Process specific image from XML
            samples = [(args.image, input_path, args.image)]
        else:
            # Process all images in XML
            try:
                image_names = get_all_images_from_cvat_xml(input_path)
                if not image_names:
                    print(f"Error: No images found in {input_path}")
                    return
                samples = [(name, input_path, name) for name in image_names]
                print(f"Found {len(samples)} images in {input_path}")
            except Exception as e:
                print(f"Error reading XML file {input_path}: {str(e)}")
                return

    report = process_samples(samples)

    # Determine output file path
    if args.report_file:
        output_path = Path(args.report_file)
    else:
        output_path = Path("cvat_validation_report.json")

    # Save report to JSON file
    try:
        with open(output_path, "w") as f:
            f.write(report.model_dump_json(indent=2, exclude_none=True))
        print(f"Validation report saved to: {output_path.absolute()}")
    except Exception as e:
        print(f"Error saving report to {output_path}: {str(e)}")
        return

    # Also print summary to stdout
    print(
        f"Processed {len(samples)} samples, found {len(report.samples)} samples with errors"
    )


if __name__ == "__main__":
    main()
