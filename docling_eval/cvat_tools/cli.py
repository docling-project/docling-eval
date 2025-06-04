import argparse
import json
from pathlib import Path
from typing import List, Tuple

from .document import DocumentStructure
from .models import CVATValidationRunReport, CVATValidationReport
from .parser import find_samples_in_directory
from .validator import Validator


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
        help="Image filename to process (if input is a directory)",
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
        if not args.image:
            print("Error: --image argument required when processing a single XML file")
            return
        samples = [(args.image, input_path, args.image)]

    report = process_samples(samples)
    print(json.dumps(report.model_dump(), indent=2))


if __name__ == "__main__":
    main()
