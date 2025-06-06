import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from docling_core.types.doc.base import ImageRefMode
from pydantic import BaseModel

from docling_eval.cvat_tools.cvat_to_docling import convert_cvat_to_docling
from docling_eval.cvat_tools.parser import find_samples_in_directory


class ConversionResult(BaseModel):
    """Result of a single conversion."""

    sample_name: str
    output_files: Dict[str, str] = {}


class ConversionFailure(BaseModel):
    """Failed conversion result."""

    sample_name: str
    error: str


class BatchConversionReport(BaseModel):
    """Report for batch conversion results."""

    successful_conversions: List[ConversionResult] = []
    failed_conversions: List[ConversionFailure] = []

    @property
    def total_processed(self) -> int:
        """Total number of processed samples."""
        return len(self.successful_conversions) + len(self.failed_conversions)

    @property
    def successful_count(self) -> int:
        """Number of successful conversions."""
        return len(self.successful_conversions)

    @property
    def failed_count(self) -> int:
        """Number of failed conversions."""
        return len(self.failed_conversions)

    def add_success(self, sample_name: str, output_files: Dict[str, str]):
        """Add a successful conversion."""
        self.successful_conversions.append(
            ConversionResult(sample_name=sample_name, output_files=output_files)
        )

    def add_failure(self, sample_name: str, error_message: str):
        """Add a failed conversion."""
        self.failed_conversions.append(
            ConversionFailure(sample_name=sample_name, error=error_message)
        )


def process_samples_for_conversion(
    samples: List[Tuple[str, Path, str]],
    output_dir: Path,
    save_formats: Optional[List[str]] = None,
    verbose: bool = False,
) -> BatchConversionReport:
    """Process a list of samples and convert them to DoclingDocuments.

    Args:
        samples: List of (sample_name, xml_path, image_filename) tuples
        output_dir: Directory to save output files
        ocr_framework: OCR framework to use for conversion
        save_formats: List of formats to save (json, html, md, txt)
        verbose: Whether to print detailed information

    Returns:
        BatchConversionReport with conversion results
    """
    if save_formats is None:
        save_formats = ["json", "html", "md", "txt", "viz"]

    report = BatchConversionReport()
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_name, xml_path, image_filename in samples:
        if verbose:
            print(f"Processing {sample_name}...")

        try:
            # Find the actual image file path
            image_dir = xml_path.parent
            image_extensions = [
                ".png",
                ".jpg",
                ".jpeg",
                ".pdf",
                ".tif",
                ".tiff",
                ".bmp",
            ]
            image_path = None

            for ext in image_extensions:
                potential_path = image_dir / f"{Path(image_filename).stem}{ext}"
                if potential_path.exists():
                    image_path = potential_path
                    break

            if image_path is None:
                # Try the exact filename
                image_path = image_dir / image_filename
                if not image_path.exists():
                    raise FileNotFoundError(
                        f"Image file not found for {image_filename}"
                    )

            # Convert to DoclingDocument
            doc = convert_cvat_to_docling(xml_path, image_path)

            if doc:
                # Prepare output files
                output_files = {}
                stem = Path(image_filename).stem

                # Save in different formats
                if "json" in save_formats:
                    json_output = output_dir / f"{stem}.json"
                    doc.save_as_json(json_output)
                    output_files["json"] = str(json_output)

                if "html" in save_formats:
                    html_output = output_dir / f"{stem}.html"
                    doc.save_as_html(
                        html_output,
                        image_mode=ImageRefMode.EMBEDDED,
                        split_page_view=True,
                    )
                    output_files["html"] = str(html_output)

                if "md" in save_formats:
                    md_output = output_dir / f"{stem}.md"
                    doc.save_as_markdown(md_output, image_mode=ImageRefMode.EMBEDDED)
                    output_files["markdown"] = str(md_output)

                if "txt" in save_formats:
                    txt_output = output_dir / f"{stem}.txt"
                    with open(txt_output, "w") as fp:
                        fp.write(doc.export_to_element_tree())
                    output_files["txt"] = str(txt_output)

                if "viz" in save_formats:
                    viz_imgs = doc.get_visualization()
                    for page_no, img in viz_imgs.items():
                        if page_no is not None:
                            img.save(output_dir / f"{stem}_docling_p{page_no}.png")

                report.add_success(sample_name, output_files)

                if verbose:
                    print(f"  ✓ Converted successfully")
                    print(f"    - Texts: {len(doc.texts)}")
                    print(f"    - Pictures: {len(doc.pictures)}")
                    print(f"    - Tables: {len(doc.tables)}")
                    print(f"    - Output files: {list(output_files.keys())}")
                else:
                    print(f"  ✓ {sample_name}")

            else:
                report.add_failure(sample_name, "Conversion returned None")
                if verbose:
                    print(f"  ✗ Conversion failed - returned None")

        except Exception as e:
            error_message = str(e)
            report.add_failure(sample_name, error_message)
            if verbose:
                print(f"  ✗ Error: {error_message}")
            else:
                print(f"  ✗ {sample_name}: {error_message}")

    return report


def main():
    """Main CLI for CVAT batch conversion: converts CVAT annotations to DoclingDocuments."""
    parser = argparse.ArgumentParser(
        description="Convert CVAT annotations to DoclingDocuments in batch."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input directory or XML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for converted documents (default: input_path/docling_output)",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Image filename to process (optional, if input is a directory)",
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["json", "html", "md", "txt", "viz"],
        choices=["json", "html", "md", "txt", "viz"],
        help="Output formats to save (default: json html md txt viz)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print detailed information during processing (default: False)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to save conversion report as JSON (default: None)",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist")
        return

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if input_path.is_dir():
            output_dir = input_path / "docling_output"
        else:
            output_dir = input_path.parent / "docling_output"

    # Find samples to process
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

    if not samples:
        print("No samples found to process")
        return

    print(f"Found {len(samples)} samples to process")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Process the samples
    report = process_samples_for_conversion(
        samples=samples,
        output_dir=output_dir,
        save_formats=args.formats,
        verbose=args.verbose,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Batch conversion complete:")
    print(f"  ✓ Successfully processed: {report.successful_count}")
    print(f"  ✗ Failed: {report.failed_count}")

    # Save report if requested
    if args.report:
        report_path = Path(args.report)
        with open(report_path, "w") as f:
            f.write(report.model_dump_json(indent=2))
        print(f"  📄 Report saved to: {report_path}")

    # Also print report to stdout in JSON format if not verbose
    if not args.verbose:
        print("\nConversion Report:")
        print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
