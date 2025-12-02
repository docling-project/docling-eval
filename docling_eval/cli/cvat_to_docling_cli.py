import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from docling_core.types.doc.base import ImageRefMode
from pydantic import BaseModel

from docling_eval.cvat_tools.cvat_to_docling import (
    CVATConversionResult,
    convert_cvat_folder_to_docling,
    convert_cvat_to_docling,
)
from docling_eval.cvat_tools.folder_models import CVATFolderStructure
from docling_eval.cvat_tools.folder_parser import parse_cvat_folder
from docling_eval.cvat_tools.parser import find_samples_in_directory


class ConversionResult(BaseModel):
    """Result of a single conversion."""

    sample_name: str
    output_files: Dict[str, str] = {}


class ConversionFailure(BaseModel):
    """Failed conversion result."""

    sample_name: str
    error: str
    details: Optional[Dict[str, Any]] = None


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

    def add_failure(
        self,
        sample_name: str,
        error_message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Add a failed conversion."""
        self.failed_conversions.append(
            ConversionFailure(
                sample_name=sample_name, error=error_message, details=details
            )
        )


def process_samples_for_conversion(
    samples: List[Tuple[str, Path, str]],
    output_dir: Path,
    save_formats: Optional[List[str]] = None,
    verbose: bool = False,
    force_ocr: bool = False,
    ocr_scale: float = 1.0,
) -> BatchConversionReport:
    """Process a list of samples and convert them to DoclingDocuments.

    Args:
        samples: List of (sample_name, xml_path, image_filename) tuples
        output_dir: Directory to save output files
        save_formats: List of formats to save (json, html, md, txt)
        verbose: Whether to print detailed information
        force_ocr: Force OCR on PDFs instead of using native text layer
        ocr_scale: Scale factor for rendering PDFs for OCR

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
            doc = convert_cvat_to_docling(
                xml_path, image_path, force_ocr=force_ocr, ocr_scale=ocr_scale
            )

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


def process_cvat_folder(
    folder_path: Path,
    output_dir: Path,
    xml_pattern: str = "task_{xx}_set_A",
    save_formats: Optional[List[str]] = None,
    verbose: bool = False,
    folder_structure: Optional[CVATFolderStructure] = None,
    log_validation: bool = False,
    force_ocr: bool = False,
    ocr_scale: float = 1.0,
) -> BatchConversionReport:
    """Process a CVAT export folder for document-level conversion."""

    if save_formats is None:
        save_formats = ["json", "html", "md"]

    report = BatchConversionReport()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if folder_structure is None:
            folder_structure = parse_cvat_folder(folder_path, xml_pattern)
        outcomes = convert_cvat_folder_to_docling(
            folder_path,
            xml_pattern,
            output_dir,
            save_formats,
            folder_structure=folder_structure,
            log_validation=log_validation,
            force_ocr=force_ocr,
            ocr_scale=ocr_scale,
        )

        for doc_hash, outcome in outcomes.items():
            cvat_doc = folder_structure.documents[doc_hash]
            base_filename = cvat_doc.doc_name

            if outcome.document is not None:
                output_files: Dict[str, str] = {}
                for fmt in save_formats:
                    if fmt == "json":
                        output_files["json"] = str(output_dir / f"{base_filename}.json")
                    elif fmt == "html":
                        output_files["html"] = str(output_dir / f"{base_filename}.html")
                    elif fmt == "md":
                        output_files["markdown"] = str(
                            output_dir / f"{base_filename}.md"
                        )
                    elif fmt == "txt":
                        output_files["txt"] = str(output_dir / f"{base_filename}.txt")
                    elif fmt == "viz":
                        output_files["viz"] = str(
                            output_dir / f"{base_filename}_docling_p*.png"
                        )

                report.add_success(base_filename, output_files)

                if verbose:
                    print(f"  ✓ Converted {base_filename}")
                    doc = outcome.document
                    pages_attr = getattr(doc, "pages", None)
                    texts_attr = getattr(doc, "texts", None)
                    pictures_attr = getattr(doc, "pictures", None)
                    tables_attr = getattr(doc, "tables", None)

                    if pages_attr is not None:
                        try:
                            print(f"    - Pages: {len(pages_attr)}")
                        except TypeError:
                            pass
                    if texts_attr is not None:
                        try:
                            print(f"    - Texts: {len(texts_attr)}")
                        except TypeError:
                            pass
                    if pictures_attr is not None:
                        try:
                            print(f"    - Pictures: {len(pictures_attr)}")
                        except TypeError:
                            pass
                    if tables_attr is not None:
                        try:
                            print(f"    - Tables: {len(tables_attr)}")
                        except TypeError:
                            pass
                else:
                    print(f"  ✓ {base_filename}")
            else:
                error_message = outcome.error or "Conversion returned None"
                validation_details: Dict[str, Any] = {}
                if outcome.validation_report is not None:
                    validation_details["aggregated_report"] = (
                        outcome.validation_report.model_dump()
                    )
                if outcome.per_page_reports:
                    validation_details["per_page_reports"] = {
                        page_name: report.model_dump()
                        for page_name, report in outcome.per_page_reports.items()
                    }
                details_payload: Optional[Dict[str, Any]] = (
                    validation_details if validation_details else None
                )
                report.add_failure(
                    base_filename,
                    error_message,
                    details=details_payload if log_validation else None,
                )
                if verbose and log_validation and details_payload is not None:
                    print(f"  ✗ {base_filename}: {error_message}")
                    print(json.dumps(details_payload, indent=2))
                elif verbose:
                    print(f"  ✗ {base_filename}: {error_message}")
                else:
                    print(f"  ✗ {base_filename}")

    except Exception as exc:
        report.add_failure("folder_processing", str(exc))
        if verbose:
            print(f"  ✗ Error: {exc}")

    return report


def main():
    """Main CLI for CVAT batch conversion."""
    parser = argparse.ArgumentParser(
        description="Convert CVAT annotations to DoclingDocuments in batch."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to input directory or XML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for converted documents (default: input_path/docling_output)",
    )
    parser.add_argument(
        "--folder-mode",
        action="store_true",
        help="Process CVAT folder structure instead of individual files",
    )
    parser.add_argument(
        "--xml-pattern",
        type=str,
        default="task_{xx}_set_A",
        help="XML file pattern (task_{xx}_set_A, task_{xx}_set_B, task_{xx}_preannotate)",
    )
    parser.add_argument(
        "--tasks-root",
        type=str,
        default=None,
        help="Optional path whose 'cvat_tasks' directory contains the annotation XMLs",
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
        "--log-validation",
        action="store_true",
        default=False,
        help="Log validation reports for each document",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to save conversion report as JSON (default: None)",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        default=False,
        help="Force OCR on PDFs instead of using native text layer (default: False)",
    )
    parser.add_argument(
        "--ocr-scale",
        type=float,
        default=1.0,
        help="Scale factor for rendering PDFs for OCR (default: 1.0 = 72 DPI). Higher values increase OCR quality but use more memory.",
    )

    args = parser.parse_args()

    if not args.input_path:
        print("Error: --input_path is required")
        return

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist")
        return

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if input_path.is_dir():
            output_dir = input_path / "docling_output"
        else:
            output_dir = input_path.parent / "docling_output"

    if args.folder_mode:
        if not input_path.is_dir():
            print("Error: folder-mode requires --input_path to be a directory")
            return

        tasks_root: Optional[Path] = None
        if args.tasks_root is not None:
            tasks_root = Path(args.tasks_root)
            if not tasks_root.exists():
                print(f"Error: tasks-root {tasks_root} does not exist")
                return
            if not tasks_root.is_dir():
                print(f"Error: tasks-root {tasks_root} is not a directory")
                return
            tasks_root = tasks_root.resolve()

        try:
            folder_structure = parse_cvat_folder(
                input_path,
                args.xml_pattern,
                tasks_root=tasks_root,
            )
        except Exception as exc:  # pragma: no cover - CLI feedback
            print(f"Error: failed to parse CVAT folder - {exc}")
            return

        print(f"Found {len(folder_structure.documents)} documents to process")
        print(f"Output directory: {output_dir}")
        print("=" * 60)

        report = process_cvat_folder(
            folder_path=input_path,
            output_dir=output_dir,
            xml_pattern=args.xml_pattern,
            save_formats=args.formats,
            verbose=args.verbose,
            folder_structure=folder_structure,
            log_validation=args.log_validation,
            force_ocr=args.force_ocr,
            ocr_scale=args.ocr_scale,
        )
    else:
        if input_path.is_dir():
            samples = find_samples_in_directory(input_path)
            if args.image:
                samples = [s for s in samples if s[0] == args.image]
                if not samples:
                    print(
                        f"Error: No matching image '{args.image}' found in {input_path}"
                    )
                    return
        else:
            if not args.image:
                print(
                    "Error: --image argument required when processing a single XML file"
                )
                return
            samples = [(args.image, input_path, args.image)]

        if not samples:
            print("No samples found to process")
            return

        print(f"Found {len(samples)} samples to process")
        print(f"Output directory: {output_dir}")
        print("=" * 60)

        report = process_samples_for_conversion(
            samples=samples,
            output_dir=output_dir,
            save_formats=args.formats,
            verbose=args.verbose,
            force_ocr=args.force_ocr,
            ocr_scale=args.ocr_scale,
        )

    print("\n" + "=" * 60)
    print("Batch conversion complete:")
    print(f"  ✓ Successfully processed: {report.successful_count}")
    print(f"  ✗ Failed: {report.failed_count}")

    if args.report:
        report_path = Path(args.report)
        with open(report_path, "w") as f:
            f.write(report.model_dump_json(indent=2))
        print(f"  📄 Report saved to: {report_path}")

    if not args.verbose:
        print("\nConversion Report:")
        print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
