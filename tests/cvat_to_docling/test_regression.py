"""Regression tests for CVAT to DoclingDocument conversion."""

import json
import os
from pathlib import Path
from typing import Any

import pytest
from docling_core.types.doc import DoclingDocument, ImageRefMode
from docling_core.types.doc.document import ContentLayer

from docling_eval.cvat_tools.cvat_to_docling import convert_cvat_to_docling
from docling_eval.visualisation.visualisations import save_single_document_html


def load_metadata(fixture_dir: Path) -> dict[str, Any]:
    """Load test metadata."""
    metadata_path = fixture_dir / "metadata.json"
    return json.loads(metadata_path.read_text())


def load_expected_output(fixture_dir: Path) -> DoclingDocument:
    """Load expected DoclingDocument output."""
    expected_path = fixture_dir / "expected.json"
    return DoclingDocument.model_validate_json(expected_path.read_text())


def save_expected_output(fixture_dir: Path, doc: DoclingDocument) -> None:
    """Save DoclingDocument as expected output."""
    expected_path = fixture_dir / "expected.json"
    expected_path.write_text(doc.model_dump_json(indent=2))


def discover_fixtures() -> list[Path]:
    """Discover all test fixtures."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    if not fixtures_dir.exists():
        return []

    fixtures = []
    for item in sorted(fixtures_dir.iterdir()):
        if item.is_dir() and (item / "metadata.json").exists():
            fixtures.append(item)

    return fixtures


# Discover all fixtures for parametrization
FIXTURES = discover_fixtures()
FIXTURE_IDS = [f.name for f in FIXTURES]

# Check if we're in generation mode
GENERATE_MODE = os.environ.get("DOCLING_GEN_TEST_DATA", "").lower() in (
    "1",
    "true",
    "yes",
)

# Check if we should generate visualizations
GENERATE_VIZ = os.environ.get("DOCLING_GEN_VIZ", "").lower() in ("1", "true", "yes")

# Visualization output directory
VIZ_OUTPUT_DIR = Path(__file__).parent.parent.parent / "scratch" / "cvat_regression_viz"


@pytest.mark.parametrize("fixture_dir", FIXTURES, ids=FIXTURE_IDS)
def test_cvat_to_docling_regression(fixture_dir: Path) -> None:
    """Test CVAT to DoclingDocument conversion against expected output."""
    # Load test metadata
    metadata = load_metadata(fixture_dir)

    # Input paths - check for PDF first, then PNG
    xml_path = fixture_dir / "input.xml"
    pdf_path = fixture_dir / "input.pdf"
    png_path = fixture_dir / "input.png"

    assert xml_path.exists(), f"Missing input.xml in {fixture_dir}"

    # Determine input type and path
    if pdf_path.exists():
        input_path = pdf_path
        input_type = "pdf"
        page_number = metadata.get("page_number", 1)
    elif png_path.exists():
        input_path = png_path
        input_type = "png"
        page_number = None
    else:
        pytest.fail(f"Missing input file (input.pdf or input.png) in {fixture_dir}")

    # Get conversion parameters
    conversion_params = metadata.get("conversion_params", {})
    force_ocr = conversion_params.get("force_ocr", False)
    ocr_scale = conversion_params.get("ocr_scale", 3.0)

    # Set defaults based on input type
    # PDF: cvat_input_scale=2.0, storage_scale=2.0 (144 DPI)
    # PNG: cvat_input_scale=1.0, storage_scale=1.0 (72 DPI)
    if input_type == "pdf":
        default_cvat_input_scale = 2.0
        default_storage_scale = 2.0
    else:
        default_cvat_input_scale = 1.0
        default_storage_scale = 1.0

    # Allow metadata to override defaults (only if explicitly set)
    cvat_input_scale = conversion_params.get(
        "cvat_input_scale", default_cvat_input_scale
    )
    storage_scale = conversion_params.get("storage_scale", default_storage_scale)

    # Get image identifier from metadata
    image_identifier = metadata["source"]["image_identifier"]

    # Perform conversion
    actual_doc = convert_cvat_to_docling(
        xml_path=xml_path,
        input_path=input_path,
        image_identifier=image_identifier,
        force_ocr=force_ocr,
        ocr_scale=ocr_scale,
        cvat_input_scale=cvat_input_scale,
        storage_scale=storage_scale,
    )

    assert actual_doc is not None, f"Conversion failed for {fixture_dir.name}"

    # Generate visualizations if requested
    if GENERATE_VIZ or GENERATE_MODE:
        viz_dir = VIZ_OUTPUT_DIR / fixture_dir.name
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Save HTML visualization
        actual_doc.save_as_html(
            viz_dir / "output.html",
            image_mode=ImageRefMode.EMBEDDED,
            split_page_view=True,
            included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE},
        )

        # Save JSON for inspection
        actual_doc.save_as_json(viz_dir / "output.json")

        # Save visualizations with reading order (same as test_cvat_to_docling_cli.py)
        # This generates:
        #   - visualization_layout.html (layout with reading order overlay)
        #   - visualization_key_value.html (if key-value items exist)
        visualization_path = viz_dir / "visualization.html"
        save_single_document_html(
            visualization_path, actual_doc, draw_reading_order=True
        )

    if GENERATE_MODE:
        # Generate expected output
        save_expected_output(fixture_dir, actual_doc)
        pytest.skip(f"Generated expected output for {fixture_dir.name}")
    else:
        # Compare with expected output
        expected_path = fixture_dir / "expected.json"
        if not expected_path.exists():
            pytest.fail(
                f"Missing expected.json in {fixture_dir}. "
                f"Run with DOCLING_GEN_TEST_DATA=1 to generate it."
            )

        expected_doc = load_expected_output(fixture_dir)

        # Serialize and deserialize actual_doc to match the behavior of expected_doc
        # This ensures both go through the same serialization cycle
        actual_doc_json = actual_doc.export_to_dict()
        actual_doc = DoclingDocument.model_validate(actual_doc_json)

        # Normalize references before comparison for deterministic equality
        actual_doc._normalize_references()
        expected_doc._normalize_references()

        # Compare using DoclingDocument equality
        matches = actual_doc == expected_doc

        # Get observation status
        observation_status = metadata.get("observation_status", "unknown")

        # Handle broken tests
        if matches and observation_status == "broken":
            # Reproduced the known broken behavior - expected failure
            observation = metadata.get("observation", "No observation recorded")
            pytest.xfail(f"Expected failure: {observation}")

        if not matches and observation_status == "broken":
            # Output differs from the known broken expected.json - unexpected
            pytest.fail(
                f"Test {fixture_dir.name} is marked as 'broken' but produced unexpected output "
                f"(differs from the known broken expected.json)."
            )

        # For correct/unknown tests, assert equality (will fail if not matches)
        assert matches, (
            f"Conversion output differs from expected for {fixture_dir.name}. "
            f"Test ID: {metadata['test_id']}, Description: {metadata['description']}"
        )


def test_fixtures_exist() -> None:
    """Sanity check that we have fixtures to test."""
    assert (
        len(FIXTURES) > 0
    ), "No test fixtures found. Run tests/cvat_to_docling/build_fixtures.py to create fixtures."


if __name__ == "__main__":
    # Allow running this file directly for debugging
    pytest.main([__file__, "-v"])
