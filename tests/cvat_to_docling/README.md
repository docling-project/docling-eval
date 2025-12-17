# CVAT to DoclingDocument Regression Test Suite

Regression test suite for CVAT XML to DoclingDocument conversion with 20 unique test fixtures covering various edge cases and known issues.

## Quick Start

```bash
# Run all tests
uv run pytest tests/cvat_to_docling/test_regression.py -v

# Run specific test
uv run pytest tests/cvat_to_docling/test_regression.py::test_cvat_to_docling_regression[001_rotation_issue] -v

# Generate visualizations for debugging
DOCLING_GEN_VIZ=1 uv run pytest tests/cvat_to_docling/test_regression.py -k "002_tables_lost" -v
```

## Structure

```
tests/cvat_to_docling/
├── README.md              # This file
├── conftest.py           # pytest configuration
├── test_regression.py    # Main test suite
└── fixtures/             # 20 test fixtures
    ├── 001_rotation_issue/
    │   ├── input.xml         # Single-image CVAT XML
    │   ├── input.png         # Page image (PNG for Canva sources)
    │   ├── metadata.json     # Test metadata and conversion params
    │   └── expected.json     # Expected DoclingDocument output
    ├── 002_tables_lost/
    │   ├── input.xml         # Single-image CVAT XML
    │   ├── input.pdf         # Source PDF (for DocLayNet/Misc sources)
    │   ├── reference.png     # Reference PNG (kept for visual inspection)
    │   ├── metadata.json     # Includes page_number for PDF
    │   └── expected.json     # Expected DoclingDocument output
    └── ...
```

## Running Tests

### Normal Mode (Comparison)
```bash
uv run pytest tests/cvat_to_docling/test_regression.py -v
```

### Regenerate Expected Outputs
After fixing bugs or updating conversion logic:
```bash
DOCLING_GEN_TEST_DATA=1 uv run pytest tests/cvat_to_docling/test_regression.py -v
```

### Generate Visualizations
Generate visual outputs in `scratch/cvat_regression_viz/` for inspection:
```bash
# All tests
DOCLING_GEN_VIZ=1 uv run pytest tests/cvat_to_docling/test_regression.py -v

# Specific test
DOCLING_GEN_VIZ=1 uv run pytest tests/cvat_to_docling/test_regression.py::test_cvat_to_docling_regression[002_tables_lost] -v

# Category of tests
DOCLING_GEN_VIZ=1 uv run pytest tests/cvat_to_docling/test_regression.py -k "table" -v

# Only failing tests
DOCLING_GEN_VIZ=1 uv run pytest tests/cvat_to_docling/test_regression.py --lf -v
```

**Generated visualization files:**
- `output.html` - DoclingDocument HTML export
- `output.json` - Raw DoclingDocument JSON
- `visualization_layout.html` - Layout with reading order overlay
- `visualization_key_value.html` - Key-value pairs (if present)

## Test Fixtures

20 unique fixtures covering:
- **Rotation issues** - Text with rotation attributes
- **Table issues** - Tables dropped or mishandled
- **Reading order issues** - Elements not in correct order
- **Key-value issues** - KVP parsing problems
- **Layout issues** - Complex layout handling

### Input Types

The test automatically detects input type (checks for `input.pdf` first, then `input.png`).

## Workflow for Development

### 1. Debugging Failed Tests

```bash
# Find failing test
uv run pytest tests/cvat_to_docling/test_regression.py -v

# Generate visualization
DOCLING_GEN_VIZ=1 uv run pytest tests/cvat_to_docling/test_regression.py::test_cvat_to_docling_regression[002_tables_lost] -v

# Inspect
open scratch/cvat_regression_viz/002_tables_lost/visualization_layout.html
open tests/cvat_to_docling/fixtures/002_tables_lost/input.png
```

### 2. After Fixing Bug

```bash
# Test the fix
uv run pytest tests/cvat_to_docling/test_regression.py::test_cvat_to_docling_regression[002_tables_lost] -v

# If correct, regenerate expected output
DOCLING_GEN_TEST_DATA=1 uv run pytest tests/cvat_to_docling/test_regression.py::test_cvat_to_docling_regression[002_tables_lost] -v

# Verify no regressions
uv run pytest tests/cvat_to_docling/test_regression.py -v
```

### 3. Adding New Test Case

```bash
# Create fixture directory
mkdir -p tests/cvat_to_docling/fixtures/025_new_issue

# Add files:
# - input.xml (single-image CVAT XML)
# - input.png (corresponding page image)
# - metadata.json (see format below)

# Generate expected output
DOCLING_GEN_TEST_DATA=1 uv run pytest tests/cvat_to_docling/test_regression.py::test_cvat_to_docling_regression[025_new_issue] -v

# Verify
uv run pytest tests/cvat_to_docling/test_regression.py::test_cvat_to_docling_regression[025_new_issue] -v
```

## Metadata Format

Each fixture requires a minimal `metadata.json`:

```json
{
  "test_id": "002",
  "name": "tables_lost",
  "description": "Table hit by reading order but dropped",
  "source": {
    "image_identifier": "doc_xxxxx_page_000001.png",
    "doc_name": "original_document_name.pdf"
  },
  "input_type": "pdf",
  "page_number": 1
}
```

**Fields:**
- `test_id`: Unique numeric ID
- `name`: Descriptive name
- `description`: What this test case covers
- `source.image_identifier`: Image name from CVAT XML (required)
- `source.doc_name`: Original document filename (optional)
- `input_type`: "pdf" or omit for PNG
- `page_number`: Page number for PDFs (1-indexed)
- `conversion_params`: Only include non-default values
  - Defaults: `force_ocr=False`, `ocr_scale=3.0`
  - PDF defaults: `cvat_input_scale=2.0`, `storage_scale=2.0`
  - PNG defaults: `cvat_input_scale=1.0`, `storage_scale=1.0`

## Understanding Test Results

### Passing Tests ✅
Conversion works correctly and produces expected output.

### Failing Tests ❌
Represent known bugs/issues in the conversion. This is expected and useful:
- Each failure documents a specific bug
- Use visualizations to understand the issue
- Fix bug → regenerate expected output → test passes

## Visualization Features

Generated HTML visualizations show:
- **Layout**: Bounding boxes for all elements
- **Reading order**: Numbered arrows showing flow
- **Key-value pairs**: Highlighted connections (if present)
- **Tables**: Row/column structure

Perfect for:
- Debugging conversion issues
- Understanding why tests fail
- Comparing input (CVAT) vs output (DoclingDocument)
- Visual verification of fixes

## Notes

- **Reference normalization**: Tests call `_normalize_references()` before comparison for deterministic results
- **Scale handling**: PDF fixtures use 2x scale (144 DPI), PNG fixtures use 1x (72 DPI)
- **Metadata is minimal**: Only image_identifier, doc_name, and non-default conversion params
- **Scratch directory**: `scratch/cvat_regression_viz/` is git-ignored, safe to generate freely

## Current Status

✅ **Infrastructure complete** - 19 fixtures ready (15 PDF, 4 PNG)
⏭️ **Next step** - Regenerate expected outputs:
```bash
DOCLING_GEN_TEST_DATA=1 uv run pytest tests/cvat_to_docling/test_regression.py -v
```

## Maintenance

Clean up visualizations:
```bash
rm -rf scratch/cvat_regression_viz/
```
