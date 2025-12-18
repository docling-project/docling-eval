# CVAT to DoclingDocument Regression Test Suite

Regression test suite for CVAT XML to DoclingDocument conversion covering various edge cases and known issues.

## Quick Start

```bash
# Run all tests
uv run pytest tests/cvat_to_docling/test_regression.py -v

# Run specific test
uv run pytest tests/cvat_to_docling/test_regression.py -k "002b_table_order" -v

# Generate visualizations for debugging
DOCLING_GEN_VIZ=1 uv run pytest tests/cvat_to_docling/test_regression.py -v
```

## Structure

```
tests/cvat_to_docling/
├── README.md              # This file
├── test_regression.py    # Test suite
└── fixtures/             # Test fixtures
    ├── 002a_table_order/
    │   ├── input.xml         # Single-image CVAT XML
    │   ├── input.pdf         # Source PDF
    │   ├── metadata.json     # Test metadata
    │   └── expected.json     # Expected DoclingDocument output
    ├── 007_rotation_test/
    │   ├── input.xml         # Single-image CVAT XML
    │   ├── input.png         # Page image
    │   ├── metadata.json     # Test metadata
    │   └── expected.json     # Expected DoclingDocument output
    └── ...
```

## Test Fixtures

Each fixture tests a specific aspect of the conversion:
- Rotation handling
- Table structure and reading order
- Key-value pair parsing
- Picture and caption assignment
- Complex nested layouts

The test automatically detects input type (checks for `input.pdf` first, then `input.png`).

## Metadata Format

Each fixture requires `metadata.json`:

```json
{
  "test_id": "002",
  "name": "table_order",
  "description": "Tables should be in correct reading order",
  "observation_status": "correct",
  "source": {
    "image_identifier": "doc_xxxxx_page_000001.png"
  }
}
```

**Required fields:**
- `test_id`: Unique identifier
- `name`: Short name
- `description`: What this test covers
- `source.image_identifier`: Image name from CVAT XML
- `observation_status`: Controls test behavior (see below)

## Understanding Test Results

### Test Outcomes

- **PASSED** - Output matches expected.json (observation_status: "correct")
- **XFAIL** - Output matches known broken expected.json (observation_status: "broken")
- **FAILED** - One of:
  - observation_status is "correct" but output differs from expected.json
  - observation_status is "broken" but output differs from known broken expected.json

### observation_status

**"correct"** - Use for tests that should pass:
- Test passes when actual equals expected
- Test fails when they differ → either bug introduced or need to regenerate

**"broken"** - Use for documenting known bugs:
- Test xfails when actual equals expected (reproduces known broken behavior)
- Test fails when they differ (unexpected change in broken behavior)

## Running Tests

### Normal Mode
```bash
uv run pytest tests/cvat_to_docling/test_regression.py -v
```

### Regenerate Expected Outputs
After fixing bugs or updating conversion logic:
```bash
DOCLING_GEN_TEST_DATA=1 uv run pytest tests/cvat_to_docling/test_regression.py -v
```

### Generate Visualizations
```bash
# All tests
DOCLING_GEN_VIZ=1 uv run pytest tests/cvat_to_docling/test_regression.py -v

# Specific test
DOCLING_GEN_VIZ=1 uv run pytest tests/cvat_to_docling/test_regression.py -k "table" -v
```

Outputs are written to `scratch/cvat_regression_viz/`

## Workflows

### Investigating a Failure

```bash
# Generate visualization
DOCLING_GEN_VIZ=1 uv run pytest tests/cvat_to_docling/test_regression.py -k "002b_table_order" -v

# Inspect output
open scratch/cvat_regression_viz/002b_table_order/visualization_layout.html
```

### After Fixing a Bug

```bash
# Test the fix
uv run pytest tests/cvat_to_docling/test_regression.py -k "002b_table_order" -v

# Update observation_status to "correct" in metadata.json

# Regenerate expected output
DOCLING_GEN_TEST_DATA=1 uv run pytest tests/cvat_to_docling/test_regression.py -k "002b_table_order" -v

# Verify
uv run pytest tests/cvat_to_docling/test_regression.py -v
```

## Visualizations

Generated visualizations in `scratch/cvat_regression_viz/`:
- `output.html` - DoclingDocument HTML export
- `output.json` - Raw DoclingDocument JSON
- `visualization_layout.html` - Layout with reading order overlay
- `visualization_key_value.html` - Key-value pairs (if present)

Clean up:
```bash
rm -rf scratch/cvat_regression_viz/
```
