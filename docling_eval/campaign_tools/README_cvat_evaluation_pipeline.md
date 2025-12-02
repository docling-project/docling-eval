# CVAT Evaluation Pipeline Utility

A flexible pipeline for evaluating CVAT annotations that converts CVAT XML files to DoclingDocument format and runs layout and document structure evaluations.

## Features

- Convert CVAT XML annotations to DoclingDocument JSON format
- Create ground truth datasets from CVAT annotations
- Create prediction datasets for evaluation
- Run layout and document structure evaluations
- Support for step-by-step or end-to-end execution
- Configurable evaluation modalities

## Requirements

The utility requires the following inputs:
1. **CVAT Dataset Root**: The `cvat_dataset_preannotated` directory containing `cvat_overview.json`, `cvat_tasks/`, and accompanying assets
2. **Output Directory**: Directory where all pipeline outputs will be saved

The pipeline expects the standard DocLayNet task naming convention, where ground-truth annotations live under files named `task_{xx}_set_A.xml` and predictions under `task_{xx}_set_B.xml`.

## Usage

### Command Line Interface

```bash
python cvat_evaluation_pipeline.py <cvat_dataset_preannotated> <output_dir> [OPTIONS]
```

### Required Arguments

- `cvat_dataset_preannotated`: Root directory of the CVAT export to convert
- `output_dir`: Output directory for pipeline results

### Optional Arguments

- `--tasks-root PATH`: Alternate directory whose `cvat_tasks/` contains the annotation XMLs
- `--step {gt,pred,eval,full}`: Pipeline step to run (default: full)
- `--modalities {layout,document_structure}`: Evaluation modalities to run (default: both)
- `--strict`: Strict mode - require all images to have annotations in XML files (default: allow partial annotation batches)
- `--verbose, -v`: Enable verbose logging

## Examples

### 1. Run Full Pipeline

Convert both ground truth and prediction CVAT XMLs, create datasets, and run evaluations:

```bash
python cvat_evaluation_pipeline.py \
    /path/to/cvat_dataset_preannotated \
    /path/to/output
```

### 2. Run Step by Step

**Step 1: Create Ground Truth Dataset**
```bash
python cvat_evaluation_pipeline.py \
    /path/to/cvat_dataset_preannotated \
    /path/to/output \
    --step gt
```

**Step 2: Create Prediction Dataset**
```bash
python cvat_evaluation_pipeline.py \
    /path/to/cvat_dataset_preannotated \
    /path/to/output \
    --step pred
```

**Step 3: Run Evaluation**
```bash
python cvat_evaluation_pipeline.py \
    /path/to/cvat_dataset_preannotated \
    /path/to/output \
    --step eval
```

### 3. Run Specific Evaluation Modalities

Run only layout evaluation:
```bash
python cvat_evaluation_pipeline.py \
    /path/to/cvat_dataset_preannotated \
    /path/to/output \
    --modalities layout
```

Run only document structure evaluation:
```bash
python cvat_evaluation_pipeline.py \
    /path/to/cvat_dataset_preannotated \
    /path/to/output \
    --modalities document_structure
```

### 5. Strict Mode

By default, the pipeline allows partial annotation batches where not all images need to have annotations in the XML file. This is useful when you have a large set of images but only a subset has been annotated.

To enforce that ALL images must have annotations, use the `--strict` flag:

```bash
python cvat_evaluation_pipeline.py \
    /path/to/cvat_dataset_preannotated \
    /path/to/output \
    --strict
```

### Overlaying Alternate Task Directories

When annotation XMLs live in a parallel structure (for example, predictions exported separately), point the pipeline to the primary `cvat_dataset_preannotated` root for assets and pass the override:

```bash
python cvat_evaluation_pipeline.py \
    /path/to/cvat_dataset_preannotated \
    /path/to/output \
    --tasks-root /path/to/alternate_annotations
```

The override accepts either a directory that already *is* `cvat_tasks/` or a parent folder containing it.

In strict mode:
- The pipeline will fail with an error if any image lacks annotations
- Useful for validating complete annotation batches
- Helps catch missing annotations early in the process

## Output Structure

The pipeline creates the following directory structure in the output directory:

```
output_dir/
├── ground_truth_json/          # Ground truth DoclingDocument JSON files
│   ├── gt_image1.json
│   └── gt_image2.json
├── predictions_json/           # Prediction DoclingDocument JSON files
│   ├── pred_image1.json
│   └── pred_image2.json
├── gt_dataset/                # Ground truth dataset
│   ├── test/
│   └── visualizations/
├── eval_dataset/              # Evaluation dataset
│   ├── test/
│   └── visualizations/
└── evaluation_results/        # Evaluation results
    ├── layout_evaluation/
    └── document_structure_evaluation/
```

## Pipeline Steps Explained

### Step 1: Ground Truth Dataset Creation
- Converts ground truth CVAT XML to DoclingDocument JSON format
- Creates a ground truth dataset using FileDatasetBuilder
- Generates visualizations for quality inspection

### Step 2: Prediction Dataset Creation
- Converts prediction CVAT XML to DoclingDocument JSON format
- Creates a prediction dataset using FilePredictionProvider
- Links predictions to the ground truth dataset for evaluation

### Step 3: Evaluation
- Runs layout evaluation (mean Average Precision metrics)
- Runs document structure evaluation (edit distance metrics)
- Saves detailed evaluation results and visualizations

## Error Handling

The utility includes comprehensive error handling:
- Validates input paths and file existence
- Provides clear error messages for missing requirements
- Continues processing other files if individual conversions fail
- Logs warnings for failed conversions without stopping the pipeline

## Logging

The utility provides detailed logging with timestamps:
- INFO level: Progress updates and results
- WARNING level: Non-critical issues (e.g., failed conversions)
- ERROR level: Critical errors that stop execution
- Use `--verbose` flag for DEBUG level logging

## Integration with Existing Codebase

This utility is designed to work with the existing docling-eval framework and uses:
- `docling_eval.cvat_tools.cvat_to_docling` for CVAT conversion
- `docling_eval.dataset_builders.file_dataset_builder` for dataset creation
- `docling_eval.prediction_providers.file_provider` for prediction datasets
- `docling_eval.cli.main.evaluate` for running evaluations

## Tips for Best Results

1. **Image Naming**: Ensure PNG files have consistent naming that matches the CVAT annotations
2. **XML Validation**: Verify that both ground truth and prediction XML files are valid CVAT exports
3. **Output Space**: Ensure sufficient disk space for intermediate JSON files and datasets
4. **Step-by-Step**: For large datasets, consider running steps separately for better resource management
5. **Visualization**: Check the generated visualizations to verify conversion quality 
