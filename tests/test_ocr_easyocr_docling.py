import logging
import multiprocessing
import os
from pathlib import Path

import pytest
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
from docling.models.factories import get_ocr_factory

from docling_eval.cli.main import evaluate, visualize
from docling_eval.datamodels.types import BenchMarkNames, EvaluationModality
from docling_eval.dataset_builders.pixparse_builder import PixparseDatasetBuilder
from docling_eval.prediction_providers.docling_provider import DoclingPredictionProvider

IS_CI = bool(os.getenv("CI"))

_log = logging.getLogger(__name__)

logging.getLogger("docling").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)


def create_easyocr_provider():
    """Create EasyOCR prediction provider matching the working test pattern."""
    ocr_factory = get_ocr_factory()
    ocr_options = ocr_factory.create_options(kind="easyocr")
    ocr_options.use_gpu = True

    accelerator_options = AcceleratorOptions(
        num_threads=multiprocessing.cpu_count(),
    )

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=False,
        ocr_options=ocr_options,
        accelerator_options=accelerator_options,
        images_scale=2.0,
        generate_page_images=True,
        generate_picture_images=True,
        generate_parsed_pages=True,
    )

    # IMPORTANT: Use PdfFormatOption for BOTH PDF and IMAGE inputs
    return DoclingPredictionProvider(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
        },
        do_visualization=True,
        ignore_missing_predictions=True,
    )


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_pixparse_easyocr_docling():
    """Test EasyOCR with Docling pipeline on PixParse dataset."""
    target_path = Path(f"./scratch/{BenchMarkNames.PIXPARSEIDL.value}_easyocr_docling/")

    gt_dataset = target_path / "gt_dataset"
    eval_dataset = target_path / "eval_dataset"
    evaluations_path = target_path / "evaluations" / EvaluationModality.OCR.value

    # Build ground truth dataset
    dataset = PixparseDatasetBuilder(
        target=gt_dataset,
        begin_index=1,
        end_index=3,
    )
    dataset.retrieve_input_dataset()
    dataset.save_to_disk()

    # Create EasyOCR prediction provider
    easyocr_provider = create_easyocr_provider()

    _log.info(f"Dataset name {dataset.name}")
    _log.info(f"Created ground truth dataset at {gt_dataset}")
    _log.info(f"Creating evaluation dataset at {eval_dataset}")

    easyocr_provider.create_prediction_dataset(
        name=dataset.name,
        gt_dataset_dir=gt_dataset,
        target_dataset_dir=eval_dataset,
        chunk_size=50,
    )

    evaluate(
        modality=EvaluationModality.OCR,
        benchmark=BenchMarkNames.PIXPARSEIDL,
        idir=eval_dataset,
        odir=evaluations_path,
    )

    visualize(
        modality=EvaluationModality.OCR,
        benchmark=BenchMarkNames.PIXPARSEIDL,
        idir=eval_dataset,
        odir=evaluations_path,
    )

    _log.info(f"Completed evaluation. Results saved to {target_path}")
