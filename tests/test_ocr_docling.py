import logging
import multiprocessing
import os
from pathlib import Path

import pytest
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    OcrOptions,
    PdfPipelineOptions,
)
from docling.document_converter import PdfFormatOption
from docling.models.factories import get_ocr_factory

from docling_eval.cli.main import evaluate, visualize
from docling_eval.datamodels.types import BenchMarkNames, EvaluationModality
from docling_eval.dataset_builders.xfund_builder import XFUNDDatasetBuilder
from docling_eval.prediction_providers.docling_provider import DoclingPredictionProvider

IS_CI = os.getenv("RUN_IN_CI") == "1"

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_run_xfund_builder_docling():
    target_path = Path(f"./scratch/{BenchMarkNames.XFUND.value}_docling/")

    ocr_factory = get_ocr_factory()
    ocr_options: OcrOptions = ocr_factory.create_options(
        kind="easyocr",
    )
    ocr_options.use_gpu = False

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

    prediction_provider = DoclingPredictionProvider(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
        },
        do_visualization=False,
        ignore_missing_predictions=True,
    )

    dataset = XFUNDDatasetBuilder(
        dataset_source=target_path / "input_dataset",
        target=target_path / "gt_dataset",
        end_index=1,
    )

    dataset.retrieve_input_dataset()
    dataset.save_to_disk()

    prediction_provider.create_prediction_dataset(
        name=dataset.name,
        gt_dataset_dir=target_path / "gt_dataset",
        target_dataset_dir=target_path / "eval_dataset",
        split="val",
        begin_index=0,
        end_index=2,
        chunk_size=5,
    )

    evaluate(
        modality=EvaluationModality.OCR,
        benchmark=BenchMarkNames.XFUND,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.OCR.value,
        split="val",
    )

    visualize(
        modality=EvaluationModality.OCR,
        benchmark=BenchMarkNames.XFUND,
        idir=target_path / "eval_dataset",
        odir=target_path / "evaluations" / EvaluationModality.OCR.value,
        split="val",
    )
