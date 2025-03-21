from pathlib import Path
from typing import List, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrEngine,
    OcrMacOptions,
    OcrOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    TableFormerMode,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from docling.document_converter import PdfFormatOption

from docling_eval.benchmarks.constants import BenchMarkNames, EvaluationModality
from docling_eval.cli.main import evaluate
from docling_eval_next.datamodels.dataset_record import DatasetRecord
from docling_eval_next.dataset_builders.doclaynet_v1_builder import (
    DocLayNetV1DatasetBuilder,
)
from docling_eval_next.dataset_builders.dpbench_builder import DPBenchDatasetBuilder
from docling_eval_next.dataset_builders.funsd_builder import FUNSDDatasetBuilder
from docling_eval_next.dataset_builders.omnidocbench_builder import (
    OmniDocBenchDatasetBuilder,
)
from docling_eval_next.dataset_builders.xfund_builder import XFUNDDatasetBuilder
from docling_eval_next.dataset_builders.omnidocbench_builder import (
    OmniDocBenchDatasetBuilder,
)
from docling_eval_next.prediction_providers.prediction_provider import (
    DoclingPredictionProvider,
    NullPredictionProvider,
    TableFormerPredictionProvider,
)


def create_docling_prediction_provider(
    page_image_scale: float = 2.0,
    do_ocr: bool = False,
    ocr_lang: List[str] = ["en"],
    ocr_engine: OcrEngine = OcrEngine.EASYOCR,
    artifacts_path: Optional[Path] = None,
):

    force_ocr: bool = True

    if ocr_engine == OcrEngine.EASYOCR:
        ocr_options: OcrOptions = EasyOcrOptions(force_full_page_ocr=force_ocr)
    elif ocr_engine == OcrEngine.TESSERACT_CLI:
        ocr_options = TesseractCliOcrOptions(force_full_page_ocr=force_ocr)
    elif ocr_engine == OcrEngine.TESSERACT:
        ocr_options = TesseractOcrOptions(force_full_page_ocr=force_ocr)
    elif ocr_engine == OcrEngine.OCRMAC:
        ocr_options = OcrMacOptions(force_full_page_ocr=force_ocr)
    elif ocr_engine == OcrEngine.RAPIDOCR:
        ocr_options = RapidOcrOptions(force_full_page_ocr=force_ocr)
    else:
        raise RuntimeError(f"Unexpected OCR engine type {ocr_engine}")

    if ocr_lang is not None:
        ocr_options.lang = ocr_lang

    pipeline_options = PdfPipelineOptions(
        do_ocr=do_ocr,
        ocr_options=EasyOcrOptions(force_full_page_ocr=force_ocr),
        do_table_structure=True,
        artifacts_path=artifacts_path,
    )

    pipeline_options.table_structure_options.do_cell_matching = True  # do_cell_matching
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    pipeline_options.images_scale = page_image_scale
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    return DoclingPredictionProvider(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def test_run_dpbench_e2e():
    target_path = Path("./scratch/dpbench-builer-test/")
    docling_provider = create_docling_prediction_provider(page_image_scale=2.0)

    dataset_layout = DPBenchDatasetBuilder(
        prediction_provider=docling_provider,
        target=target_path / "e2e",
    )

    dataset_layout.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_layout.save_to_disk(
        chunk_size=20, max_num_chunks=1
    )  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    evaluate(
        modality=EvaluationModality.LAYOUT,
        benchmark=BenchMarkNames.DPBENCH,
        idir=target_path / "e2e",
        odir=target_path / "e2e" / "layout",
    )


def test_run_omnidocbench_e2e():
    target_path = Path("./scratch/omnidocbench-builer-test/")
    docling_provider = create_docling_prediction_provider(page_image_scale=2.0)

    dataset_layout = OmniDocBenchDatasetBuilder(
        prediction_provider=docling_provider,
        target=target_path / "e2e",
    )

    dataset_layout.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_layout.save_to_disk(
        chunk_size=20, max_num_chunks=1
    )  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    evaluate(
        modality=EvaluationModality.LAYOUT,
        benchmark=BenchMarkNames.OMNIDOCBENCH,
        idir=target_path / "e2e",
        odir=target_path / "e2e" / "layout",
    )


def test_run_dpbench_tables():
    target_path = Path("./scratch/dpbench-builer-test/")
    tableformer_provider = TableFormerPredictionProvider()

    dataset_tables = DPBenchDatasetBuilder(
        prediction_provider=tableformer_provider,
        target=target_path / "tables",
    )

    dataset_tables.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_tables.save_to_disk(
        chunk_size=20, max_num_chunks=1
    )  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    evaluate(
        modality=EvaluationModality.TABLE_STRUCTURE,
        benchmark=BenchMarkNames.DPBENCH,
        idir=target_path / "tables",
        odir=target_path / "tables" / "tableformer",
    )


def test_run_omnidocbench_tables():
    target_path = Path("./scratch/omnidocbench-builer-test/")
    tableformer_provider = TableFormerPredictionProvider()

    dataset_tables = OmniDocBenchDatasetBuilder(
        prediction_provider=tableformer_provider,
        target=target_path / "tables",
    )

    dataset_tables.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_tables.save_to_disk(
        chunk_size=20, max_num_chunks=1
    )  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    evaluate(
        modality=EvaluationModality.TABLE_STRUCTURE,
        benchmark=BenchMarkNames.OMNIDOCBENCH,
        idir=target_path / "tables",
        odir=target_path / "tables" / "tableformer",
    )

def test_run_doclaynet_v1_e2e():
    target_path = Path("./scratch/doclaynet-v1-builder-test/")
    docling_provider = create_docling_prediction_provider(page_image_scale=2.0)

    dataset_layout = DocLayNetV1DatasetBuilder(
        prediction_provider=docling_provider,
        target=target_path / "e2e",
    )

    dataset_layout.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_layout.save_to_disk(
        chunk_size=20, max_num_chunks=1
    )  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    if False:
        evaluate(
            modality=EvaluationModality.LAYOUT,
            benchmark=BenchMarkNames.DOCLAYNETV1,
            idir=target_path / "e2e",
            odir=target_path / "e2e" / "layout",
        )


# def test_run_doclaynet_v2_e2e():
#     target_path = Path("./scratch/doclaynet-v2-builder-test/")
#     docling_provider = create_docling_prediction_provider(page_image_scale=2.0)
#
#     dataset_layout = DocLayNetV2DatasetBuilder(
#         dataset_path="",
#         prediction_provider=docling_provider,
#         target=target_path / "e2e",
#     )
#
#     dataset_layout.retrieve_input_dataset()  # fetches the source dataset from HF
#     dataset_layout.save_to_disk(
#         chunk_size=20, max_num_chunks=1
#     )  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.
#
#     if False:
#         evaluate(
#             modality=EvaluationModality.LAYOUT,
#             benchmark=BenchMarkNames.DOCLAYNETV2,
#             idir=target_path / "e2e",
#             odir=target_path / "e2e" / "layout",
#         )


def test_run_funsd():
    target_path = Path("./scratch/funsd-builder-test/")

    dataset_layout = FUNSDDatasetBuilder(
        dataset_source=target_path / "input_dataset",
        prediction_provider=NullPredictionProvider(),
        target=target_path / "e2e",
    )

    dataset_layout.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_layout.save_to_disk(
        chunk_size=20, max_num_chunks=1
    )  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.


def test_run_xfund():
    target_path = Path("./scratch/xfund-builder-test/")

    dataset_layout = XFUNDDatasetBuilder(
        dataset_source=target_path / "input_dataset",
        prediction_provider=NullPredictionProvider(),
        target=target_path / "e2e",
    )

    dataset_layout.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset_layout.save_to_disk(
        chunk_size=20, max_num_chunks=1
    )  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.
