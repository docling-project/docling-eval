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
from docling_eval_next.dataset_builders.dpbench_builder import DPBenchE2EDatasetBuilder
from docling_eval_next.prediction_providers.prediction_provider import (
    DoclingPredictionProvider,
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


def main():
    target_path = Path("./scratch/dpbench-builer-test/")
    provider = create_docling_prediction_provider(page_image_scale=2.0)

    dataset = DPBenchE2EDatasetBuilder(
        prediction_provider=provider,
        target=target_path,
    )

    dataset.retrieve_input_dataset()  # fetches the source dataset from HF
    dataset.save_to_disk()  # does all the job of iterating the dataset, making GT+prediction records, and saving them in shards as parquet.

    evaluate(
        modality=EvaluationModality.LAYOUT,
        benchmark=BenchMarkNames.DPBENCH,
        idir=target_path,
        odir=target_path / "layout",
    )


if __name__ == "__main__":
    main()
