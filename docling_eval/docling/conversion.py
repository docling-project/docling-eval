import logging
import warnings
from pathlib import Path
from typing import List, Optional

from docling.cli.main import OcrEngine
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrMacOptions,
    OcrOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    TableFormerMode,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption

warnings.filterwarnings(action="ignore", category=UserWarning, module="pydantic|torch")
warnings.filterwarnings(action="ignore", category=FutureWarning, module="easyocr")

# Set logging level for the 'docling' package
logging.getLogger("docling").setLevel(logging.WARNING)


def create_converter(
    page_image_scale: float = 2.0,
    do_ocr: bool = False,
    ocr_lang: List[str] = ["en"],
    ocr_engine: OcrEngine = OcrEngine.EASYOCR,
    timings: bool = True,
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

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Enable the profiling to measure the time spent
    settings.debug.profile_pipeline_timings = timings

    return doc_converter


def create_image_converter(
    do_ocr: bool = False,
    ocr_lang: List[str] = ["en"],
    ocr_engine: OcrEngine = OcrEngine.EASYOCR,
    timings: bool = True,
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
    )

    pipeline_options.table_structure_options.do_cell_matching = True  # do_cell_matching
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    # pipeline_options.images_scale = page_image_scale
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Enable the profiling to measure the time spent
    settings.debug.profile_pipeline_timings = timings

    return doc_converter
