from enum import Enum
from typing import List

from docling_core.types.doc import BoundingBox
from pydantic import BaseModel


class BenchMarkColumns(str, Enum):
    CONVERTER_TYPE = "converter_type"
    CONVERTER_VERSION = "converter_version"
    DOCLING_PIPELINE = "docling_pipeline"

    STATUS = "status"
    DOC_ID = "document_id"
    DOC_PATH = "document_filepath"
    DOC_HASH = "document_filehash"

    GROUNDTRUTH = "GroundTruthDocument"
    PREDICTION = "PredictedDocument"
    ORIGINAL = "BinaryDocument"

    # PAGE_IMAGES = "page_images"
    # PICTURES = "pictures"

    GROUNDTRUTH_PAGE_IMAGES = "GroundTruthPageImages"
    GROUNDTRUTH_PICTURES = "GroundTruthPictures"
    GROUNDTRUTH_SEGMENTED_PAGES = "ground_truth_segmented_pages"

    PREDICTION_PAGE_IMAGES = "PredictionPageImages"
    PREDICTION_PICTURES = "PredictionPictures"
    PREDICTED_SEGMENTED_PAGES = "predicted_segmented_pages"

    MIMETYPE = "mimetype"
    TIMINGS = "timings"

    MODALITIES = "modalities"


class EvaluationModality(str, Enum):
    END2END = "end-to-end"
    LAYOUT = "layout"  # To compute maP on page-segmentation
    TABLE_STRUCTURE = "table_structure"  # to compute TEDS for tables
    DOCUMENT_STRUCTURE = (
        "document_structure"  # to compute edit distance between document structures
    )
    CODE_TRANSCRIPTION = "code_transcription"  # to compute BLEU between code sections
    MATH_TRANSCRIPTION = "math_transcription"  # to compute BLEU between latex formulas
    READING_ORDER = "reading_order"  # to compute the order
    MARKDOWN_TEXT = "markdown_text"  # to compute the text accuracy
    CAPTIONING = "captioning"  # to compute the accuracy of captions to table/figure
    BBOXES_TEXT = "bboxes_text"
    OCR = "ocr"
    KEY_VALUE = "key_value"
    QUESTION_ANSWERING = "question_answering"
    TIMINGS = "timings"


class BenchMarkNames(str, Enum):
    # End-to-End
    DPBENCH = "DPBench"
    DOCLING_DPBENCH = "DoclingDPBench"
    OMNIDOCBENCH = "OmniDocBench"
    WORDSCAPE = "WordScape"

    # Layout
    PUBLAYNET = "PubLayNet"
    DOCLAYNETV1 = "DocLayNetV1"
    DOCLAYNETV2 = "DocLayNetV2"
    FUNSD = "FUNSD"
    XFUND = "XFUND"

    # Table
    PUB1M = "Pub1M"
    PUBTABNET = "PubTabNet"
    FINTABNET = "FinTabNet"
    WIKITABNET = "WikiTabNet"

    DOCVQA = "DocVQA"

    # Formula
    # ???

    # OCR
    PIXPARSEIDL = "PixparseIdl"

    CVAT = "CVAT"

    PLAIN_FILES = "PlainFiles"


class ConverterTypes(str, Enum):
    DOCLING = "Docling"
    SMOL_DOCLING = "SmolDocling"


class PredictionFormats(str, Enum):
    DOCLING_DOCUMENT = "doclingdocument"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    YAML = "yaml"
    DOCTAGS = "doctags"


class PageToken(BaseModel):
    bbox: BoundingBox

    text: str
    id: int


class PageTokens(BaseModel):
    tokens: List[PageToken]

    height: float
    width: float


class ConsolidationFormats(str, Enum):
    EXCEL = "excel"


class PredictionProviderType(str, Enum):
    """Types of prediction providers available."""

    DOCLING = "Docling"
    PDF_DOCLING = "PDF_Docling"
    OCR_DOCLING = "OCR_Docling"
    MacOCR_DOCLING = "MacOCR_Docling"
    EasyOCR_DOCLING = "EasyOCR_Docling"

    TABLEFORMER = "TableFormer"
    FILE = "File"
    SMOLDOCLING = "SmolDocling"
    AWS = "AWS"
    AZURE = "Azure"
    GOOGLE = "Google"
