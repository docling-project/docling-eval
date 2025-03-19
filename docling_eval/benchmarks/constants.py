from enum import Enum


class BenchMarkColumns(str, Enum):
    DOCLING_VERSION = "docling_version"
    AZURE_VERSION = "azure_version"
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

    PREDICTION_PAGE_IMAGES = "PredictionPageImages"
    PREDICTION_PICTURES = "PredictionPictures"

    MIMETYPE = "mimetype"
    TIMINGS = "timings"

    MODALITIES = "modalities"


class EvaluationModality(str, Enum):
    END2END = "end-to-end"
    LAYOUT = "layout"  # To compute maP on page-segmentation
    TABLE_STRUCTURE = "table_structure"  # to compute TEDS for tables
    CODE_TRANSCRIPTION = "code_transcription"  # to compute BLEU between code sections
    MATH_TRANSCRIPTION = "math_transcription"  # to compute BLEU between latex formulas
    READING_ORDER = "reading_order"  # to compute the order
    MARKDOWN_TEXT = "markdown_text"  # to compute the text accuracy
    CAPTIONING = "captioning"  # to compute the accuracy of captions to table/figure
    BBOXES_TEXT = "bboxes_text"


class BenchMarkNames(str, Enum):

    # End-to-End
    DPBENCH = "DPBench"
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

    # Formula
    # ???

    # OCR
    # ???


class ConverterTypes(str, Enum):
    DOCLING = "Docling"
    SMOL_DOCLING = "SmolDocling"
