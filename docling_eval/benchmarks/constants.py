from enum import Enum


class BenchMarkColumns(str, Enum):
    DOCLING_VERSION = "docling_version"
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


class EvaluationModality(str, Enum):
    END2END = "end-to-end"
    LAYOUT = "layout"
    TABLEFORMER = "tableformer"
    CODEFORMER = "codeformer"
    READING_ORDER = "reading_order"
    MARKDOWN_TEXT = "markdown_text"


class BenchMarkNames(str, Enum):

    # End-to-End
    DPBENCH = "DPBench"
    OMNIDOCBENCH = "OmniDocBench"
    WORDSCAPE = "WordScape"

    # Layout
    PUBLAYNET = "PubLayNet"
    DOCLAYNETV1 = "DocLayNetV1"
    DOCLAYNETV2 = "DocLayNetV2"

    # Table
    PUB1M = "Pub1M"
    PUBTABNET = "PubTabNet"
    FINTABNET = "FinTabNet"
    WIKITABNET = "WikiTabNet"

    # Formula
    # ???

    # OCR
    # ???
