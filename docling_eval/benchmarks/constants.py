from enum import Enum


class BenchMarkColumns(str, Enum):
    DOCLING_VERSION = "docling_version"

    STATUS = "status"
    DOC_ID = "document_id"

    GROUNDTRUTH = "GroundTruthDocument"
    PREDICTION = "PredictedDocument"
    ORIGINAL = "BinaryDocument"

    PAGE_IMAGES = "page_images"
    PICTURES = "pictures"

    MIMETYPE = "mimetype"
    TIMINGS = "timings"


class EvaluationModality(str, Enum):
    END2END = "end-to-end"
    LAYOUT = "layout"
    TABLEFORMER = "tableformer"
    CODEFORMER = "codeformer"
    READING_ORDER = "reading_order"


class BenchMarkNames(str, Enum):

    # End-to-End
    DPBENCH = "DPBench"
    OMNIDOCBENCH = "OmniDocBench"
    WORDSCAPE = "WordScape"

    # Layout
    PUBLAYNET = "PubLayNet"
    DOCLAYNET = "DocLayNet"

    # Table
    PUB1M = "Pub1M"
    PUBTABNET = "PubTabNet"
    FINTABNET = "FinTabNet"
    WIKITABNET = "WikiTabNet"

    # Formula
    # ???

    # OCR
    # ???
