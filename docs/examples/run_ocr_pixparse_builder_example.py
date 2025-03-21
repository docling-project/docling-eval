import logging
import os
from pathlib import Path

from docling.datamodel.pipeline_options import OcrEngine

from docling_eval.evaluators.ocr.ocr_evaluator import OCREvaluator
from docling_eval_next.dataset_builders.pixparse_builder import (
    OCRBenchmarkDatasetBuilder,
)
from docling_eval_next.prediction_providers.hyperscalers import (
    PixparsePredictionProvider,
)
from docling_eval_next.utils.hyperscalers.utils import Hyperscaler, get_hyperscaler


def main():
    idir = Path("/Users/sami/Desktop/IBM/docling-eval/pixparse-idl")
    odir = Path("/Users/sami/Desktop/IBM/docling-eval/pixparse-idl/output")

    os.makedirs(odir, exist_ok=True)

    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("AWS_ACCESS_KEY_ID", "")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    os.environ["GOOGLE_PROJECT_ID"] = os.environ.get("GOOGLE_PROJECT_ID", "")
    os.environ["GOOGLE_PROCESSOR_ID"] = os.environ.get("GOOGLE_PROCESSOR_ID", "")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS", ""
    )
    os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"] = os.environ.get(
        "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", ""
    )
    os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"] = os.environ.get(
        "AZURE_DOCUMENT_INTELLIGENCE_KEY", ""
    )

    logging.info("Create the converted PIXPARSE dataset")

    provider = PixparsePredictionProvider()

    easyocr_dataset = OCRBenchmarkDatasetBuilder(
        name="easyocr",
        prediction_provider=provider,
        dataset_local_path=idir,
        # target=os.path.join(odir, "easyocr"),
        target=odir / "easyocr",
        ocr_engine=OcrEngine.EASYOCR,
        reprocess=False,
    )
    easyocr_dataset.save_to_disk()

    aws_dataset = OCRBenchmarkDatasetBuilder(
        name="aws",
        prediction_provider=provider,
        dataset_local_path=idir,
        target=odir / "aws",
        hyperscaler=Hyperscaler.AWS,
        reprocess=False,
    )

    aws_dataset.save_to_disk()

    azure_dataset = OCRBenchmarkDatasetBuilder(
        name="azure",
        prediction_provider=provider,
        dataset_local_path=idir,
        target=odir / "azure",
        hyperscaler=Hyperscaler.AZURE,
        reprocess=False,
    )

    azure_dataset.save_to_disk()

    google_dataset = OCRBenchmarkDatasetBuilder(
        name="google",
        prediction_provider=provider,
        dataset_local_path=idir,
        target=odir / "google",
        hyperscaler=Hyperscaler.GOOGLE,
        reprocess=False,
    )

    google_dataset.save_to_disk()

    hyperscaler = get_hyperscaler("wdu")
    wdu_dataset = OCRBenchmarkDatasetBuilder(
        name="wdu",
        prediction_provider=provider,
        dataset_local_path=idir,
        target=odir / "wdu",
        hyperscaler=hyperscaler,
        reprocess=False,
    )

    wdu_dataset.save_to_disk()

    logging.info("Evaluate the OCR for the Pixparse dataset")
    evaluator = OCREvaluator()
    dataset_path = Path("/Users/sami/Desktop/IBM/docling-eval/pixparse-idl/output")
    output_path = Path("/Users/sami/Desktop/IBM/docling-eval/pixparse-idl")
    evaluation_results = evaluator(dataset_path, output_path)
    logging.info(f"Completed evaluation. Results saved to {output_path}")


if __name__ == "__main__":
    main()
