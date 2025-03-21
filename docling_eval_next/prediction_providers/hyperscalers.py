import json
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from docling.cli.main import OcrEngine
from docling.document_converter import DocumentConverter
from docling_core.types import DoclingDocument
from docling_core.types.io import DocumentStream

from docling_eval.benchmarks.utils import docling_version
from docling_eval.converters.conversion import create_image_docling_converter
from docling_eval.converters.hyperscalers import (
    convert_azure_output_to_docling,
    convert_google_output_to_docling,
    convert_textract_output_to_docling,
)
from docling_eval_next.prediction_providers.prediction_provider import (
    BasePredictionProvider,
)
from docling_eval_next.utils.hyperscalers.hyperscaler_clients import (
    initialize_hyperscaler_client,
    process_with_azure,
    process_with_google,
    process_with_textract,
)
from docling_eval_next.utils.hyperscalers.utils import (
    CustomHyperscaler,
    Hyperscaler,
    check_service_env_vars,
    read_image_content,
    write_json_output,
)


class PixparsePredictionProvider(BasePredictionProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if kwargs is not None:
            if "format_options" in kwargs:
                self.doc_converter = DocumentConverter(
                    format_options=kwargs["format_options"]
                )
            else:
                self.doc_converter = DocumentConverter()

    def process_with_service(
        self,
        image_file: Path,
        doc_id: str,
        input_dir: Path,
        service: Union[Hyperscaler, CustomHyperscaler, OcrEngine],
    ) -> DoclingDocument:
        """Process an image with a specific service and return standardized record."""
        if service is None:
            raise ValueError("Service cannot be None")
        service_name = service.value

        if not check_service_env_vars(service):
            logging.error(
                f"Cannot process with {service.name} due to missing environment variables"
            )
            raise EnvironmentError(
                f"Missing required environment variables for {service.name}"
            )

        image_content_bytes = read_image_content(image_file)
        converted_result = None
        if isinstance(service, Hyperscaler):
            clients = initialize_hyperscaler_client(service)
            output_raw_path = input_dir / doc_id / f"{service_name}_raw.json"

            if service == Hyperscaler.AWS:
                aws_result: Optional[dict] = process_with_textract(
                    clients["textract"], image_content_bytes
                )
                if aws_result is not None:
                    write_json_output(output_raw_path, aws_result)
                    converted_result = convert_textract_output_to_docling(
                        aws_result, image_file
                    )
                else:
                    logging.warning("AWS Textract processing returned None result")
                    return {}  # type: ignore
            elif service == Hyperscaler.GOOGLE:
                google_result: Optional[dict] = process_with_google(
                    clients["google"],
                    clients["google_processor_name"],
                    image_content_bytes,
                )
                if google_result is not None:
                    write_json_output(output_raw_path, google_result)
                    converted_result = convert_google_output_to_docling(
                        google_result, image_file
                    )
                else:
                    logging.warning("Google processing returned None result")
                    return {}  # type: ignore
            elif service == Hyperscaler.AZURE:
                azure_result: Optional[dict] = process_with_azure(
                    clients["azure"], image_content_bytes
                )
                if azure_result is not None:
                    write_json_output(output_raw_path, azure_result)
                    converted_result = convert_azure_output_to_docling(
                        azure_result, image_file
                    )
                else:
                    logging.warning("Azure processing returned None result")
                    return {}  # type: ignore

        elif isinstance(service, OcrEngine):
            docling_ocr_doc_converter = create_image_docling_converter(
                do_ocr=True, ocr_engine=service
            )
            converted_result = docling_ocr_doc_converter.convert(image_file).document

        # print(converted_result)
        converted_results_dir = input_dir / doc_id / "normalized"
        converted_results_dir.mkdir(parents=True, exist_ok=True)
        converted_path = converted_results_dir / f"{doc_id}_{service_name}.docling.json"

        if converted_result is not None:
            with open(converted_path, "w") as f:
                json.dump(converted_result.export_to_dict(), f, indent=2)
        else:
            logging.error("Converted result is None, cannot export to JSON")
            raise ValueError("Converted result is None, export failed")

        pred_doc = converted_result  # Already in docling document format.

        if pred_doc is None:
            raise ValueError(
                "Prediction document is None, cannot return a valid DoclingDocument"
            )
        return pred_doc

    def predict(
        self,
        gt_doc: DoclingDocument,
        stream: Optional[DocumentStream] = None,
        **extra_args,
    ) -> DoclingDocument:
        reprocess = extra_args.get("reprocess", False)
        image_file = extra_args.get("image_file", "")
        doc_id = extra_args.get("doc_id", "")
        input_dir = extra_args.get("input_dir", "")
        output_dir = extra_args.get("output_dir")
        service = extra_args.get("service")

        if service is None:
            raise ValueError("Service cannot be None")
        service_name = service.value

        pred_doc: Optional[DoclingDocument] = None
        if reprocess:
            pred_doc = self.process_with_service(image_file, doc_id, input_dir, service)
        else:
            # load existing converted or raw result if available
            converted_path = (
                input_dir
                / doc_id
                / "normalized"
                / f"{doc_id}_{service_name}.docling.json"
            )

            if converted_path.exists():
                pred_doc = DoclingDocument.load_from_json(converted_path)
            else:
                logging.info(
                    f"No converted result found for {doc_id} with {service_name}, loading raw files"
                )
                output_raw_path = input_dir / doc_id / f"{service_name}_raw.json"
                with open(output_raw_path, "r") as f:
                    result = json.load(f)

                converters: Dict[
                    Hyperscaler, Callable[[Any, Path], DoclingDocument]
                ] = {
                    Hyperscaler.AZURE: convert_azure_output_to_docling,
                    Hyperscaler.AWS: convert_textract_output_to_docling,
                    Hyperscaler.GOOGLE: convert_google_output_to_docling,
                }

                pred_doc = converters.get(service, lambda *args: None)(  # type: ignore
                    result, image_file
                )
                if pred_doc:
                    converted_results_dir = input_dir / doc_id / "normalized"
                    converted_results_dir.mkdir(parents=True, exist_ok=True)

                    converted_path = (
                        converted_results_dir / f"{doc_id}_{service_name}.docling.json"
                    )

                    with open(converted_path, "w") as f:
                        json.dump(pred_doc.export_to_dict(), f, indent=2)

        if pred_doc is None:
            raise ValueError(
                "Prediction document is None, cannot return a valid DoclingDocument"
            )
        return pred_doc

    def info(self) -> Dict:
        return {"asset": "Docling", "version": docling_version()}
