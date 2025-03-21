import logging
import os
from typing import Any, Dict, Optional, Tuple

import boto3  # type: ignore
from google.cloud import documentai
from google.protobuf.json_format import MessageToDict

from docling_eval_next.utils.hyperscalers.utils import Hyperscaler


def initialize_textract_client():
    """Initializes and returns the AWS Textract client."""
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", "us-east-1")

    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError(
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in environment variables."
        )

    return boto3.client(
        "textract",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
    )


def initialize_google_doc_ai_client() -> Tuple[Optional[Any], Optional[str]]:
    """Initializes and returns the Google Document AI client and processor name."""
    if not hasattr(documentai, "DocumentProcessorServiceClient"):
        logging.info(
            "Warning: google-cloud-documentai library not installed. Google Doc AI functionality will be disabled."
        )
        return None, None

    google_project_id = os.getenv("GOOGLE_PROJECT_ID")
    google_location = os.getenv("GOOGLE_LOCATION", "us")
    google_processor_id = os.getenv("GOOGLE_PROCESSOR_ID")

    if not google_project_id or not google_processor_id:
        raise ValueError(
            "GOOGLE_PROJECT_ID and GOOGLE_PROCESSOR_ID must be set in environment variables."
        )

    google_client = documentai.DocumentProcessorServiceClient()
    google_processor_name = f"projects/{google_project_id}/locations/{google_location}/processors/{google_processor_id}"
    return google_client, google_processor_name


def initialize_azure_document_intelligence_client() -> Optional[Any]:
    """Initializes and returns the Azure Document Intelligence client."""
    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential
    except ImportError:
        logging.info(
            "Warning: azure-ai-documentintelligence library not installed. Azure functionality will be disabled."
        )
        return None

    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    if not endpoint or not key:
        raise ValueError(
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY must be set in environment variables."
        )

    return DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))


def process_with_textract(textract_client, image_content: bytes) -> Dict:
    """Processes image content with AWS Textract."""
    if textract_client is None:
        logging.info("Textract client not initialized. Skipping processing.")
        return {}
    try:
        response = textract_client.analyze_document(
            Document={"Bytes": image_content}, FeatureTypes=["TABLES", "FORMS"]
        )
        return response
    except Exception as e:
        logging.info(f"Error processing with Textract: {e}")
        return {}


def process_with_google(
    google_client,
    google_processor_name: str,
    image_content: bytes,
    mime_type: str = "image/tiff",
) -> Optional[Dict]:
    """Processes image content with Google Document AI."""
    if google_client is None or google_processor_name is None:
        logging.info("Google Doc AI client not initialized. Skipping processing.")
        return None

    try:
        raw_document = documentai.RawDocument(
            content=image_content, mime_type=mime_type
        )
        request = documentai.ProcessRequest(
            name=google_processor_name, raw_document=raw_document
        )
        response = google_client.process_document(request=request)
        return MessageToDict(response.document._pb)
    except Exception as e:
        logging.info(f"Error processing with Google Doc AI: {e}")
        return None


def process_with_azure(azure_client, image_content: bytes) -> Optional[Dict]:
    """Processes image content with Azure Document Intelligence."""
    if azure_client is None:
        logging.info(
            "Azure Document Intelligence client not initialized. Skipping processing."
        )
        return None
    try:
        poller = azure_client.begin_analyze_document("prebuilt-read", image_content)
        result = poller.result()
        return result.as_dict()
    except Exception as e:
        logging.info(f"Error processing with Azure Document Intelligence: {e}")
        return None


def initialize_hyperscaler_client(hyperscaler: Hyperscaler) -> Dict[str, Any]:
    """Initialize a specific hyperscaler client."""
    clients = {}

    if hyperscaler == Hyperscaler.AWS:
        clients["textract"] = initialize_textract_client()
    elif hyperscaler == Hyperscaler.GOOGLE:
        clients["google"], clients["google_processor_name"] = (
            initialize_google_doc_ai_client()
        )
    elif hyperscaler == Hyperscaler.AZURE:
        clients["azure"] = initialize_azure_document_intelligence_client()

    return clients
