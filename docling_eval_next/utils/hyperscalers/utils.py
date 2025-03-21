# utils.py

import base64
import io
import json
import logging
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

from docling.cli.main import OcrEngine
from PIL import Image


class Hyperscaler(str, Enum):
    AWS = "aws"
    AZURE = "azure"
    GOOGLE = "google"


class CustomHyperscaler:
    def __init__(self, value):
        self.value = value
        self.name = value.upper()

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"<CustomHyperscaler.{self.name}: '{self.value}'>"

    def __bool__(self):
        return True


def get_hyperscaler(value):
    try:
        return Hyperscaler(value)
    except ValueError:
        # Create a custom string-like object that mimics an enum member
        return CustomHyperscaler(value)


def get_required_env_vars_by_service() -> Dict[str, List[str]]:
    """Required environment variables for each service."""
    return {
        "AWS": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
        "GOOGLE": [
            "GOOGLE_PROJECT_ID",
            "GOOGLE_PROCESSOR_ID",
            "GOOGLE_APPLICATION_CREDENTIALS",
        ],
        "AZURE": [
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
            "AZURE_DOCUMENT_INTELLIGENCE_KEY",
        ],
    }


def check_required_env_vars(services: Optional[List[str]] = None) -> bool:
    """Check if all required environment variables are set for specified services"""
    env_vars_by_service = get_required_env_vars_by_service()

    if services is None:
        services = list(env_vars_by_service.keys())

    missing_vars = []
    missing_services = set()

    for service in services:
        if service not in env_vars_by_service:
            logging.warning(f"Unknown service: {service}")
            continue

        for var in env_vars_by_service[service]:
            if not os.environ.get(var):
                missing_vars.append(var)
                missing_services.add(service)

    if missing_vars:
        logging.error(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
        for service in missing_services:
            logging.error(
                f"Service {service} is missing required environment variables"
            )
        return False

    return True


def check_service_env_vars(
    service: Union[Hyperscaler, CustomHyperscaler, OcrEngine],
) -> bool:
    """Check if the required environment variables for a specific service are set."""
    if isinstance(service, OcrEngine):
        return True

    if isinstance(service, CustomHyperscaler):
        return True

    # For hyperscalers, check the appropriate environment variables
    if isinstance(service, Hyperscaler):
        service_name = service.name  # AWS, GOOGLE, or AZURE
        return check_required_env_vars([service_name])

    logging.error(f"Unknown service type: {type(service)}")
    return False


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle bytes and Pydantic models"""

    def default(self, obj):
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode("utf-8")
        # Check for both v1 and v2 Pydantic models.
        if hasattr(obj, "model_dump"):  # For Pydantic v2 models
            return obj.model_dump()
        if hasattr(obj, "dict"):  # For Pydantic v1 models
            return obj.dict()
        return super().default(obj)


def create_directories(dir_paths: List[Path]):
    """Creates directories if they don't exist."""
    for dir_path in dir_paths:
        dir_path.mkdir(parents=True, exist_ok=True)


def read_image_content(image_path: Path) -> bytes:
    """Reads image content in binary mode."""
    with open(image_path, "rb") as image:
        return image.read()


def write_json_output(output_path: Path, data: Dict):
    """Writes JSON data to a file."""
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def save_shard_to_disk(items: List[Dict], dataset_path: Path, shard_id: int) -> None:
    """Saves a shard of records to disk using custom JSON encoder."""
    shard_path = dataset_path / f"shard_{shard_id:05d}.jsonl"
    with open(shard_path, "w") as f:
        for item in items:
            json_str = json.dumps(item, cls=CustomJSONEncoder)
            f.write(json_str + "\n")


def save_hyperscaler_shard(
    items: List[Dict], output_dir: Path, hyperscaler: Hyperscaler, shard_id: int
) -> None:
    """Save a shard of hyperscaler predictions to disk."""
    service_name = hyperscaler.value
    shard_path = output_dir / f"{service_name}_shard_{shard_id:05d}.jsonl"

    with open(shard_path, "w") as f:
        for item in items:
            json_str = json.dumps(item, cls=CustomJSONEncoder)
            f.write(json_str + "\n")


def write_dataset_info(
    name: str, output_dir: Path, num_train_rows: int, num_test_rows: int
) -> None:
    """Writes dataset information to a file."""
    info = {
        "name": name,
        "num_train_rows": num_train_rows,
        "num_test_rows": num_test_rows,
        "creation_date": str(datetime.now()),
    }

    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)


def convert_image_to_bytes(image_path: Path) -> bytes:
    """Loads an image and converts it to bytes."""
    with Image.open(image_path) as img:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="TIFF")  # Or appropriate format
        return img_byte_arr.getvalue()
