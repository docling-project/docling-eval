import base64
import io
import logging
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd  # import-untyped
from datasets import Dataset, DatasetInfo, Features, concatenate_datasets
from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.document import DoclingDocument, PageItem
from PIL import Image  # as PILImage
from pydantic import AnyUrl

from docling_eval.docling.constants import HTML_DEFAULT_HEAD


def docling_version() -> str:
    return version("docling")  # may raise PackageNotFoundError


def create_styled_html(body: str) -> str:

    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        HTML_DEFAULT_HEAD,
        "<body>",
        body,
        "</body></html>",
    ]
    return "".join(html_lines)


def get_binary(file_path: Path):
    """Read binary document into buffer."""
    with open(file_path, "rb") as f:
        return f.read()


def map_to_records(item: Dict):
    """Map cells from pdf-parser into a records."""
    header = item["header"]
    data = item["data"]

    # Create a DataFrame
    df = pd.DataFrame(data, columns=header)
    return df.to_dict(orient="records")


def from_pil_to_base64(img: Image.Image) -> str:
    # Convert the image to a base64 str
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")  # Specify the format (e.g., JPEG, PNG, etc.)
    image_bytes = buffered.getvalue()

    # Encode the bytes to a Base64 string
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return image_base64


def from_pil_to_base64uri(img: Image.Image) -> AnyUrl:

    image_base64 = from_pil_to_base64(img)
    uri = AnyUrl(f"data:image/png;base64,{image_base64}")

    return uri


def to_base64(item: Dict[str, Any]) -> str:
    image_bytes = item["bytes"]

    # Wrap the bytes in a BytesIO object
    image_stream = BytesIO(image_bytes)

    # Open the image using PIL
    image = Image.open(image_stream)

    # Convert the image to a bytes object
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")  # Specify the format (e.g., JPEG, PNG, etc.)
    image_bytes = buffered.getvalue()

    # Encode the bytes to a Base64 string
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return image_base64


def to_pil(uri):

    base64_string = str(uri)
    base64_string = base64_string.split(",")[1]

    # Step 1: Decode the Base64 string
    image_data = base64.b64decode(base64_string)

    # Step 2: Open the image using Pillow
    image = Image.open(BytesIO(image_data))

    return image


def extract_images(
    document: DoclingDocument,
    pictures_column: str,
    page_images_column: str,
):

    pictures = []
    page_images = []

    # Save page images
    for img_no, picture in enumerate(document.pictures):
        if picture.image is not None:
            # img = picture.image.pil_image
            # pictures.append(to_pil(picture.image.uri))
            pictures.append(picture.image.pil_image)
            picture.image.uri = Path(f"{pictures_column}/{img_no}")

    # Save page images
    for page_no, page in document.pages.items():
        if page.image is not None:
            # img = page.image.pil_image
            # img.show()
            page_images.append(page.image.pil_image)
            page.image.uri = Path(f"{page_images_column}/{page_no}")

    return document, pictures, page_images


def insert_images(
    document: DoclingDocument,
    pictures: List[Dict[str, Any]],
    page_images: List[Dict[str, Any]],
):

    # Save page images
    for pic_no, picture in enumerate(document.pictures):
        if picture.image is not None:
            if pic_no < len(pictures):
                b64 = to_base64(pictures[pic_no])

                image_ref = document.pictures[pic_no].image
                if image_ref is not None:
                    image_ref.uri = AnyUrl(f"data:image/png;base64,{b64}")
                    document.pictures[pic_no].image = image_ref
                else:
                    logging.warning(f"image-ref is none for picture {pic_no}")

                """
                if document.pictures[pic_no].image is not None:                    
                    document.pictures[pic_no].image.uri = AnyUrl(
                        f"data:image/png;base64,{b64}"
                    )
                else:
                    logging.warning(f"image-ref is none for picture {pic_no}")
                """

            """
            else:
                document.pictures[pic_no].image.uri = None
                # logging.warning(f"inconsistent number of images in the document ({len(pictures)} != {len(document.pictures)})")
            """

    # Save page images
    for page_no, page in document.pages.items():
        if page.image is not None:
            # print(f"inserting image to page: {page_no}")
            b64 = to_base64(page_images[page_no - 1])

            image_ref = document.pages[page_no].image
            if image_ref is not None:
                image_ref.uri = AnyUrl(f"data:image/png;base64,{b64}")
                document.pages[page_no].image = image_ref

    return document


def save_shard_to_disk(
    items: List[Any],
    dataset_path: Path,
    thread_id: int = 0,
    shard_id: int = 0,
    features: Optional[Features] = None,
    shard_format: str = "parquet",
):
    """Save shard of to disk."""

    batch = Dataset.from_list(items)  # , features=features)

    output_file = dataset_path / f"shard_{thread_id:06}_{shard_id:06}.{shard_format}"
    logging.info(f"Saved shard {shard_id} to {output_file} with {len(items)} documents")

    if shard_format == "json":
        batch.to_json(output_file)

    elif shard_format == "parquet":
        batch.to_parquet(output_file)

    else:
        raise ValueError(f"Unsupported shard_format: {shard_format}")

    shard_id += 1

    return shard_id, [], 0


def load_shard_from_disk(parquet_path: Path):
    """
    Load Parquet shard into a single Hugging Face dataset.
    """
    return Dataset.from_parquet(parquet_path)


def load_shards_from_disk(dataset_path: Path):
    """
    Load all Parquet shards from a directory into a single Hugging Face dataset.
    """
    parquet_files = sorted(list(Path(dataset_path).glob("*.parquet")))
    datasets = [Dataset.from_parquet(str(pfile)) for pfile in parquet_files]
    return concatenate_datasets(datasets)


def generate_dataset_info(
    output_dir: Path,
    features: Features,
    description: str = "",
    license: str = "CC-BY 4.0",
    version="1.0.0",
):
    """
    Generate dataset_info.json manually for a dataset.
    """
    dataset_info = DatasetInfo(
        description=description,
        features=features,
        license="CC-BY 4.0",
        version="1.0.0",
    )
    dataset_info.save_to_disk(str(output_dir))


def crop_bounding_box(page_image: Image.Image, page: PageItem, bbox: BoundingBox):
    """
    Crop a bounding box from a PIL image.

    :param img: PIL Image object
    :param l: Left coordinate
    :param t: Top coordinate (from bottom-left origin)
    :param r: Right coordinate
    :param b: Bottom coordinate (from bottom-left origin)
    :return: Cropped PIL Image
    """
    width = float(page.size.width)
    height = float(page.size.height)

    img_width = float(page_image.width)
    img_height = float(page_image.height)

    scale_x = img_width / width
    scale_y = img_height / height

    bbox = bbox.to_top_left_origin(page.size.height)

    l = bbox.l * scale_x
    t = bbox.t * scale_y
    r = bbox.r * scale_x
    b = bbox.b * scale_y

    # Crop using the converted coordinates
    cropped_image = page_image.crop((l, t, r, b))

    return cropped_image
