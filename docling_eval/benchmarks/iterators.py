import argparse
import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, Generator, List, Set, Tuple

from PIL import Image as PILImage
from datasets import Dataset
from docling_core.types.doc.document import (
    DEFAULT_EXPORT_LABELS,
    ContentItem,
    DocItem,
    DoclingDocument,
    PictureItem,
    TableItem,
)
from docling_core.types.doc.labels import DocItemLabel

from docling_eval.docling.utils import create_styled_html, crop_bounding_box

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _to_pil(item: Dict):
    image_bytes = item["bytes"]

    # Wrap the bytes in a BytesIO object
    image_stream = BytesIO(image_bytes)

    # Open the image using PIL
    image = PILImage.open(image_stream)

    return image


def iterate_docitems_per_page(
    dataset_path: Path,
    column_name: str = "DoclingDocument",
    labels: Set[DocItemLabel] = DEFAULT_EXPORT_LABELS,
) -> Generator[Tuple[int, PILImage.Image, List[DocItem], DoclingDocument], None, None]:

    logging.info(f"reading {dataset_path}")

    # Load the dataset from parquet
    ds = Dataset.from_parquet(str(dataset_path))

    # Verify column names
    column_names = list(ds.features.keys())
    for _ in [column_name, "page_images"]:
        if _ not in column_names:
            raise ValueError(f"`{_}` not in {column_names}")

    # Iterate through rows
    for row in ds:

        page_images = row["page_images"]

        doc = DoclingDocument.model_validate(json.loads(row[column_name]))

        page_to_objects: Dict[int, List[DocItem]] = {}

        for item, level in doc.iterate_items():
            if isinstance(item, DocItem) and item.label in labels:
                for prov in item.prov:
                    if prov.page_no not in page_to_objects:
                        page_to_objects[prov.page_no] = [item]
                    else:
                        page_to_objects[prov.page_no].append(item)

        for page_no, items in page_to_objects.items():
            if 0 <= page_no - 1 and page_no - 1 < len(page_images):
                yield (page_no, _to_pil(page_images[page_no - 1]), items, doc)
            else:
                raise ValueError(
                    f"`{page_no - 1}` not in range of page_images ({len(page_images)})"
                )


def iterate_docitems(
    dataset_path: Path,
    column_name: str = "DoclingDocument",
    labels: Set[DocItemLabel] = DEFAULT_EXPORT_LABELS,
) -> Generator[
    Tuple[int, PILImage.Image, PILImage.Image, DocItem, DoclingDocument], None, None
]:

    logging.info(f"reading {dataset_path}")

    # Load the dataset from parquet
    ds = Dataset.from_parquet(str(dataset_path))

    # Verify column names
    column_names = list(ds.features.keys())
    for _ in [column_name, "page_images"]:
        if _ not in column_names:
            raise ValueError(f"`{_}` not in {column_names}")

    # Iterate through rows
    for row in ds:
        page_images = row["page_images"]

        doc = DoclingDocument.model_validate(json.loads(row[column_name]))

        page_to_objects: Dict[int, List[ContentItem]] = {}

        for item, level in doc.iterate_items():
            if isinstance(item, DocItem) and item.label in labels:
                for prov in item.prov:

                    if prov.page_no - 1 < 0 or len(page_images) <= prov.page_no - 1:
                        raise ValueError(
                            f"`{prov.page_no - 1}` not in range of page_images ({len(page_images)})"
                        )

                    page_image = _to_pil(page_images[prov.page_no - 1])
                    item_image = crop_bounding_box(
                        page_image=page_image,
                        page=doc.pages[prov.page_no],
                        bbox=prov.bbox,
                    )

                    yield (prov.page_no, page_image, item_image, item, doc)


def iterate_tables(
    dataset_path: Path, column_name: str = "DoclingDocument"
) -> Generator[
    Tuple[int, PILImage.Image, PILImage.Image, TableItem, DoclingDocument], None, None
]:
    for page_no, page_image, item_image, item, doc in iterate_docitems(
        dataset_path, column_name=column_name, labels={DocItemLabel.TABLE}
    ):
        if isinstance(item, TableItem):
            yield (page_no, page_image, item_image, item, doc)


def iterate_pictures(
    dataset_path: Path, column_name: str = "DoclingDocument"
) -> Generator[
    Tuple[int, PILImage.Image, PILImage.Image, PictureItem, DoclingDocument], None, None
]:
    for page_no, page_image, item_image, item, doc in iterate_docitems(
        dataset_path, column_name=column_name, labels={DocItemLabel.PICTURE}
    ):
        if isinstance(item, PictureItem):
            yield (page_no, page_image, item_image, item, doc)


def parse_arguments():
    """Parse arguments for directory parsing."""

    parser = argparse.ArgumentParser(description="Process shards from HF dataset.")
    parser.add_argument(
        "-i",
        "--input-shards",
        help="input file or directory with shards in parquet",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        help="output directory",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--mode",
        help="Mode of dataset",
        choices=["page", "table", "docitems"],
        required=False,
        default="table",
    )
    parser.add_argument(
        "-c",
        "--column-name",
        help="Name of the column where the document are stored",
        choices=[
            "DoclingDocument",
            "GroundTruthDocument",
            "PredictedDocument",
            "ParsedDocument",
        ],
        required=False,
        default="DoclingDocument",
    )
    args = parser.parse_args()

    return (
        Path(args.input_shards),
        Path(args.output_directory),
        args.mode,
        args.column_name,
    )


def main():

    path_obj, odir, mode, column_name = parse_arguments()

    logging.info(path_obj)

    # Check if the path exists
    if not path_obj.exists():
        raise FileNotFoundError(f"The path '{path_obj}' does not exist.")

    ishards = []

    # Check if it is a directory
    if path_obj.is_dir():
        # List all .parquet files in the directory
        parquet_files = list(path_obj.glob("*.parquet"))
        ishards = [Path(filename) for filename in parquet_files]

    # Check if it is a Parquet file
    elif path_obj.is_file() and path_obj.suffix == ".parquet":
        ishards = [path_obj]

    # If it's neither a directory nor a Parquet file
    else:
        raise ValueError(
            f"The path '{path}' is neither a directory containing Parquet files nor a Parquet file."
        )

    logging.info(f"#-shards: {len(ishards)}")

    # Create the directory if it does not exist
    os.makedirs(odir, exist_ok=True)

    if mode == "page":

        cnt = 0
        for ishard in ishards:
            for page_no, page_image, page_items, doc in iterate_page_objects(
                ishard, column_name=column_name
            ):
                logging.info(f"page: {page_no} => #-objects: {len(page_items)}")

                page = doc.pages[page_no]

                json_items = {
                    "page": page_dim.model_dump(
                        mode="json", by_alias=True, exclude_none=True
                    ),
                    "clusters": [],
                }
                for item in page_items:
                    json_items["clusters"].append(
                        item.model_dump(mode="json", by_alias=True, exclude_none=True)
                    )

                with open(odir / f"page_{cnt:06}.json", "w") as fw:
                    fw.write(json.dumps(json_items, indent=2))

                page_image.save(odir / f"page_{cnt:06}.png")
                cnt += 1

    elif mode == "table":

        cnt = 0
        for ishard in ishards:
            for page_no, page_image, table_image, table, doc in iterate_tables(
                ishard, column_name=column_name
            ):
                table_image.show()

                page_dim = doc.pages[page_no]

                table_html = table.export_to_html(doc)
                table_json = table.model_dump(
                    mode="json", by_alias=True, exclude_none=True
                )

                html = create_styled_html(table_html)

                table_json["page_dimension"] = page_dim.model_dump(
                    mode="json", by_alias=True, exclude_none=True
                )

                with open(odir / f"table_{cnt:06}.html", "w") as fw:
                    fw.write(html)

                with open(odir / f"table_{cnt:06}.json", "w") as fw:
                    fw.write(json.dumps(table_json, indent=2))

                table_image.save(odir / f"table_{cnt:06}.png")
                cnt += 1

    elif mode == "docitems":

        cnt = 0
        for ishard in ishards:
            for page_no, page_image, item_image, item, doc in iterate_docitems(
                ishard, column_name=column_name
            ):
                page = doc.pages[page_no]

                item_json = item.model_dump(
                    mode="json", by_alias=True, exclude_none=True
                )
                item_json["page"] = page.model_dump(
                    mode="json", by_alias=True, exclude_none=True
                )
                with open(odir / f"{item.label}_{cnt:06}.json", "w") as fw:
                    fw.write(json.dumps(item_json, indent=2))

                item_image.save(odir / f"{item.label}_{cnt:06}.png")
                cnt += 1


if __name__ == "__main__":
    main()
