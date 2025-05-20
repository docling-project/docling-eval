import base64
import copy
import hashlib
import io
import json
import logging
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import PIL.Image
from bs4 import BeautifulSoup  # type: ignore
from datasets import Dataset, Features
from datasets import Image as Features_Image
from datasets import Sequence, Value, load_dataset
from datasets.iterable_dataset import IterableDataset
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import InputFormat, Page
from docling.datamodel.document import InputDocument
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import (
    DoclingDocument,
    GraphData,
    ImageRef,
    PageItem,
    TableCell,
    TableData,
)
from docling_core.types.doc.labels import GraphCellLabel
from docling_core.types.doc.page import BoundingRectangle, SegmentedPage, TextCell
from PIL import Image
from pydantic import AnyUrl
from torch import Tensor

from docling_eval.datamodels.types import (
    BenchMarkColumns,
    EvaluationModality,
    PredictionProviderType,
)

SPECIAL_CHARS = list("*:;,.?()!@#$%^&[]{}/\\\"'~+-_<>=")
CLOSE_THRESHOLD = 0.3
NUMBERS_CLOSE_THRESHOLD = 0.7
CLOSE_LEFT_THRESHOLD = 0.7
INSIDE_THRESHOLD = 0.15


def get_binhash(binary_data: bytes) -> str:
    # Create a hash object (e.g., SHA-256)
    hash_object = hashlib.sha256()

    # Update the hash object with the binary data
    hash_object.update(binary_data)

    # Get the hexadecimal digest of the hash
    hash_hex = hash_object.hexdigest()
    return hash_hex


def write_datasets_info(
    name: str,
    output_dir: Path,
    num_train_rows: int,
    num_test_rows: int,
    features: Features,
):
    schema = features.to_dict()
    # print(json.dumps(schema, indent=2))

    dataset_infos = {
        "train": {
            "description": f"Training split of {name}",
            "schema": schema,
            "num_rows": num_train_rows,
        },
        "test": {
            "description": f"Test split of {name}",
            "schema": schema,
            "num_rows": num_test_rows,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "dataset_infos.json", "w") as fw:
        json.dump(dataset_infos, fw, indent=2)


def get_input_document(file: Path | BytesIO) -> InputDocument:
    return InputDocument(
        path_or_stream=file,
        format=InputFormat.PDF,  # type: ignore[arg-type]
        filename=file.name if isinstance(file, Path) else "foo",
        backend=DoclingParseV4DocumentBackend,
    )


def from_pil_to_base64uri(img: Image.Image) -> AnyUrl:

    image_base64 = from_pil_to_base64(img)
    uri = AnyUrl(f"data:image/png;base64,{image_base64}")

    return uri


def add_pages_to_true_doc(
    pdf_path: Path | BytesIO, true_doc: DoclingDocument, image_scale: float = 1.0
):
    in_doc = get_input_document(pdf_path)
    assert in_doc.valid, "Input doc must be valid."
    # assert in_doc.page_count == 1, "doc must have one page."

    # add the pages
    page_images: List[Image.Image] = []

    for page_no in range(0, in_doc.page_count):
        page = Page(page_no=page_no)
        page._backend = in_doc._backend.load_page(page.page_no)  # type: ignore[attr-defined]

        if page._backend is not None and page._backend.is_valid():
            page.size = page._backend.get_size()

            page_width, page_height = page.size.width, page.size.height

            page_image = page.get_image(scale=image_scale)
            if page_image is not None:
                page_images.append(page_image)
                image_ref = ImageRef(
                    mimetype="image/png",
                    dpi=round(72 * image_scale),
                    size=Size(
                        width=float(page_image.width), height=float(page_image.height)
                    ),
                    # uri=Path(f"{BenchMarkColumns.PAGE_IMAGES}/{page_no}"),
                    uri=from_pil_to_base64uri(page_image),
                )
                page_item = PageItem(
                    page_no=page_no + 1,
                    size=Size(width=float(page_width), height=float(page_height)),
                    image=image_ref,
                )

                true_doc.pages[page_no + 1] = page_item
                # page_image.show()
            else:
                logging.warning("did not get image for page `add_pages_to_true_doc`")

            page._backend.unload()

    return true_doc, page_images


def yield_cells_from_html_table(
    table_html: str, text_cells: Optional[List[Dict]] = None
):
    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table") or soup  # Ensure table context
    rows = table.find_all("tr")  # type: ignore

    max_cols = 0
    for row in rows:
        # cols = row.find_all(["td", "th"])
        # max_cols = max(max_cols, len(cols))  # Determine maximum columns

        num_cols = 0
        for cell in row.find_all(["td", "th"]):
            num_cols += int(cell.get("colspan", 1))

        max_cols = max(max_cols, num_cols)  # Determine maximum columns

    # Create grid to track cell positions
    grid = [[None for _ in range(max_cols)] for _ in range(len(rows))]

    text_cell_id = 0
    for row_idx, row in enumerate(rows):
        col_idx = 0  # Start from first column
        for cell in row.find_all(["td", "th"]):
            # Skip over filled grid positions (handle previous rowspan/colspan)
            while grid[row_idx][col_idx] is not None:
                col_idx += 1

            # Get text, rowspan, and colspan
            text = cell.get_text(strip=True)

            if len(text) == 0 and text_cells is not None:
                text_cell = text_cells[text_cell_id]
                text = "".join(text_cell["tokens"])

            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))

            # Fill grid positions and yield (row, column, text)
            for r in range(rowspan):
                for c in range(colspan):
                    grid[row_idx + r][col_idx + c] = text

            # print(f"Row: {row_idx + 1}, Col: {col_idx + 1}, Text: {text}")
            yield row_idx, col_idx, rowspan, colspan, text

            col_idx += colspan  # Move to next column after colspan

            text_cell_id += 1


def convert_html_table_into_docling_tabledata(
    table_html: str, text_cells: Optional[List] = None
) -> TableData:

    num_rows = -1
    num_cols = -1

    cells = []

    try:
        for row_idx, col_idx, rowspan, colspan, text in yield_cells_from_html_table(
            table_html=table_html, text_cells=text_cells
        ):
            cell = TableCell(
                row_span=rowspan,
                col_span=colspan,
                start_row_offset_idx=row_idx,
                end_row_offset_idx=row_idx + rowspan,
                start_col_offset_idx=col_idx,
                end_col_offset_idx=col_idx + colspan,
                text=text,
            )
            cells.append(cell)

            num_rows = max(row_idx + rowspan, num_rows)
            num_cols = max(col_idx + colspan, num_cols)

    except:
        logging.error("No table-structure identified")

    return TableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells)


def docling_version() -> str:
    return version("docling")  # may raise PackageNotFoundError


def docling_models_version() -> str:
    return version("docling-ibm-models")  # may raise PackageNotFoundError


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

    pictures: list[PIL.Image.Image] = []
    page_images: list[PIL.Image.Image] = []

    # Save page images
    for img_no, picture in enumerate(document.pictures):
        if picture.image is not None and picture.image.pil_image is not None:
            img_ind = len(pictures)

            pictures.append(picture.image.pil_image)
            picture.image.uri = Path(f"{pictures_column}/{img_ind}")

    # Save page images
    for page_no, page in document.pages.items():
        if page.image is not None and page.image.pil_image is not None:
            img_ind = len(page_images)

            page_images.append(page.image.pil_image)
            page.image.uri = Path(f"{page_images_column}/{img_ind}")

    return document, pictures, page_images


def insert_images_from_pil(
    document: DoclingDocument,
    pictures: List[PIL.Image.Image],
    page_images: List[PIL.Image.Image],
) -> DoclingDocument:

    # Inject picture images
    for pic_no, picture in enumerate(document.pictures):
        if picture.image is not None:

            uri = str(picture.image.uri)
            if uri.startswith(BenchMarkColumns.GROUNDTRUTH_PICTURES) or uri.startswith(
                BenchMarkColumns.PREDICTION_PICTURES
            ):

                img_parts = str(picture.image.uri).split("/")
                img_ind = int(img_parts[-1])

                assert img_ind < len(pictures)

                picture.image._pil = pictures[img_ind]
                picture.image.uri = from_pil_to_base64uri(pictures[img_ind])

    # Inject page images
    for page_no, page in document.pages.items():
        if page.image is not None:

            uri = str(page.image.uri)
            if uri.startswith(
                BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES
            ) or uri.startswith(BenchMarkColumns.PREDICTION_PAGE_IMAGES):
                img_parts = str(page.image.uri).split("/")
                img_ind = int(img_parts[-1])

                assert img_ind < len(page_images)

                page.image._pil = page_images[img_ind]
                page.image.uri = from_pil_to_base64uri(page_images[img_ind])

    return document


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
    if shard_format == "json":
        batch.to_json(output_file)

    elif shard_format == "parquet":
        batch.to_parquet(output_file)

    else:
        raise ValueError(f"Unsupported shard_format: {shard_format}")

    logging.info(f"Saved shard {shard_id} to {output_file} with {len(items)} documents")

    shard_id += 1

    return shard_id, [], 0


def dataset_exists(
    ds_path: Path,
    split: str,
) -> bool:
    r"""
    It returns True if a parquet dataset exists for the given split and has data.
    """
    try:
        parquet_files = str(ds_path / split / "*.parquet")
        ds: IterableDataset = load_dataset(
            "parquet",
            data_files={split: parquet_files},
            split=split,
            streaming=True,
        )
        for d in ds:
            return True
    except Exception as ex:
        pass
    return False


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


def set_selection_range(
    begin_index: int, end_index: int, ds_len: int
) -> Tuple[int, int]:
    r"""
    Set the final values of begin_index, end_index out of their initial values and dataset length
    Raises exception if the indices are out of range
    """
    if end_index == -1 or end_index > ds_len:
        end_index = ds_len

    if begin_index > end_index:
        raise RuntimeError("Cannot have from_sample_index > to_sample_index")

    if begin_index >= ds_len or end_index > ds_len:
        raise IndexError(f"The sample indices go beyond the dataset size: {ds_len}")

    return begin_index, end_index


def classify_cells(graph: GraphData) -> None:
    """
    for a graph consisting of a list of GraphCell objects (nodes) and a list of GraphLink objects
    (directed edges), update each cell's label according to the following rules:
      - If a node has no outgoing edges, label it as VALUE.
      - If a node has no incoming edges and has one or more outgoing edges, label it as KEY.
      - If a node has one or more incoming edges and one or more outgoing edges, but every neighbor it points to is a leaf (has no outgoing edges), label it as KEY.
      - Otherwise, label it as UNSPECIFIED.

    this function modifies the cells in place.
    """
    # for tracking the values
    indegree = defaultdict(int)
    outdegree = defaultdict(int)
    outgoing_neighbors: defaultdict[int, List[int]] = defaultdict(list)

    cells, links = graph.cells, graph.links

    # initialization
    for cell in cells:
        indegree[cell.cell_id] = 0
        outdegree[cell.cell_id] = 0
        outgoing_neighbors[cell.cell_id] = []

    # populate the values
    for link in links:
        src = link.source_cell_id
        tgt = link.target_cell_id
        outdegree[src] += 1
        indegree[tgt] += 1
        outgoing_neighbors[src].append(tgt)

    # now, we assign labels based on the computed degrees.
    for cell in cells:
        cid = cell.cell_id
        if outdegree[cid] == 0:
            # if a node is a leaf, it is a VALUE.
            cell.label = GraphCellLabel.VALUE
        elif indegree[cid] == 0:
            # no incoming and at least one outgoing means it is a KEY.
            cell.label = GraphCellLabel.KEY
        elif outdegree[cid] > 0 and indegree[cid] > 0:
            # if all outgoing neighbors are leaves (i.e. outdegree == 0),
            # then this node is a KEY.
            if all(outdegree[neighbor] == 0 for neighbor in outgoing_neighbors[cid]):
                cell.label = GraphCellLabel.KEY
            else:
                # otherwise, it is UNSPECIFIED.
                cell.label = GraphCellLabel.UNSPECIFIED
        else:
            # fallback case.
            cell.label = GraphCellLabel.UNSPECIFIED


def sort_cell_ids(doc: DoclingDocument) -> None:
    mapping = {}
    for i, item in enumerate(doc.key_value_items[0].graph.cells):
        mapping[item.cell_id] = i
    for i, item in enumerate(doc.key_value_items[0].graph.cells):
        item.cell_id = mapping[item.cell_id]
    for i, link in enumerate(doc.key_value_items[0].graph.links):
        link.source_cell_id = mapping[link.source_cell_id]
        link.target_cell_id = mapping[link.target_cell_id]


def get_package_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def tensor_to_float(t: Union[Tensor, float]) -> float:
    r"""Get float from tensor item"""
    if isinstance(t, Tensor):
        return float(t.item())
    return t


def modalities_of_prediction_type(
    prediction_provider_type: PredictionProviderType,
) -> Optional[List[EvaluationModality]]:
    r"""
    Return a list of EvaluationModality supported by the given prediction_provider_type
    """
    from docling_eval.prediction_providers.docling_provider import (
        DoclingPredictionProvider,
    )
    from docling_eval.prediction_providers.tableformer_provider import (
        TableFormerPredictionProvider,
    )

    # TODO: Update this map as more prediction providers are implemented
    prediction_type_class = {
        PredictionProviderType.DOCLING: DoclingPredictionProvider,
        PredictionProviderType.TABLEFORMER: TableFormerPredictionProvider,
    }

    if prediction_provider_type not in prediction_type_class:
        return None

    prediction_provider_class = prediction_type_class[prediction_provider_type]
    if not hasattr(prediction_provider_class, "prediction_modalities"):
        return None

    return prediction_provider_class.prediction_modalities


def get_y_axis_iou(rect1: BoundingRectangle, rect2: BoundingRectangle):
    """Calculate the IoU (Intersection over Union) in the y-axis between two BoundingRectangle objects."""
    top1 = min(rect1.r_y0, rect1.r_y1, rect1.r_y2, rect1.r_y3)
    bottom1 = max(rect1.r_y0, rect1.r_y1, rect1.r_y2, rect1.r_y3)

    # Find top and bottom of rect2
    top2 = min(rect2.r_y0, rect2.r_y1, rect2.r_y2, rect2.r_y3)
    bottom2 = max(rect2.r_y0, rect2.r_y1, rect2.r_y2, rect2.r_y3)

    y_overlap = max(0, min(bottom1, bottom2) - max(top1, top2))
    y_union = max(bottom1, bottom2) - min(top1, top2)

    return y_overlap / y_union if y_union > 0 else 0


def text_cell_to_word_dict(cell: TextCell):
    """Convert a TextCell to a word dictionary format used by the merging functions."""
    left = min(cell.rect.r_x0, cell.rect.r_x1, cell.rect.r_x2, cell.rect.r_x3)
    top = min(cell.rect.r_y0, cell.rect.r_y1, cell.rect.r_y2, cell.rect.r_y3)
    right = max(cell.rect.r_x0, cell.rect.r_x1, cell.rect.r_x2, cell.rect.r_x3)
    bottom = max(cell.rect.r_y0, cell.rect.r_y1, cell.rect.r_y2, cell.rect.r_y3)

    bbox = (left, top, right, bottom)
    return {
        "word": cell.text,
        "bbox": bbox,
        "rect": cell.rect,
        "orig": cell.orig,
        "from_ocr": cell.from_ocr,
    }


def word_dict_to_text_cell(word_dict):
    """Convert a word dictionary back to a TextCell."""
    if "rect" in word_dict:
        rect = word_dict["rect"]
    else:
        bbox = word_dict["bbox"]
        rect = BoundingRectangle(
            r_x0=bbox[0],  # left-top
            r_y0=bbox[1],
            r_x1=bbox[2],  # right-top
            r_y1=bbox[1],
            r_x2=bbox[2],  # right-bottom
            r_y2=bbox[3],
            r_x3=bbox[0],  # left-bottom
            r_y3=bbox[3],
            coord_origin=CoordOrigin.TOPLEFT,
        )

    return TextCell(
        rect=rect,
        text=word_dict["word"],
        orig=word_dict.get("orig", word_dict["word"]),
        from_ocr=word_dict.get("from_ocr", False),
    )


def find_close_right_and_left(
    special_char_word, words_array, threshold, only_numbers=False
):
    """Find words that are close to the left and right of a special character word."""
    special_char_left = special_char_word["bbox"][0]
    special_char_right = special_char_word["bbox"][2]

    left_found = False
    right_found = False

    left_word = None
    right_word = None
    found_flag = False

    for word in words_array:
        y_axis_iou = get_y_axis_iou(special_char_word["rect"], word["rect"])
        if y_axis_iou < 0.6:
            continue

        bbox = word["bbox"]
        left = bbox[0]
        right = bbox[2]
        top = bbox[1]
        bottom = bbox[3]
        height = bottom - top + 1
        margin = int(threshold * height + 0.5)
        inside_margin = -int(INSIDE_THRESHOLD * height + 0.5)

        if not left_found:
            left_diff = special_char_left - right
            end_char = word["word"][-1] if word["word"] else ""
            if (left_diff <= margin) and (left_diff >= 0):
                if (not only_numbers) or (only_numbers and end_char.isdigit()):
                    left_found = True
                    left_word = word
                    if right_found:
                        found_flag = True
                        break

        if not right_found:
            right_diff = left - special_char_right
            start_char = word["word"][0] if word["word"] else ""
            if (right_diff <= margin) and (right_diff >= inside_margin):
                if (not only_numbers) or (only_numbers and start_char.isdigit()):
                    right_found = True
                    right_word = word
                    if left_found:
                        found_flag = True
                        break

    return found_flag, left_word, right_word


def find_close_right(special_char_word, words_array, threshold):
    """Find words that are close to the right of a special character word."""
    special_char_right = special_char_word["bbox"][2]

    right_word = None
    found_flag = False

    for word in words_array:
        y_axis_iou = get_y_axis_iou(special_char_word["rect"], word["rect"])
        if y_axis_iou < 0.6:
            continue

        bbox = word["bbox"]
        left = bbox[0]
        top = bbox[1]
        bottom = bbox[3]
        height = bottom - top + 1
        margin = int(threshold * height + 0.5)
        inside_margin = -int(INSIDE_THRESHOLD * height + 0.5)

        right_diff = left - special_char_right
        if (right_diff <= margin) and (right_diff >= inside_margin):
            right_word = word
            found_flag = True
            break

    return found_flag, right_word


def find_close_left(special_char_word, words_array, threshold):
    """Find words that are close to the left of a special character word."""
    special_char_left = special_char_word["bbox"][0]

    left_word = None
    found_flag = False

    for word in words_array:
        y_axis_iou = get_y_axis_iou(special_char_word["rect"], word["rect"])
        if y_axis_iou < 0.6:
            continue

        bbox = word["bbox"]
        right = bbox[2]
        top = bbox[1]
        bottom = bbox[3]
        height = bottom - top + 1
        margin = int(threshold * height + 0.5)
        inside_margin = -int(INSIDE_THRESHOLD * height + 0.5)

        left_diff = special_char_left - right
        if (left_diff <= margin) and (left_diff >= inside_margin):
            left_word = word
            found_flag = True
            break

    return found_flag, left_word


def merge_close_left_and_right(
    words_array, special_char_array, threshold, only_numbers
):
    """Merge words that have special characters between them."""
    words_array = copy.deepcopy(words_array)
    found = True
    while found:
        found = False
        for word in words_array:
            if word["word"] in special_char_array:
                words_array_minus_special = copy.deepcopy(words_array)
                words_array_minus_special.remove(word)
                find_flag, left_word, right_word = find_close_right_and_left(
                    word, words_array_minus_special, threshold, only_numbers
                )
                if find_flag:
                    new_content = left_word["word"] + word["word"] + right_word["word"]
                    new_left = left_word["bbox"][0]
                    new_right = right_word["bbox"][2]
                    new_top = min(
                        left_word["bbox"][1], word["bbox"][1], right_word["bbox"][1]
                    )
                    new_bottom = max(
                        left_word["bbox"][3], word["bbox"][3], right_word["bbox"][3]
                    )

                    new_rect = BoundingRectangle(
                        r_x0=new_left,
                        r_y0=new_top,
                        r_x1=new_right,
                        r_y1=new_top,
                        r_x2=new_right,
                        r_y2=new_bottom,
                        r_x3=new_left,
                        r_y3=new_bottom,
                        coord_origin=CoordOrigin.TOPLEFT,
                    )

                    new_word = copy.deepcopy(word)
                    new_word["word"] = new_content
                    new_word["rect"] = new_rect
                    new_word["bbox"] = (new_left, new_top, new_right, new_bottom)
                    new_word["merge"] = True

                    words_array.remove(left_word)
                    words_array.remove(word)
                    words_array.remove(right_word)
                    words_array.append(new_word)
                    found = True
                    break
    return words_array


def merge_to_the_right(words_array, special_char_array, threshold):
    """Merge words where the special character is at the end of a word."""
    words_array = copy.deepcopy(words_array)
    found = True
    while found:
        found = False
        for word in words_array:
            if word["word"] and word["word"][-1] in special_char_array:
                words_array_minus_special = copy.deepcopy(words_array)
                words_array_minus_special.remove(word)
                find_flag, right_word = find_close_right(
                    word, words_array_minus_special, threshold
                )
                if find_flag:
                    new_content = word["word"] + right_word["word"]
                    new_left = word["bbox"][0]
                    new_right = right_word["bbox"][2]
                    new_top = min(word["bbox"][1], right_word["bbox"][1])
                    new_bottom = max(word["bbox"][3], right_word["bbox"][3])

                    new_rect = BoundingRectangle(
                        r_x0=new_left,
                        r_y0=new_top,
                        r_x1=new_right,
                        r_y1=new_top,
                        r_x2=new_right,
                        r_y2=new_bottom,
                        r_x3=new_left,
                        r_y3=new_bottom,
                        coord_origin=CoordOrigin.TOPLEFT,
                    )

                    new_word = copy.deepcopy(word)
                    new_word["word"] = new_content
                    new_word["rect"] = new_rect
                    new_word["bbox"] = (new_left, new_top, new_right, new_bottom)
                    new_word["merge"] = True

                    words_array.remove(word)
                    words_array.remove(right_word)
                    words_array.append(new_word)
                    found = True
                    break
    return words_array


def merge_to_the_left(words_array, special_char_array, threshold):
    """Merge words where the special character is at the beginning of a word."""
    words_array = copy.deepcopy(words_array)
    found = True
    while found:
        found = False
        for word in words_array:
            if word["word"] and word["word"][0] in special_char_array:
                words_array_minus_special = copy.deepcopy(words_array)
                words_array_minus_special.remove(word)
                find_flag, left_word = find_close_left(
                    word, words_array_minus_special, threshold
                )
                if find_flag:
                    new_content = left_word["word"] + word["word"]
                    new_left = left_word["bbox"][0]
                    new_right = word["bbox"][2]
                    new_top = min(left_word["bbox"][1], word["bbox"][1])
                    new_bottom = max(left_word["bbox"][3], word["bbox"][3])

                    new_rect = BoundingRectangle(
                        r_x0=new_left,
                        r_y0=new_top,
                        r_x1=new_right,
                        r_y1=new_top,
                        r_x2=new_right,
                        r_y2=new_bottom,
                        r_x3=new_left,
                        r_y3=new_bottom,
                        coord_origin=CoordOrigin.TOPLEFT,
                    )

                    new_word = copy.deepcopy(word)
                    new_word["word"] = new_content
                    new_word["rect"] = new_rect
                    new_word["bbox"] = (new_left, new_top, new_right, new_bottom)
                    new_word["merge"] = True

                    words_array.remove(left_word)
                    words_array.remove(word)
                    words_array.append(new_word)
                    found = True
                    break
    return words_array


def global_merge(segmented_pages: Dict[int, SegmentedPage]) -> Dict[int, SegmentedPage]:
    """
    Merges close word cells in the segmented pages based on specified criteria.

    Args:
        segmented_pages: Dictionary mapping page numbers to SegmentedPage objects

    Returns:
        Updated dictionary of segmented pages with merged word cells
    """
    result_pages = {}

    for page_no, page in segmented_pages.items():
        words_arr = [text_cell_to_word_dict(cell) for cell in page.word_cells]

        # Apply merging functions
        new_words_arr = merge_close_left_and_right(
            words_arr,
            special_char_array=SPECIAL_CHARS,
            threshold=CLOSE_THRESHOLD,
            only_numbers=False,
        )
        new_words_arr2 = merge_close_left_and_right(
            new_words_arr,
            special_char_array=list(",.-/"),
            threshold=NUMBERS_CLOSE_THRESHOLD,
            only_numbers=True,
        )
        new_words_arr3 = merge_to_the_left(
            new_words_arr2,
            special_char_array=list(",."),
            threshold=CLOSE_LEFT_THRESHOLD,
        )
        new_words_arr4 = merge_to_the_left(
            new_words_arr3, special_char_array=SPECIAL_CHARS, threshold=CLOSE_THRESHOLD
        )
        new_words_arr5 = merge_to_the_right(
            new_words_arr4, special_char_array=SPECIAL_CHARS, threshold=CLOSE_THRESHOLD
        )
        new_words_arr6 = merge_to_the_left(
            new_words_arr5,
            special_char_array=list(")]}"),
            threshold=CLOSE_LEFT_THRESHOLD,
        )
        new_words_arr7 = merge_to_the_right(
            new_words_arr6,
            special_char_array=list("([{"),
            threshold=CLOSE_LEFT_THRESHOLD,
        )

        merged_cells = [word_dict_to_text_cell(word) for word in new_words_arr7]

        page.word_cells = merged_cells

        result_pages[page_no] = page

    return result_pages
