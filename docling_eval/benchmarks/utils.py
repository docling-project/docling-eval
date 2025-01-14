import copy
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import pypdfium2 as pdfium
from bs4 import BeautifulSoup  # type: ignore
from datasets import Features
from datasets import Image as Features_Image
from datasets import Sequence, Value
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import InputFormat, Page
from docling.datamodel.document import InputDocument
from docling_core.types.doc.base import Size
from docling_core.types.doc.document import (
    DocItem,
    DoclingDocument,
    ImageRef,
    ImageRefMode,
    PageItem,
    TableCell,
    TableData,
)
from docling_core.types.doc.labels import DocItemLabel
from PIL import Image, ImageDraw, ImageFont

from docling_eval.benchmarks.constants import BenchMarkColumns
from docling_eval.docling.constants import (
    HTML_COMPARISON_PAGE,
    HTML_COMPARISON_PAGE_WITH_CLUSTERS,
    HTML_DEFAULT_HEAD_FOR_COMP,
    HTML_INSPECTION,
)
from docling_eval.docling.utils import from_pil_to_base64, from_pil_to_base64uri

"""
def write_datasets_info(
    name: str, output_dir: Path, num_train_rows: int, num_test_rows: int
):

    columns = [
        {"name": BenchMarkColumns.DOCLING_VERSION, "type": "string"},
        {"name": BenchMarkColumns.STATUS, "type": "string"},
        {"name": BenchMarkColumns.DOC_ID, "type": "string"},
        {"name": BenchMarkColumns.GROUNDTRUTH, "type": "string"},
        {"name": BenchMarkColumns.PREDICTION, "type": "string"},
        {"name": BenchMarkColumns.ORIGINAL, "type": "string"},
        {"name": BenchMarkColumns.MIMETYPE, "type": "string"},
        {"name": BenchMarkColumns.PREDICTION_PICTURES, "type": {"list": {"item": "Image"}}},
        {"name": BenchMarkColumns.PREDICTION_PAGE_IMAGES, "type": {"list": {"item": "Image"}}},
        {"name": BenchMarkColumns.GROUNDTRUTH_PICTURES, "type": {"list": {"item": "Image"}}},
        {"name": BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES, "type": {"list": {"item": "Image"}}},
    ]

    dataset_infos = {
        "train": {
            "description": f"Training split of {name}",
            "schema": {"columns": columns},
            "num_rows": num_train_rows,
        },
        "test": {
            "description": f"Test split of {name}",
            "schema": {"columns": columns},
            "num_rows": num_test_rows,
        },
    }

    with open(output_dir / f"dataset_infos.json", "w") as fw:
        fw.write(json.dumps(dataset_infos, indent=2))
"""


def write_datasets_info(
    name: str, output_dir: Path, num_train_rows: int, num_test_rows: int
):
    features = Features(
        {
            BenchMarkColumns.DOCLING_VERSION: Value("string"),
            BenchMarkColumns.STATUS: Value("string"),
            BenchMarkColumns.DOC_ID: Value("string"),
            BenchMarkColumns.GROUNDTRUTH: Value("string"),
            BenchMarkColumns.PREDICTION: Value("string"),
            BenchMarkColumns.ORIGINAL: Value("string"),
            BenchMarkColumns.MIMETYPE: Value("string"),
            BenchMarkColumns.PREDICTION_PICTURES: Sequence(Features_Image()),
            BenchMarkColumns.PREDICTION_PAGE_IMAGES: Sequence(Features_Image()),
            BenchMarkColumns.GROUNDTRUTH_PICTURES: Sequence(Features_Image()),
            BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES: Sequence(Features_Image()),
        }
    )

    schema = features.to_dict()
    print(json.dumps(schema, indent=2))

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


def get_input_document(file: Path):
    return InputDocument(
        path_or_stream=file,
        format=InputFormat.PDF,  # type: ignore[arg-type]
        filename=file.name,
        backend=DoclingParseV2DocumentBackend,
    )


def add_pages_to_true_doc(
    pdf_path: Path, true_doc: DoclingDocument, image_scale: float = 1.0
):
    in_doc = get_input_document(pdf_path)
    assert in_doc.valid, "Input doc must be valid."
    # assert in_doc.page_count == 1, "doc must have one page."

    # add the pages
    page_images: List[Image.Image] = []

    for page_no in range(0, in_doc.page_count):
        page = Page(page_no=page_no)
        page._backend = in_doc._backend.load_page(page.page_no)

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
    rows = table.find_all("tr")

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


def save_comparison_html(
    filename: Path,
    true_doc: DoclingDocument,
    pred_doc: DoclingDocument,
    page_image: Image.Image,
    true_labels: Set[DocItemLabel],
    pred_labels: Set[DocItemLabel],
):

    true_doc_html = true_doc.export_to_html(
        image_mode=ImageRefMode.EMBEDDED,
        html_head=HTML_DEFAULT_HEAD_FOR_COMP,
        labels=true_labels,
    )

    pred_doc_html = pred_doc.export_to_html(
        image_mode=ImageRefMode.EMBEDDED,
        html_head=HTML_DEFAULT_HEAD_FOR_COMP,
        labels=pred_labels,
    )

    # since the string in srcdoc are wrapped by ', we need to replace all ' by it HTML convention
    true_doc_html = true_doc_html.replace("'", "&#39;")
    pred_doc_html = pred_doc_html.replace("'", "&#39;")

    image_base64 = from_pil_to_base64(page_image)

    """
    # Convert the image to a bytes object
    buffered = io.BytesIO()
    page_image.save(
        buffered, format="PNG"
    )  # Specify the format (e.g., JPEG, PNG, etc.)
    image_bytes = buffered.getvalue()

    # Encode the bytes to a Base64 string
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    """

    comparison_page = copy.deepcopy(HTML_COMPARISON_PAGE)
    comparison_page = comparison_page.replace("BASE64PAGE", image_base64)
    comparison_page = comparison_page.replace("TRUEDOC", true_doc_html)
    comparison_page = comparison_page.replace("PREDDOC", pred_doc_html)

    with open(str(filename), "w") as fw:
        fw.write(comparison_page)


def draw_clusters_with_reading_order(
    doc: DoclingDocument,
    page_image: Image.Image,
    labels: Set[DocItemLabel],
    page_no: int = 1,
    reading_order: bool = True,
):

    # img = copy.deepcopy(page_image)
    img = page_image.copy()
    draw = ImageDraw.Draw(img)

    # Load a font (adjust the font size and path as needed)
    font = ImageFont.load_default()
    try:
        font = ImageFont.truetype("arial.ttf", size=15)
    except IOError:
        font = ImageFont.load_default()

    x0, y0 = None, None

    for item, level in doc.iterate_items():
        if isinstance(item, DocItem):  # and item.label in labels:
            for prov in item.prov:

                if page_no != prov.page_no:
                    continue

                bbox = prov.bbox.to_top_left_origin(
                    page_height=doc.pages[prov.page_no].size.height
                )
                bbox = bbox.normalized(doc.pages[prov.page_no].size)

                bbox.l = round(bbox.l * img.width)
                bbox.r = round(bbox.r * img.width)
                bbox.t = round(bbox.t * img.height)
                bbox.b = round(bbox.b * img.height)

                if bbox.b > bbox.t:
                    bbox.b, bbox.t = bbox.t, bbox.b

                if not reading_order:
                    x0, y0 = None, None
                elif x0 is None and y0 is None:
                    x0 = (bbox.l + bbox.r) / 2.0
                    y0 = (bbox.b + bbox.t) / 2.0
                else:
                    x1 = (bbox.l + bbox.r) / 2.0
                    y1 = (bbox.b + bbox.t) / 2.0

                    # Arrow parameters
                    start_point = (x0, y0)  # Starting point of the arrow
                    end_point = (x1, y1)  # Ending point of the arrow
                    arrowhead_length = 20  # Length of the arrowhead
                    arrowhead_width = 10  # Width of the arrowhead

                    arrow_color = "red"
                    line_width = 2

                    # Draw the arrow shaft (line)
                    draw.line(
                        [start_point, end_point], fill=arrow_color, width=line_width
                    )

                    # Calculate the arrowhead points
                    dx = end_point[0] - start_point[0]
                    dy = end_point[1] - start_point[1]
                    angle = (dx**2 + dy**2) ** 0.5 + 0.01  # Length of the arrow shaft

                    # Normalized direction vector for the arrow shaft
                    ux, uy = dx / angle, dy / angle

                    # Base of the arrowhead
                    base_x = end_point[0] - ux * arrowhead_length
                    base_y = end_point[1] - uy * arrowhead_length

                    # Left and right points of the arrowhead
                    left_x = base_x - uy * arrowhead_width
                    left_y = base_y + ux * arrowhead_width
                    right_x = base_x + uy * arrowhead_width
                    right_y = base_y - ux * arrowhead_width

                    # Draw the arrowhead (triangle)
                    draw.polygon(
                        [end_point, (left_x, left_y), (right_x, right_y)],
                        fill=arrow_color,
                    )

                    x0, y0 = x1, y1

                # Draw rectangle with only a border
                rectangle_color = "blue"
                border_width = 1
                draw.rectangle(
                    [bbox.l, bbox.b, bbox.r, bbox.t],
                    outline=rectangle_color,
                    width=border_width,
                )

                # Calculate label size using getbbox
                text_bbox = font.getbbox(str(item.label))
                label_width = text_bbox[2] - text_bbox[0]
                label_height = text_bbox[3] - text_bbox[1]
                label_x = bbox.l
                label_y = (
                    bbox.b - label_height
                )  # - 5  # Place the label above the rectangle

                # Draw label text
                draw.text(
                    (label_x, label_y),
                    str(item.label),
                    fill=rectangle_color,
                    font=font,
                )

    return img


def save_comparison_html_with_clusters(
    filename: Path,
    true_doc: DoclingDocument,
    pred_doc: DoclingDocument,
    page_image: Image.Image,
    true_labels: Set[DocItemLabel],
    pred_labels: Set[DocItemLabel],
):

    def draw_clusters(doc: DoclingDocument, labels: Set[DocItemLabel]):

        img = copy.deepcopy(page_image)
        draw = ImageDraw.Draw(img)

        # Load a font (adjust the font size and path as needed)
        font = ImageFont.load_default()
        try:
            font = ImageFont.truetype("arial.ttf", size=15)
        except IOError:
            font = ImageFont.load_default()

        x0, y0 = None, None

        for item, level in doc.iterate_items():
            if isinstance(item, DocItem):  # and item.label in labels:
                for prov in item.prov:

                    bbox = prov.bbox.to_top_left_origin(
                        page_height=doc.pages[prov.page_no].size.height
                    )
                    bbox = bbox.normalized(doc.pages[prov.page_no].size)

                    bbox.l = round(bbox.l * img.width)
                    bbox.r = round(bbox.r * img.width)
                    bbox.t = round(bbox.t * img.height)
                    bbox.b = round(bbox.b * img.height)

                    if bbox.b > bbox.t:
                        bbox.b, bbox.t = bbox.t, bbox.b

                    if x0 is None and y0 is None:
                        x0 = (bbox.l + bbox.r) / 2.0
                        y0 = (bbox.b + bbox.t) / 2.0
                    else:
                        x1 = (bbox.l + bbox.r) / 2.0
                        y1 = (bbox.b + bbox.t) / 2.0

                        # Arrow parameters
                        start_point = (x0, y0)  # Starting point of the arrow
                        end_point = (x1, y1)  # Ending point of the arrow
                        arrowhead_length = 20  # Length of the arrowhead
                        arrowhead_width = 10  # Width of the arrowhead

                        arrow_color = "red"
                        line_width = 2

                        # Draw the arrow shaft (line)
                        draw.line(
                            [start_point, end_point], fill=arrow_color, width=line_width
                        )

                        # Calculate the arrowhead points
                        dx = end_point[0] - start_point[0]
                        dy = end_point[1] - start_point[1]
                        angle = (
                            dx**2 + dy**2
                        ) ** 0.5 + 0.01  # Length of the arrow shaft

                        # Normalized direction vector for the arrow shaft
                        ux, uy = dx / angle, dy / angle

                        # Base of the arrowhead
                        base_x = end_point[0] - ux * arrowhead_length
                        base_y = end_point[1] - uy * arrowhead_length

                        # Left and right points of the arrowhead
                        left_x = base_x - uy * arrowhead_width
                        left_y = base_y + ux * arrowhead_width
                        right_x = base_x + uy * arrowhead_width
                        right_y = base_y - ux * arrowhead_width

                        # Draw the arrowhead (triangle)
                        draw.polygon(
                            [end_point, (left_x, left_y), (right_x, right_y)],
                            fill=arrow_color,
                        )

                        x0, y0 = x1, y1

                    # Draw rectangle with only a border
                    rectangle_color = "blue"
                    border_width = 1
                    draw.rectangle(
                        [bbox.l, bbox.b, bbox.r, bbox.t],
                        outline=rectangle_color,
                        width=border_width,
                    )

                    # Calculate label size using getbbox
                    text_bbox = font.getbbox(str(item.label))
                    label_width = text_bbox[2] - text_bbox[0]
                    label_height = text_bbox[3] - text_bbox[1]
                    label_x = bbox.l
                    label_y = (
                        bbox.b - label_height
                    )  # - 5  # Place the label above the rectangle

                    # Draw label text
                    draw.text(
                        (label_x, label_y),
                        str(item.label),
                        fill=rectangle_color,
                        font=font,
                    )

        return img

    true_doc_html = true_doc.export_to_html(
        image_mode=ImageRefMode.EMBEDDED,
        html_head=HTML_DEFAULT_HEAD_FOR_COMP,
        labels=true_labels,
    )

    pred_doc_html = pred_doc.export_to_html(
        image_mode=ImageRefMode.EMBEDDED,
        html_head=HTML_DEFAULT_HEAD_FOR_COMP,
        labels=pred_labels,
    )

    # since the string in srcdoc are wrapped by ', we need to replace all ' by it HTML convention
    true_doc_html = true_doc_html.replace("'", "&#39;")
    pred_doc_html = pred_doc_html.replace("'", "&#39;")

    true_doc_img = draw_clusters(doc=true_doc, labels=true_labels)
    pred_doc_img = draw_clusters(doc=pred_doc, labels=pred_labels)

    true_doc_img_b64 = from_pil_to_base64(true_doc_img)
    pred_doc_img_b64 = from_pil_to_base64(pred_doc_img)

    comparison_page = copy.deepcopy(HTML_COMPARISON_PAGE_WITH_CLUSTERS)
    comparison_page = comparison_page.replace("BASE64TRUEPAGE", true_doc_img_b64)
    comparison_page = comparison_page.replace("TRUEDOC", true_doc_html)
    comparison_page = comparison_page.replace("BASE64PREDPAGE", pred_doc_img_b64)
    comparison_page = comparison_page.replace("PREDDOC", pred_doc_html)

    with open(str(filename), "w") as fw:
        fw.write(comparison_page)


def save_inspection_html(
    filename: Path, doc: DoclingDocument, labels: Set[DocItemLabel]
):

    html_doc = doc.export_to_html(image_mode=ImageRefMode.EMBEDDED, labels=labels)
    html_doc = html_doc.replace("'", "&#39;")

    page_images = []
    page_template = '<div class="image-wrapper"><img src="data:image/png;base64,BASE64PAGE" alt="Example Image"></div>'
    for page_no, page in doc.pages.items():
        page_img = page.image.pil_image

        page_img = draw_clusters_with_reading_order(
            doc=doc,
            page_image=page_img,
            labels=labels,
            page_no=page_no,
            reading_order=True,
        )

        page_base64 = from_pil_to_base64(page_img)
        page_images.append(page_template.replace("BASE64PAGE", page_base64))

    html_viz = copy.deepcopy(HTML_INSPECTION)
    html_viz = html_viz.replace("PREDDOC", html_doc)
    html_viz = html_viz.replace("PAGE_IMAGES", "\n".join(page_images))

    with open(str(filename), "w") as fw:
        fw.write(html_viz)
