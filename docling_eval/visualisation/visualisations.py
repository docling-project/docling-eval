import copy
import logging
import re
from pathlib import Path
from typing import Set

from docling.datamodel.base_models import BoundingBox, Cluster
from docling.utils.visualization import draw_clusters
from docling_core.experimental.serializer.html import (
    HTMLDocSerializer,
    HTMLOutputStyle,
    HTMLParams,
)
from docling_core.types.doc.document import (
    ContentLayer,
    DocItem,
    DoclingDocument,
    ImageRefMode,
)
from docling_core.types.doc.labels import DocItemLabel
from PIL import Image, ImageDraw, ImageFont

from docling_eval.utils.utils import from_pil_to_base64
from docling_eval.visualisation.constants import (
    HTML_COMPARISON_PAGE,
    HTML_COMPARISON_PAGE_WITH_CLUSTERS,
    HTML_DEFAULT_HEAD_FOR_COMP,
    HTML_INSPECTION,
)


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


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    arrow_coords: tuple[float, float, float, float],
    line_width: int = 2,
    color: str = "red",
):
    r"""
    Draw an arrow inside the given draw object
    """
    x0, y0, x1, y1 = arrow_coords

    # Arrow parameters
    start_point = (x0, y0)  # Starting point of the arrow
    end_point = (x1, y1)  # Ending point of the arrow
    arrowhead_length = 20  # Length of the arrowhead
    arrowhead_width = 10  # Width of the arrowhead

    # Draw the arrow shaft (line)
    draw.line([start_point, end_point], fill=color, width=line_width)

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
        fill=color,
    )
    return draw


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

    for item, level in doc.iterate_items(
        included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
    ):
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
                    assert x0 is not None
                    assert y0 is not None

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


# FIXME: to be deleted
def save_comparison_html_with_clusters_legacy(
    filename: Path,
    true_doc: DoclingDocument,
    pred_doc: DoclingDocument,
    page_image: Image.Image,
    true_labels: Set[DocItemLabel],
    pred_labels: Set[DocItemLabel],
    draw_reading_order: bool = True,
):
    if 1 not in true_doc.pages:
        logging.error(f"1 not in true_doc.pages -> skipping {filename} ")
        return
    elif 1 not in pred_doc.pages:
        logging.error(f"1 not in pred_doc.pages -> skipping {filename} ")
        return

    def draw_doc_layout(doc: DoclingDocument, image: Image.Image):
        r"""
        Draw the document clusters and optionaly the reading order
        """
        clusters = []
        for idx, (elem, _) in enumerate(
            doc.iterate_items(
                included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
            )
        ):
            if not isinstance(elem, DocItem):
                continue
            if len(elem.prov) == 0:
                continue  # Skip elements without provenances
            prov = elem.prov[0]

            if prov.page_no not in true_doc.pages or prov.page_no != 1:
                logging.error(f"{prov.page_no} not in true_doc.pages -> skipping! ")
                continue

            tlo_bbox = prov.bbox.to_top_left_origin(
                page_height=doc.pages[prov.page_no].size.height
            )
            cluster = Cluster(
                id=idx,
                label=elem.label,
                bbox=BoundingBox.model_validate(tlo_bbox),
                cells=[],
            )
            clusters.append(cluster)

        scale_x = image.width / doc.pages[1].size.width
        scale_y = image.height / doc.pages[1].size.height
        draw_clusters(image, clusters, scale_x, scale_y)

        return image

    def draw_doc_reading_order(doc: DoclingDocument, image: Image.Image):
        r"""
        Draw the reading order
        """
        draw = ImageDraw.Draw(image)
        x0, y0 = None, None

        for elem, _ in doc.iterate_items(
            included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
        ):
            if not isinstance(elem, DocItem):
                continue
            if len(elem.prov) == 0:
                continue  # Skip elements without provenances
            prov = elem.prov[0]

            if prov.page_no not in true_doc.pages or prov.page_no != 1:
                logging.error(f"{prov.page_no} not in true_doc.pages -> skipping! ")
                continue

            tlo_bbox = prov.bbox.to_top_left_origin(
                page_height=doc.pages[prov.page_no].size.height
            )
            ro_bbox = tlo_bbox.normalized(doc.pages[prov.page_no].size)
            ro_bbox.l = round(ro_bbox.l * image.width)
            ro_bbox.r = round(ro_bbox.r * image.width)
            ro_bbox.t = round(ro_bbox.t * image.height)
            ro_bbox.b = round(ro_bbox.b * image.height)

            if ro_bbox.b > ro_bbox.t:
                ro_bbox.b, ro_bbox.t = ro_bbox.t, ro_bbox.b

            if x0 is None and y0 is None:
                x0 = (ro_bbox.l + ro_bbox.r) / 2.0
                y0 = (ro_bbox.b + ro_bbox.t) / 2.0
            else:
                assert x0 is not None
                assert y0 is not None

                x1 = (ro_bbox.l + ro_bbox.r) / 2.0
                y1 = (ro_bbox.b + ro_bbox.t) / 2.0

                draw = draw_arrow(
                    draw,
                    (x0, y0, x1, y1),
                    line_width=2,
                    color="red",
                )
                x0, y0 = x1, y1
        return image

    # HTML rendering
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

    true_doc_img = draw_doc_layout(true_doc, copy.deepcopy(page_image))
    pred_doc_img = draw_doc_layout(pred_doc, copy.deepcopy(page_image))

    if draw_reading_order:
        true_doc_img = draw_doc_reading_order(true_doc, true_doc_img)
        pred_doc_img = draw_doc_reading_order(pred_doc, pred_doc_img)

    true_doc_img_b64 = from_pil_to_base64(true_doc_img)
    pred_doc_img_b64 = from_pil_to_base64(pred_doc_img)

    comparison_page = copy.deepcopy(HTML_COMPARISON_PAGE_WITH_CLUSTERS)
    comparison_page = comparison_page.replace("BASE64TRUEPAGE", true_doc_img_b64)
    comparison_page = comparison_page.replace("TRUEDOC", true_doc_html)
    comparison_page = comparison_page.replace("BASE64PREDPAGE", pred_doc_img_b64)
    comparison_page = comparison_page.replace("PREDDOC", pred_doc_html)

    with open(str(filename), "w") as fw:
        fw.write(comparison_page)


def save_comparison_html_with_clusters(
    filename: Path,
    true_doc: DoclingDocument,
    pred_doc: DoclingDocument,
    true_labels: Set[DocItemLabel],
    pred_labels: Set[DocItemLabel],
    draw_reading_order: bool = True,
):
    """Save comparison html with clusters."""

    def get_missing_pageimg(width=800, height=1100, text="MISSING PAGE"):
        """Get missing page imgage."""
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        # Create a white background image
        image = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(image)

        # Try to use a standard font or fall back to default
        try:
            # For larger installations, you might have Arial or other fonts
            font = ImageFont.truetype("arial.ttf", size=60)
        except IOError:
            # Fall back to default font
            font = ImageFont.load_default().font_variant(size=60)

        # Get text size to center it
        text_width, text_height = (
            draw.textsize(text, font=font)
            if hasattr(draw, "textsize")
            else (draw.textlength(text, font=font), font.size)
        )

        # Position for the text (centered and angled)
        position = ((width - text_width) // 2, (height - text_height) // 2)

        # Draw the watermark text (light gray and rotated)
        draw.text(position, text, fill=(200, 200, 200), font=font)

        # Rotate the image 45 degrees to create diagonal watermark effect
        image = image.rotate(45, expand=False, fillcolor="white")

        return image

    true_page_imgs = true_doc.get_visualization(show_label=False)
    pred_page_imgs = pred_doc.get_visualization(show_label=False)

    true_page_nos = true_page_imgs.keys()
    pred_page_nos = pred_page_imgs.keys()

    if true_page_nos != pred_page_nos:
        logging.error(
            f"incompatible true_page_nos versus pred_page_nos: \ntrue_page_nos: {true_page_nos}\npred_page_nos: {pred_page_nos}"
        )

    page_nos = true_page_nos | pred_page_nos

    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        HTML_DEFAULT_HEAD_FOR_COMP,
        "<body>",
    ]

    html_parts.append("<table>")
    html_parts.append("<tbody>")

    # Compile a regular expression pattern to match content within <body> tags
    pattern = re.compile(
        r"<body[^>]*>\n<div class='page'>(.*?)</div>\n</body>",
        re.DOTALL | re.IGNORECASE,
    )

    for page_no in page_nos:

        if page_no in true_page_imgs:
            true_doc_img_b64 = from_pil_to_base64(true_page_imgs[page_no])
        else:
            logging.error(f"{page_no} not in true_page_imgs, get default image.")
            true_doc_img_b64 = from_pil_to_base64(get_missing_pageimg())

        if page_no in pred_page_imgs:
            pred_doc_img_b64 = from_pil_to_base64(pred_page_imgs[page_no])
        else:
            logging.error(f"{page_no} not in pred_page_imgs, get default image.")
            pred_doc_img_b64 = from_pil_to_base64(get_missing_pageimg())

        true_doc_page = true_doc.export_to_html(
            image_mode=ImageRefMode.EMBEDDED, page_no=page_no
        )
        pred_doc_page = pred_doc.export_to_html(
            image_mode=ImageRefMode.EMBEDDED, page_no=page_no
        )

        # Search for the pattern in the HTML string
        mtch = pattern.search(true_doc_page)
        if mtch:
            true_doc_page_body = mtch.group(1).strip()
        else:
            logging.error(f"could not find body in true_doc_page")
            true_doc_page_body = "<p>Nothing Found</p>"

        # Search for the pattern in the HTML string
        mtch = pattern.search(pred_doc_page)
        if mtch:
            pred_doc_page_body = mtch.group(1).strip()
        else:
            logging.error(f"could not find body in pred_doc_page")
            pred_doc_page_body = "<p>Nothing Found</p>"

        if len(true_doc_page_body) == 0:
            true_doc_page_body = "<p>Nothing Found</p>"

        if len(pred_doc_page_body) == 0:
            pred_doc_page_body = "<p>Nothing Found</p>"

        html_parts.append("<tr>")

        html_parts.append("<td>")
        html_parts.append(f'<img src="data:image/png;base64,{true_doc_img_b64}">')
        html_parts.append("</td>")

        html_parts.append("<td>")
        html_parts.append(f"<div class='page'>\n{true_doc_page_body}\n</div>")
        html_parts.append("</td>")

        html_parts.append("<td>")
        html_parts.append(f'<img src="data:image/png;base64,{pred_doc_img_b64}">')
        html_parts.append("</td>")

        html_parts.append("<td>")
        html_parts.append(f"<div class='page'>\n{pred_doc_page_body}\n</div>")
        html_parts.append("</td>")

        html_parts.append("</tr>")

    html_parts.append("</tbody>")
    html_parts.append("</table>")

    # Close HTML structure
    html_parts.extend(["</body>", "</html>"])

    # Join with newlines
    html_content = "\n".join(html_parts)

    with open(str(filename), "w") as fw:
        fw.write(html_content)


def save_inspection_html(
    filename: Path, doc: DoclingDocument, labels: Set[DocItemLabel]
):

    html_doc = doc.export_to_html(image_mode=ImageRefMode.EMBEDDED, labels=labels)
    html_doc = html_doc.replace("'", "&#39;")

    page_images = []
    page_template = '<div class="image-wrapper"><img src="data:image/png;base64,BASE64PAGE" alt="Example Image"></div>'
    for page_no, page in doc.pages.items():
        # page_img = page.image.pil_image

        if page.image is not None and page.image.pil_image is not None:

            page_img = draw_clusters_with_reading_order(
                doc=doc,
                page_image=page.image.pil_image,
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
