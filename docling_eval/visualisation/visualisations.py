import copy
import logging
import re
from pathlib import Path
from typing import Set

from docling.datamodel.base_models import BoundingBox, Cluster
from docling.utils.visualization import draw_clusters
from docling_core.transforms.serializer.html import (
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
    HTML_DEFAULT_HEAD_FOR_COMP_v2,
)


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
        HTML_DEFAULT_HEAD_FOR_COMP_v2,
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
