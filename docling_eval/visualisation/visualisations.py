import copy
import logging
import re
from pathlib import Path
from typing import Optional, Set

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


def get_missing_pageimg(
    width: int = 800, height: int = 1100, text: str = "MISSING PAGE"
) -> Image.Image:
    """Get missing page image.

    Args:
        width: Image width
        height: Image height
        text: Text to display on the image

    Returns:
        PIL Image with the missing page text
    """
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
        font = ImageFont.load_default().font_variant(size=60)  # type: ignore

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


def _get_document_visualization_data(
    doc: DoclingDocument,
    page_no: int,
    pattern: re.Pattern,
) -> tuple[str, str]:
    """Get visualization data for a document page.

    Args:
        doc: Document to visualize
        page_no: Page number to visualize
        pattern: Regex pattern to extract body content

    Returns:
        Tuple of (base64_image, html_content)
    """
    page_imgs = doc.get_visualization(show_label=False)

    if page_no in page_imgs:
        doc_img_b64 = from_pil_to_base64(page_imgs[page_no])
    else:
        logging.error(f"{page_no} not in page_imgs, get default image.")
        doc_img_b64 = from_pil_to_base64(get_missing_pageimg())

    doc_page = doc.export_to_html(image_mode=ImageRefMode.EMBEDDED, page_no=page_no)

    # Search for the pattern in the HTML string
    mtch = pattern.search(doc_page)
    if mtch:
        doc_page_body = mtch.group(1).strip()
    else:
        logging.error(f"could not find body in doc_page")
        doc_page_body = "<p>Nothing Found</p>"

    if len(doc_page_body) == 0:
        doc_page_body = "<p>Nothing Found</p>"

    return doc_img_b64, doc_page_body


def _create_visualization_html(
    filename: Path,
    true_doc: DoclingDocument,
    template: str,
    pred_doc: Optional[DoclingDocument] = None,
    draw_reading_order: bool = True,
) -> None:
    """Create and save HTML visualization.

    Args:
        filename: Path to save the visualization
        true_doc: Document to visualize
        template: HTML template to use
        pred_doc: Optional predicted document for comparison
        draw_reading_order: Whether to draw reading order
    """
    # Compile regex pattern once
    pattern = re.compile(
        r"<body[^>]*>\n<div class='page'>(.*?)</div>\n</body>",
        re.DOTALL | re.IGNORECASE,
    )

    # Add CSS class based on view type
    view_class = "comparison-view" if pred_doc is not None else "single-view"
    template = template.replace(
        "<style>",
        f"<style>\n.{view_class} td {{ width: {'25%' if pred_doc is not None else '50%'}; }}",
    )

    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        template,
        "<body>",
    ]

    html_parts.append(f"<table class='{view_class}'>")
    html_parts.append("<tbody>")

    # Get page numbers and convert to set of integers
    true_page_nos = {
        k for k in true_doc.get_visualization(show_label=False).keys() if k is not None
    }
    if pred_doc is not None:
        pred_page_nos = {
            k
            for k in pred_doc.get_visualization(show_label=False).keys()
            if k is not None
        }
        if true_page_nos != pred_page_nos:
            logging.error(
                f"incompatible page numbers: \n"
                f"true pages: {true_page_nos}\npred pages: {pred_page_nos}"
            )
        page_nos = true_page_nos | pred_page_nos
    else:
        page_nos = true_page_nos

    # Process each page
    for page_no in page_nos:
        html_parts.append("<tr>")

        # Process true document
        true_img_b64, true_html = _get_document_visualization_data(
            true_doc, page_no, pattern
        )
        html_parts.append("<td>")
        html_parts.append(f'<img src="data:image/png;base64,{true_img_b64}">')
        html_parts.append("</td>")
        html_parts.append("<td>")
        html_parts.append(f"<div class='page'>\n{true_html}\n</div>")
        html_parts.append("</td>")

        # Process predicted document if present
        if pred_doc is not None:
            pred_img_b64, pred_html = _get_document_visualization_data(
                pred_doc, page_no, pattern
            )
            html_parts.append("<td>")
            html_parts.append(f'<img src="data:image/png;base64,{pred_img_b64}">')
            html_parts.append("</td>")
            html_parts.append("<td>")
            html_parts.append(f"<div class='page'>\n{pred_html}\n</div>")
            html_parts.append("</td>")

        html_parts.append("</tr>")

    html_parts.append("</tbody>")
    html_parts.append("</table>")
    html_parts.extend(["</body>", "</html>"])

    with open(str(filename), "w") as fw:
        fw.write("\n".join(html_parts))


def save_single_document_html(
    filename: Path,
    doc: DoclingDocument,
    draw_reading_order: bool = True,
) -> None:
    """Save single document visualization with its HTML content.

    Args:
        filename: Path to save the visualization
        doc: Document to visualize
        draw_reading_order: Whether to draw reading order
    """
    _create_visualization_html(
        filename=filename,
        true_doc=doc,
        template=HTML_DEFAULT_HEAD_FOR_COMP_v2,
        draw_reading_order=draw_reading_order,
    )


def save_comparison_html_with_clusters(
    filename: Path,
    true_doc: DoclingDocument,
    pred_doc: DoclingDocument,
    draw_reading_order: bool = True,
) -> None:
    """Save comparison html with clusters.

    Args:
        filename: Path to save the visualization
        true_doc: Ground truth document
        pred_doc: Predicted document
        draw_reading_order: Whether to draw reading order
    """
    _create_visualization_html(
        filename=filename,
        true_doc=true_doc,
        template=HTML_DEFAULT_HEAD_FOR_COMP_v2,
        pred_doc=pred_doc,
        draw_reading_order=draw_reading_order,
    )
