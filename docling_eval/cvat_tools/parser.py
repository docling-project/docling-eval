import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.document import ContentLayer
from docling_core.types.doc.labels import DocItemLabel, GraphCellLabel

from docling_eval.cvat_tools.models import (
    CVATAnnotationPath,
    CVATElement,
    CVATImageInfo,
)

logger = logging.getLogger("docling_eval.cvat_tools.")


def cvat_box_to_bbox(xtl: float, ytl: float, xbr: float, ybr: float) -> BoundingBox:
    """Convert CVAT box coordinates to BoundingBox (TOPLEFT origin)."""
    return BoundingBox(l=xtl, t=ytl, r=xbr, b=ybr, coord_origin=CoordOrigin.TOPLEFT)


def get_all_images_from_cvat_xml(xml_path: Path) -> List[str]:
    """Get all image names from a CVAT XML file.

    Args:
        xml_path: Path to the CVAT XML file

    Returns:
        List of image names found in the XML file
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_names = []
    for img in root.findall(".//image"):
        image_names.append(img.attrib.get("name", ""))

    return image_names


class MissingImageInCVATXML(Exception):
    """Raised when an image is not found in the CVAT XML annotation file."""

    pass


def find_samples_in_directory(root_dir: Path) -> List[Tuple[str, Path, str]]:
    """Find all image files and their corresponding annotations.xml in the root directory."""
    samples = []
    for dirpath, _, filenames in os.walk(root_dir):
        images = [
            f
            for f in filenames
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ]
        if not images:
            continue

        xml_path = Path(dirpath) / "annotations.xml"
        if not xml_path.exists():
            continue

        for img in images:
            samples.append((img, xml_path, img))

    return samples


def _parse_image_element(
    image_el: ET.Element, box_id_start: int = 0, path_id_start: int = 0
) -> tuple[list[CVATElement], list[CVATAnnotationPath], CVATImageInfo]:
    """
    Parse a single <image> element and extract elements and paths.
    Returns (elements, paths, image_info).
    """
    image_info = CVATImageInfo(
        width=float(image_el.attrib["width"]),
        height=float(image_el.attrib["height"]),
        name=image_el.attrib["name"],
    )
    elements = []
    paths = []
    box_id = box_id_start
    path_id = path_id_start
    for box in image_el.findall("box"):
        label_str = box.attrib["label"]
        try:
            label = DocItemLabel(label_str)
        except ValueError:
            try:
                label = GraphCellLabel(label_str)  # type: ignore
            except ValueError:
                # Skip invalid labels
                logger.debug(f"Skipping invalid label: {label_str}")
                continue
        xtl = float(box.attrib["xtl"])
        ytl = float(box.attrib["ytl"])
        xbr = float(box.attrib["xbr"])
        ybr = float(box.attrib["ybr"])
        bbox = cvat_box_to_bbox(xtl, ytl, xbr, ybr)
        attributes = {}
        content_layer = None
        type_ = None
        level = None
        for attr in box.findall("attribute"):
            name = attr.attrib["name"]
            value = attr.text.strip() if attr.text else None
            attributes[name] = value
            if name == "content_layer" and value is not None:
                try:
                    content_layer = ContentLayer(value)
                except Exception:
                    content_layer = ContentLayer.BODY
            elif name == "type":
                type_ = value
            elif name == "level":
                if value is not None:
                    try:
                        level = int(value)
                    except Exception:
                        level = None
        if content_layer is None:
            content_layer = ContentLayer.BODY
        elements.append(
            CVATElement(
                id=box_id,
                label=label,
                bbox=bbox,
                content_layer=content_layer,
                type=type_,
                level=level,
                attributes=attributes,
            )
        )
        box_id += 1
    for poly in image_el.findall("polyline"):
        poly_label = poly.attrib["label"]
        points_str = poly.attrib["points"]
        points = [tuple(map(float, pt.split(","))) for pt in points_str.split(";")]
        attributes = {}
        level = None
        for attr in poly.findall("attribute"):
            name = attr.attrib["name"]
            value = attr.text.strip() if attr.text else None
            attributes[name] = value
            if name == "level" and value is not None:
                try:
                    level = int(value)
                except Exception:
                    level = None
        paths.append(
            CVATAnnotationPath(
                id=path_id,
                label=poly_label,
                points=points,
                level=level,
                attributes=attributes,
            )
        )
        path_id += 1
    return elements, paths, image_info


def parse_cvat_xml(
    xml_path: Path, image_name: Optional[str] = None
) -> dict[str, tuple[list[CVATElement], list[CVATAnnotationPath], CVATImageInfo]]:
    """
    Parse a CVAT annotations.xml file and extract elements and paths for all images or a specific image.
    Returns a dict mapping image name to (elements, paths, image_info).
    If image_name is given, only that image is returned (as a single-item dict).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    result = {}
    if image_name is not None:
        image_el = root.find(f".//image[@name='{image_name}']")
        if image_el is None:
            raise MissingImageInCVATXML(
                f"No <image> element for {image_name} in {xml_path}"
            )
        elements, paths, image_info = _parse_image_element(image_el)
        result[image_name] = (elements, paths, image_info)
    else:
        for image_el in root.findall(".//image"):
            name = image_el.attrib["name"]
            elements, paths, image_info = _parse_image_element(image_el)
            result[name] = (elements, paths, image_info)
    return result
