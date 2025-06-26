import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.document import ContentLayer

from docling_eval.cvat_tools.models import (
    CVATAnnotationPath,
    CVATElement,
    CVATImageInfo,
)


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


def parse_cvat_xml_for_image(
    xml_path: Path, image_filename: str
) -> Tuple[List[CVATElement], List[CVATAnnotationPath], CVATImageInfo]:
    """Parse a CVAT XML file for a specific image and return elements, paths, and image info."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find the matching image element
    image_el = None
    image_filename_no_ext = Path(image_filename).stem

    matching_images = []
    for img in root.findall(".//image"):
        if image_filename_no_ext in img.attrib.get("name", ""):
            matching_images.append(img)

    if len(matching_images) > 1:
        raise ValueError(
            f"Multiple matching images found for {image_filename} in {xml_path}"
        )
    elif len(matching_images) == 1:
        image_el = matching_images[0]
    else:
        raise ValueError(f"No <image> element for {image_filename} in {xml_path}")

    # Parse image info
    image_info = CVATImageInfo(
        width=float(image_el.attrib["width"]),
        height=float(image_el.attrib["height"]),
        name=image_el.attrib["name"],
    )

    # Parse elements
    elements: List[CVATElement] = []
    box_id = 0
    for box in image_el.findall("box"):
        try:
            label = box.attrib["label"]
            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])
            bbox = cvat_box_to_bbox(xtl, ytl, xbr, ybr)

            # Parse attributes
            attributes = {}
            content_layer = ContentLayer.BODY  # Default
            type_ = None
            level = None

            for attr in box.findall("attribute"):
                name = attr.attrib["name"]
                value = attr.text.strip() if attr.text else None
                attributes[name] = value

                if name == "content_layer" and value is not None:
                    content_layer = ContentLayer(value.lower())
                elif name == "type":
                    type_ = value
                elif name == "level" and value is not None:
                    try:
                        level = int(value)
                    except (ValueError, TypeError):
                        level = None

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

        except (ValueError, KeyError) as e:
            # Skip invalid elements
            continue

    # Parse paths
    paths: List[CVATAnnotationPath] = []
    path_id = 0
    for poly in image_el.findall("polyline"):
        label = poly.attrib["label"]
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
                except (ValueError, TypeError):
                    level = None

        paths.append(
            CVATAnnotationPath(
                id=path_id,
                label=label,
                points=points,
                level=level,
                attributes=attributes,
            )
        )
        path_id += 1

    return elements, paths, image_info


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
