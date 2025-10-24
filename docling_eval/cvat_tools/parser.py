import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Tuple

from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.document import ContentLayer
from docling_core.types.doc.labels import DocItemLabel, GraphCellLabel

from docling_eval.cvat_tools.models import (
    CVATAnnotationPath,
    CVATElement,
    CVATImageInfo,
    TableStructLabel,
)

logger = logging.getLogger("docling_eval.cvat_tools.")

MANUAL_LABEL_MAP: dict[str, DocItemLabel] = {
    "fillable_field": DocItemLabel.EMPTY_VALUE,
}


@dataclass(frozen=True)
class ParsedCVATImage:
    """Parsed representation of a single <image> entry in a CVAT XML file."""

    name: str
    elements: Tuple[CVATElement, ...]
    paths: Tuple[CVATAnnotationPath, ...]
    image_info: CVATImageInfo


@dataclass(frozen=True)
class ParsedCVATFile:
    """Parsed CVAT XML file containing multiple images."""

    xml_path: Path
    images: Mapping[str, ParsedCVATImage]

    def get_image(self, image_name: str) -> ParsedCVATImage:
        try:
            return self.images[image_name]
        except KeyError as exc:
            raise MissingImageInCVATXML(
                f"No <image> element for {image_name} in {self.xml_path}"
            ) from exc

    @property
    def image_names(self) -> Tuple[str, ...]:
        return tuple(self.images.keys())


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
    parsed_file = parse_cvat_file(xml_path)
    return list(parsed_file.image_names)


class MissingImageInCVATXML(Exception):
    """Raised when an image is not found in the CVAT XML annotation file."""

    pass


def find_samples_in_directory(root_dir: Path) -> List[Tuple[str, Path, str]]:
    """Discover CVAT samples in ``root_dir``.

    This first looks for structured CVAT exports containing ``cvat_overview.json`` and
    collects their page/image information. When no such exports are found, it
    falls back to scanning for ``annotations.xml`` files accompanied by images.
    """

    structured_samples: List[Tuple[str, Path, str]] = []
    processed: set[Path] = set()

    for overview_path in root_dir.rglob("cvat_overview.json"):
        folder_path = overview_path.parent
        folder_key = folder_path.resolve()
        if folder_key in processed:
            continue
        processed.add(folder_key)

        try:
            from .folder_parser import parse_cvat_folder

            folder_structure = parse_cvat_folder(folder_path)
        except Exception:
            continue

        for document in folder_structure.documents.values():
            for page in document.pages:
                structured_samples.append(
                    (page.image_filename, page.xml_path, page.image_filename)
                )

    if structured_samples:
        return structured_samples

    samples: List[Tuple[str, Path, str]] = []
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
    # Local import to avoid touching the module-level imports if you prefer
    from docling_eval.cvat_tools.models import TableStructLabel

    image_info = CVATImageInfo(
        width=float(image_el.attrib["width"]),
        height=float(image_el.attrib["height"]),
        name=image_el.attrib["name"],
    )

    elements: list[CVATElement] = []
    paths: list[CVATAnnotationPath] = []
    box_id = box_id_start
    path_id = path_id_start

    # ---- parse <box> (rectangles) with strict labels (DocItemLabel | GraphCellLabel | TableStructLabel)
    for box in image_el.findall("box"):
        label_str = box.attrib["label"]

        # Parse into one of the known enums; skip if unknown
        label_obj: Optional[object] = None
        normalized_label = label_str.lower().replace(" ", "_")

        manual_label = MANUAL_LABEL_MAP.get(normalized_label)
        if manual_label is None:
            manual_label = MANUAL_LABEL_MAP.get(label_str)
        if manual_label is not None:
            label_obj = manual_label
        else:
            try:
                label_obj = DocItemLabel(normalized_label)
            except ValueError:
                try:
                    label_obj = GraphCellLabel(label_str)  # type: ignore[assignment]
                except ValueError:
                    try:
                        label_obj = TableStructLabel(label_str)  # type: ignore[assignment]
                    except ValueError:
                        logger.debug(f"Skipping invalid label: {label_str}")
                        continue

        xtl = float(box.attrib["xtl"])
        ytl = float(box.attrib["ytl"])
        xbr = float(box.attrib["xbr"])
        ybr = float(box.attrib["ybr"])
        bbox = cvat_box_to_bbox(xtl, ytl, xbr, ybr)  # -> BoundingBox(l,t,r,b) TOPLEFT

        # Parse child <attribute> tags; default content_layer to BODY
        attributes: dict[str, str | None] = {}
        content_layer = None
        type_: Optional[str] = None
        level: Optional[int] = None

        for attr in box.findall("attribute"):
            name = attr.attrib["name"]
            value = attr.text.strip() if attr.text else None
            attributes[name] = value
            if name == "content_layer" and value is not None:
                try:
                    content_layer = ContentLayer(value.lower())
                except Exception:
                    content_layer = ContentLayer.BODY
            elif name == "type":
                type_ = value
            elif name == "level" and value is not None:
                try:
                    level = int(value)
                except Exception:
                    level = None

        if content_layer is None:
            content_layer = ContentLayer.BODY

        elements.append(
            CVATElement(
                id=box_id,
                label=label_obj,  # Union[DocItemLabel, GraphCellLabel, TableStructLabel]
                bbox=bbox,
                content_layer=content_layer,
                type=type_,
                level=level,
                attributes=attributes,
            )
        )
        box_id += 1

    # ---- parse <polyline> (paths)
    for poly in image_el.findall("polyline"):
        poly_label = poly.attrib["label"]
        points_str = poly.attrib["points"]
        points = [tuple(map(float, pt.split(","))) for pt in points_str.split(";")]

        attributes: dict[str, str | None] = {}  # type: ignore
        level: Optional[int] = None  # type: ignore
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


def parse_cvat_file(xml_path: Path) -> ParsedCVATFile:
    """Parse ``xml_path`` once and return all contained images."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    images: Dict[str, ParsedCVATImage] = {}
    for image_el in root.findall(".//image"):
        name = image_el.attrib["name"]
        elements, paths, image_info = _parse_image_element(image_el)
        images[name] = ParsedCVATImage(
            name=name,
            elements=tuple(elements),
            paths=tuple(paths),
            image_info=image_info,
        )

    return ParsedCVATFile(xml_path=xml_path, images=MappingProxyType(images))
