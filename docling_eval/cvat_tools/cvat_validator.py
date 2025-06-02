"""
CVAT Annotation Validation Tool

- Parses CVAT XML annotation files and corresponding images
- Builds containment trees and parses reading-order/group/merge paths
- Validates annotation structure according to project rules
- Outputs element tree and validation report per sample

Usage:
    python -m docling_eval.cvat_validator <input_root_dir>
"""

# 1. Imports & dependencies
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, List, Optional, Tuple

from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.labels import DocItemLabel

# 2. Pydantic models for CVAT elements/paths
# --------------------------------------------------
from pydantic import BaseModel, Field

# External dependencies (to be used):
# - lxml or xml.etree.ElementTree for XML parsing
# - pydantic v2 for data models
# - rtree for spatial queries


class Element(BaseModel):
    """A rectangle element (box) in CVAT annotation, using BoundingBox from docling_core."""

    id: int
    label: DocItemLabel
    bbox: BoundingBox
    content_layer: str
    type: Optional[str] = None
    level: Optional[int] = None
    attributes: dict[str, Any] = Field(default_factory=dict)


# Helper to convert CVAT box attributes to BoundingBox
# (CVAT uses xtl, ytl, xbr, ybr, which are top-left origin)
def cvat_box_to_bbox(xtl: float, ytl: float, xbr: float, ybr: float) -> BoundingBox:
    """Convert CVAT box coordinates to BoundingBox (TOPLEFT origin)."""
    return BoundingBox(l=xtl, t=ytl, r=xbr, b=ybr, coord_origin=CoordOrigin.TOPLEFT)


class CVATAnnotationPath(BaseModel):
    """A polyline path in CVAT annotation (reading-order, merge, group, etc)."""

    id: int
    label: str
    points: list[tuple[float, float]]
    level: Optional[int] = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class ValidationError(BaseModel):
    """Validation error for reporting issues in annotation."""

    error_type: str
    message: str
    element_id: Optional[int] = None
    path_id: Optional[int] = None


# 3. XML parsing functions
# --------------------------------------------------
# Functions to parse CVAT XML files and extract elements/paths into models


def parse_cvat_xml(xml_path: Path) -> tuple[list[Element], list[Path], dict]:
    """
    Parse a CVAT annotations.xml file and extract elements and paths.
    Returns (elements, paths, image_info_dict).
    image_info_dict contains width, height, and filename for the image.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find the <image> element (assume one per file)
    image_el = root.find(".//image")
    if image_el is None:
        raise ValueError("No <image> element found in CVAT XML.")
    image_info = {
        "width": float(image_el.attrib["width"]),
        "height": float(image_el.attrib["height"]),
        "name": image_el.attrib["name"],
    }

    elements = []
    paths = []
    box_id = 0
    path_id = 0

    for box in image_el.findall("box"):
        label = box.attrib["label"]
        xtl = float(box.attrib["xtl"])
        ytl = float(box.attrib["ytl"])
        xbr = float(box.attrib["xbr"])
        ybr = float(box.attrib["ybr"])
        bbox = cvat_box_to_bbox(xtl, ytl, xbr, ybr)
        # Parse attributes
        attributes = {}
        content_layer = None
        type_ = None
        level = None
        for attr in box.findall("attribute"):
            name = attr.attrib["name"]
            value = attr.text.strip() if attr.text else None
            attributes[name] = value
            if name == "content_layer":
                content_layer = value
            elif name == "type":
                type_ = value
            elif name == "level":
                try:
                    level = int(value)
                except Exception:
                    level = None
        # Fallback if content_layer not found
        if content_layer is None:
            content_layer = "BODY"  # or raise error?
        elements.append(
            Element(
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
        label = poly.attrib["label"]
        points_str = poly.attrib["points"]
        points = [tuple(map(float, pt.split(","))) for pt in points_str.split(";")]
        attributes = {}
        level = None
        for attr in poly.findall("attribute"):
            name = attr.attrib["name"]
            value = attr.text.strip() if attr.text else None
            attributes[name] = value
            if name == "level":
                try:
                    level = int(value)
                except Exception:
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


# Test function for parsing and printing the tree
# --------------------------------------------------
def test_parse_and_print_elements_and_paths(xml_path: Path):
    """
    Test function: parse the XML and print a simple tree of elements and paths.
    Useful for debugging the parsing step before structural analysis.
    """
    elements, paths, image_info = parse_cvat_xml(xml_path)
    print(
        f"Image: {image_info['name']} ({image_info['width']}x{image_info['height']})\n"
    )
    print("Elements:")
    for el in elements:
        print(
            f"  [Element {el.id}] label={el.label} bbox=({el.bbox.l:.1f},{el.bbox.t:.1f},{el.bbox.r:.1f},{el.bbox.b:.1f}) layer={el.content_layer} type={el.type} level={el.level}"
        )
    print("\nPaths:")
    for path in paths:
        print(
            f"  [Path {path.id}] label={path.label} level={path.level} points={len(path.points)}"
        )


# Test function for parsing and printing the containment tree (for debugging, not CLI)
# --------------------------------------------------
def test_parse_and_print_containment(xml_path: Path):
    """
    Test function: parse the XML, build the containment tree, and print it indented.
    Useful for verifying containment logic. Not part of the CLI.
    """
    elements, _, image_info = parse_cvat_xml(xml_path)
    roots = build_containment_tree(elements)
    print(
        f"Containment tree for {image_info['name']} ({image_info['width']}x{image_info['height']}):\n"
    )

    def print_tree(node, indent=0):
        el = node.element
        print(
            "  " * indent
            + f"[{el.label} id={el.id}] bbox=({el.bbox.l:.1f},{el.bbox.t:.1f},{el.bbox.r:.1f},{el.bbox.b:.1f})"
        )
        for child in node.children:
            print_tree(child, indent + 1)

    for root in roots:
        print_tree(root)


# 4. Containment tree builder
# --------------------------------------------------
from typing import List, Optional


class TreeNode:
    """
    Node in the containment tree. Holds an Element, parent, and children.
    """

    def __init__(self, element: Element):
        self.element = element
        self.parent: Optional["TreeNode"] = None
        self.children: List["TreeNode"] = []

    def add_child(self, child: "TreeNode"):
        self.children.append(child)
        child.parent = self

    def __repr__(self):
        return f"TreeNode({self.element.label}, id={self.element.id})"


def contains(parent: BoundingBox, child: BoundingBox, iou_thresh=0.99) -> bool:
    # Parent contains child if intersection over child area is very high
    intersection = parent.intersection_area_with(child)
    return intersection / (child.area() + 1e-6) > iou_thresh


def build_containment_tree(elements: List[Element]) -> List[TreeNode]:
    """
    Build a containment tree from elements based on spatial containment and content_layer.
    Returns a list of root TreeNodes (elements with no parent).
    """
    nodes = [TreeNode(el) for el in elements]
    for i, node in enumerate(nodes):
        best_parent = None
        best_area = None
        for j, candidate in enumerate(nodes):
            if i == j:
                continue
            if node.element.content_layer != candidate.element.content_layer:
                continue
            if contains(candidate.element.bbox, node.element.bbox):
                area = candidate.element.bbox.area()
                if best_area is None or area < best_area:
                    best_parent = candidate
                    best_area = area
        if best_parent:
            best_parent.add_child(node)
    roots = [node for node in nodes if node.parent is None]
    return roots


import math

# 5. Path parsing logic
# --------------------------------------------------
from collections import defaultdict


def map_path_points_to_elements(
    paths: List[Path], elements: List[Element], proximity_thresh: float = 5.0
) -> dict:
    """
    For each path, map its control points to the DEEPEST elements they touch (by bbox containment or proximity).
    Returns: {path_id: [element_id, ...]}
    """
    path_to_elements = defaultdict(list)
    for path in paths:
        if not path.label.startswith("reading_order"):
            continue
        for pt in path.points:
            # Find all elements whose bbox contains the point
            candidates = []
            for el in elements:
                bbox = el.bbox
                if (
                    bbox.l - proximity_thresh <= pt[0] <= bbox.r + proximity_thresh
                    and bbox.t - proximity_thresh <= pt[1] <= bbox.b + proximity_thresh
                ):
                    candidates.append(el)
            if candidates:
                # Pick the deepest (smallest area) element
                deepest = min(candidates, key=lambda e: e.bbox.area())
                eid = deepest.id
                # Only add if not a duplicate in sequence
                if (
                    not path_to_elements[path.id]
                    or path_to_elements[path.id][-1] != eid
                ):
                    path_to_elements[path.id].append(eid)
    return dict(path_to_elements)


def find_node_by_element_id(
    tree_roots: List["TreeNode"], element_id: int
) -> Optional["TreeNode"]:
    """Find the TreeNode for a given element_id in the tree."""
    stack = list(tree_roots)
    while stack:
        node = stack.pop()
        if node.element.id == element_id:
            return node
        stack.extend(node.children)
    return None


def get_ancestors(node: "TreeNode") -> list:
    """Return a list of ancestors from closest to root."""
    ancestors = []
    while node.parent:
        node = node.parent
        ancestors.append(node)
    return ancestors


def closest_common_ancestor(nodes: List["TreeNode"]) -> Optional["TreeNode"]:
    """Find the closest common ancestor of a list of nodes."""
    if not nodes:
        return None
    ancestor_lists = [[node] + get_ancestors(node) for node in nodes]
    # Find the first common node in all ancestor lists
    for candidate in ancestor_lists[0]:
        if all(candidate in lst for lst in ancestor_lists[1:]):
            return candidate
    return None


def associate_reading_order_paths_to_containers(
    path_to_elements: dict, tree_roots: List["TreeNode"]
) -> dict:
    """
    For each reading-order path, associate it to the closest parent container that contains all touched elements.
    Returns: {path_id: container_node}
    """
    path_to_container = {}
    for path_id, el_ids in path_to_elements.items():
        touched_nodes = [find_node_by_element_id(tree_roots, eid) for eid in el_ids]
        touched_nodes = [n for n in touched_nodes if n is not None]
        if not touched_nodes:
            continue
        ancestor = closest_common_ancestor(touched_nodes)
        if ancestor is not None:
            path_to_container[path_id] = ancestor
        else:
            # fallback: parent of first touched element
            path_to_container[path_id] = (
                touched_nodes[0].parent if touched_nodes[0] else None
            )
    return path_to_container


def build_global_reading_order(
    paths: List[Path],
    path_to_elements: dict,
    path_to_container: dict,
    tree_roots: List["TreeNode"],
) -> list:
    """
    Merge all reading-order paths into a single global reading order as described.
    Returns: [element_id, ...] in reading order.
    """
    # Find level-1 reading order path
    level1_path = next(
        (
            p
            for p in paths
            if p.label.startswith("reading_order") and (p.level == 1 or p.level is None)
        ),
        None,
    )
    if not level1_path:
        return []
    visited = set()
    result = []

    def insert_with_ancestors(eid, path_container_id):
        node = find_node_by_element_id(tree_roots, eid)
        # Collect ancestors up to (but not including) the path's associated container
        ancestors = []
        current = node.parent
        while current and current.element.id != path_container_id:
            ancestors.append(current)
            current = current.parent
        for ancestor in reversed(ancestors):
            if ancestor.element.id not in visited:
                result.append(ancestor.element.id)
                visited.add(ancestor.element.id)
        # Insert the element itself
        if eid not in visited:
            result.append(eid)
            visited.add(eid)

    def insert_path(path_id):
        path_container = path_to_container.get(path_id)
        path_container_id = path_container.element.id if path_container else None
        # Insert the path's associated container first (if not already visited)
        if path_container and path_container.element.id not in visited:
            result.append(path_container.element.id)
            visited.add(path_container.element.id)
        for eid in path_to_elements.get(path_id, []):
            if eid in visited:
                continue
            node = find_node_by_element_id(tree_roots, eid)
            # If this is a container with a level 2+ reading order path, insert those first
            container_paths = [
                pid
                for pid, cnode in path_to_container.items()
                if cnode and cnode.element.id == eid and pid != path_id
            ]
            for pid in container_paths:
                # Only insert if not already visited
                for sub_eid in path_to_elements.get(pid, []):
                    if sub_eid not in visited:
                        insert_path(pid)
            # Insert ancestors and the element itself
            insert_with_ancestors(eid, path_container_id)
            # Also insert all direct children of this container if they are touched by the path
            if node and node.children:
                child_ids = [child.element.id for child in node.children]
                for child_id in child_ids:
                    if (
                        child_id in path_to_elements.get(path_id, [])
                        and child_id not in visited
                    ):
                        insert_with_ancestors(child_id, path_container_id)

    insert_path(level1_path.id)
    return result


# Test function for printing the global reading order (for debugging, not CLI)
# --------------------------------------------------
def test_print_reading_ordered_tree(xml_path: Path):
    """
    Test function: parse XML, build containment tree, map reading order, and print the global reading order.
    Not part of the CLI.
    """
    elements, paths, image_info = parse_cvat_xml(xml_path)
    tree_roots = build_containment_tree(elements)
    path_to_elements = map_path_points_to_elements(paths, elements)
    path_to_container = associate_reading_order_paths_to_containers(
        path_to_elements, tree_roots
    )
    global_order = build_global_reading_order(
        paths, path_to_elements, path_to_container, tree_roots
    )
    print(f"Global reading order for {image_info['name']}:")
    for eid in global_order:
        node = find_node_by_element_id(tree_roots, eid)
        container = node.parent.element.label if node and node.parent else None
        print(
            f"  [Element {eid}] label={node.element.label if node else '?'} parent={container}"
        )


# 6. Validation logic
# --------------------------------------------------
# Functions to check all validation rules and collect errors


# Validation report models
class ValidationReport(BaseModel):
    sample_name: str
    errors: List[ValidationError]


class ValidationRunReport(BaseModel):
    samples: List[ValidationReport]


# Validation functions


def validate_valid_labels(elements: List[Element]) -> List[ValidationError]:
    errors = []
    if DocItemLabel is None:
        return errors
    valid_labels = set(item.value for item in DocItemLabel)
    for el in elements:
        if el.label not in valid_labels:
            errors.append(
                ValidationError(
                    error_type="invalid_label",
                    message=f"Element {el.id} has invalid label '{el.label}'",
                    element_id=el.id,
                )
            )
    return errors


def validate_first_level_reading_order(paths: List[Path]) -> List[ValidationError]:
    errors = []
    found = any(
        p.label.startswith("reading_order") and (p.level == 1 or p.level is None)
        for p in paths
    )
    if not found:
        errors.append(
            ValidationError(
                error_type="missing_first_level_reading_order",
                message="No first-level reading-order path found.",
            )
        )
    return errors


def validate_second_level_reading_order_has_parent(
    paths: List[Path], path_to_container: dict
) -> List[ValidationError]:
    errors = []
    for p in paths:
        if p.label.startswith("reading_order") and p.level and p.level > 1:
            container = path_to_container.get(p.id)
            if container is None or container.parent is None:
                errors.append(
                    ValidationError(
                        error_type="second_level_reading_order_no_parent",
                        message=f"Second-level reading-order path {p.id} has no parent container.",
                        path_id=p.id,
                    )
                )
    return errors


def validate_every_element_touched(
    elements: List[Element], path_to_elements: dict
) -> List[ValidationError]:
    errors = []
    touched = set()
    for elist in path_to_elements.values():
        touched.update(elist)
    for el in elements:
        if el.content_layer.upper() != "BACKGROUND" and el.id not in touched:
            errors.append(
                ValidationError(
                    error_type="element_not_touched_by_reading_order",
                    message=f"Element {el.id} ({el.label}) not touched by any reading-order path.",
                    element_id=el.id,
                )
            )
    return errors


# Validation orchestrator


def validate_sample(
    elements, paths, tree_roots, path_to_elements, path_to_container
) -> ValidationReport:
    errors = []
    errors.extend(validate_valid_labels(elements))
    errors.extend(validate_first_level_reading_order(paths))
    errors.extend(
        validate_second_level_reading_order_has_parent(paths, path_to_container)
    )
    errors.extend(validate_every_element_touched(elements, path_to_elements))
    return ValidationReport(sample_name="sample", errors=errors)


def validate_and_report(samples: List[tuple]) -> ValidationRunReport:
    reports = []
    for (
        sample_name,
        elements,
        paths,
        tree_roots,
        path_to_elements,
        path_to_container,
    ) in samples:
        report = validate_sample(
            elements, paths, tree_roots, path_to_elements, path_to_container
        )
        report.sample_name = sample_name
        reports.append(report)
    return ValidationRunReport(samples=reports)


# Test function: validate and print JSON report for a single file
# --------------------------------------------------
import json


def test_validate_and_print_report(xml_path: Path):
    """
    Test function: parse XML, build tree, compute reading order, validate, and print JSON report.
    Not part of the CLI.
    """
    elements, paths, image_info = parse_cvat_xml(xml_path)
    roots = build_containment_tree(elements)
    path_to_elements = map_path_points_to_elements(paths, elements)
    path_to_container = associate_reading_order_paths_to_containers(
        path_to_elements, roots
    )
    report = validate_and_report(
        [
            (
                image_info["name"],
                elements,
                paths,
                roots,
                path_to_elements,
                path_to_container,
            )
        ]
    )
    print(json.dumps(report.model_dump(), indent=2))


# 7. Reporting/output
# --------------------------------------------------
# Functions to print or write the element tree and validation report


# 8. Main function & CLI entrypoint
# --------------------------------------------------
def main():
    """Main CLI for CVAT validation: accepts a directory or XML file, and optional image name."""
    import argparse

    parser = argparse.ArgumentParser(description="CVAT annotation validation tool")
    parser.add_argument(
        "input_path", type=str, help="Path to root directory or annotations.xml file"
    )
    parser.add_argument(
        "--image-name",
        type=str,
        default=None,
        help="If input is XML, only process <image> with this name",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    samples = []
    if input_path.is_dir():
        for sample_name, xml_path, image_filename in find_samples_in_directory(
            input_path
        ):
            elements, paths, image_info = parse_cvat_xml_for_image(
                xml_path, image_filename
            )
            roots = build_containment_tree(elements)
            path_to_elements = map_path_points_to_elements(paths, elements)
            path_to_container = associate_reading_order_paths_to_containers(
                path_to_elements, roots
            )
            samples.append(
                (
                    sample_name,
                    elements,
                    paths,
                    roots,
                    path_to_elements,
                    path_to_container,
                )
            )
    else:
        tree = ET.parse(input_path)
        root = tree.getroot()
        if args.image_name:
            # Only process the specified image
            elements, paths, image_info = parse_cvat_xml_for_image(
                input_path, args.image_name
            )
            roots = build_containment_tree(elements)
            path_to_elements = map_path_points_to_elements(paths, elements)
            path_to_container = associate_reading_order_paths_to_containers(
                path_to_elements, roots
            )
            samples.append(
                (
                    args.image_name,
                    elements,
                    paths,
                    roots,
                    path_to_elements,
                    path_to_container,
                )
            )
        else:
            # Process all <image> elements
            for image_el in root.findall(".//image"):
                image_filename = image_el.attrib["name"]
                elements, paths, image_info = parse_cvat_xml_for_image(
                    input_path, image_filename
                )
                roots = build_containment_tree(elements)
                path_to_elements = map_path_points_to_elements(paths, elements)
                path_to_container = associate_reading_order_paths_to_containers(
                    path_to_elements, roots
                )
                samples.append(
                    (
                        image_filename,
                        elements,
                        paths,
                        roots,
                        path_to_elements,
                        path_to_container,
                    )
                )
    report = validate_and_report(samples)
    print(json.dumps(report.model_dump(), indent=2))


# Utility: Apply reading order to the tree
# --------------------------------------------------
def apply_reading_order_to_tree(tree_roots, global_order):
    """
    Reorder the children of each container node in the tree to match the global reading order.
    This does not modify the global_order or the tree structure, only the order of children lists.
    """

    def collect_all_nodes(roots):
        stack = list(roots)
        all_nodes = []
        while stack:
            node = stack.pop()
            all_nodes.append(node)
            stack.extend(node.children)
        return all_nodes

    id_to_node = {node.element.id: node for node in collect_all_nodes(tree_roots)}

    for node in id_to_node.values():
        if node.children:
            # Only keep children that are in global_order, in the right order
            ordered_children = [
                id_to_node[child_id]
                for child_id in global_order
                if any(child.element.id == child_id for child in node.children)
            ]
            # Optionally, append any children not in global_order
            remaining = [
                child for child in node.children if child not in ordered_children
            ]
            node.children = ordered_children + remaining


# Test function: print containment tree after applying reading order
# --------------------------------------------------
def test_parse_and_print_containment_with_reading_order(xml_path: Path):
    """
    Test function: parse the XML, build the containment tree, compute and apply reading order, and print the tree indented.
    Useful for verifying containment and reading order logic. Not part of the CLI.
    """
    elements, paths, image_info = parse_cvat_xml(xml_path)
    roots = build_containment_tree(elements)
    path_to_elements = map_path_points_to_elements(paths, elements)
    path_to_container = associate_reading_order_paths_to_containers(
        path_to_elements, roots
    )
    global_order = build_global_reading_order(
        paths, path_to_elements, path_to_container, roots
    )
    apply_reading_order_to_tree(roots, global_order)
    print(
        f"Containment tree (reading order applied) for {image_info['name']} ({image_info['width']}x{image_info['height']}):\n"
    )

    def print_tree(node, indent=0):
        el = node.element
        print(
            "  " * indent
            + f"[{el.label} id={el.id}] bbox=({el.bbox.l:.1f},{el.bbox.t:.1f},{el.bbox.r:.1f},{el.bbox.b:.1f})"
        )
        for child in node.children:
            print_tree(child, indent + 1)

    for root in roots:
        print_tree(root)


# Utility: Find all samples in a directory
# --------------------------------------------------
def find_samples_in_directory(root_dir: Path) -> List[Tuple[str, Path, str]]:
    """
    Find all image files and their corresponding annotations.xml in the root directory (recursively).
    For each image, return (sample_name, xml_path, image_filename).
    """
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


# Utility: For a given xml_path and image_filename, parse only the matching <image> element
# --------------------------------------------------
def parse_cvat_xml_for_image(xml_path: Path, image_filename: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    image_el = None
    for img in root.findall(".//image"):
        if img.attrib.get("name") == image_filename:
            image_el = img
            break
    if image_el is None:
        raise ValueError(f"No <image> element for {image_filename} in {xml_path}")
    image_info = {
        "width": float(image_el.attrib["width"]),
        "height": float(image_el.attrib["height"]),
        "name": image_el.attrib["name"],
    }
    elements = []
    paths = []
    box_id = 0
    path_id = 0
    for box in image_el.findall("box"):
        label = box.attrib["label"]
        try:
            label = DocItemLabel(label)
        except ValueError:
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
            if name == "content_layer":
                content_layer = value
            elif name == "type":
                type_ = value
            elif name == "level":
                try:
                    level = int(value)
                except Exception:
                    level = None
        if content_layer is None:
            content_layer = "BODY"
        elements.append(
            Element(
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
        label = poly.attrib["label"]
        points_str = poly.attrib["points"]
        points = [tuple(map(float, pt.split(","))) for pt in points_str.split(";")]
        attributes = {}
        level = None
        for attr in poly.findall("attribute"):
            name = attr.attrib["name"]
            value = attr.text.strip() if attr.text else None
            attributes[name] = value
            if name == "level":
                try:
                    level = int(value)
                except Exception:
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


# Utility: Get all descendants of an element in the tree
# --------------------------------------------------
def get_descendant_ids(node):
    ids = set()
    stack = [node]
    while stack:
        n = stack.pop()
        ids.add(n.element.id)
        stack.extend(n.children)
    return ids


# Updated validation: element is touched if it or any descendant is touched
# --------------------------------------------------
def validate_every_element_touched(
    elements: List[Element], path_to_elements: dict, tree_roots
) -> List[ValidationError]:
    errors = []
    touched = set()
    for elist in path_to_elements.values():
        touched.update(elist)
    # Build id->node mapping
    id_to_node = {}
    stack = list(tree_roots)
    while stack:
        node = stack.pop()
        id_to_node[node.element.id] = node
        stack.extend(node.children)
    for el in elements:
        if el.content_layer.upper() == "BACKGROUND":
            continue
        node = id_to_node.get(el.id)
        if not node:
            continue
        descendant_ids = get_descendant_ids(node)
        if not (descendant_ids & touched):
            errors.append(
                ValidationError(
                    error_type="element_not_touched_by_reading_order",
                    message=f"Element {el.id} ({el.label}) not touched by any reading-order path.",
                    element_id=el.id,
                )
            )
    return errors


# Update validate_sample to pass tree_roots to validate_every_element_touched
# --------------------------------------------------
def validate_sample(
    elements, paths, tree_roots, path_to_elements, path_to_container
) -> ValidationReport:
    errors = []
    errors.extend(validate_valid_labels(elements))
    errors.extend(validate_first_level_reading_order(paths))
    errors.extend(
        validate_second_level_reading_order_has_parent(paths, path_to_container)
    )
    errors.extend(
        validate_every_element_touched(elements, path_to_elements, tree_roots)
    )
    return ValidationReport(sample_name="sample", errors=errors)


# Update test_validate_and_print_report to process a directory of samples
# --------------------------------------------------
def test_validate_and_print_report(input_path: Path):
    """
    Test function: parse XML/images, build tree, compute reading order, validate, and print JSON report for all samples in a directory or a single file.
    Not part of the CLI.
    """
    samples = []
    if input_path.is_dir():
        for sample_name, xml_path, image_filename in find_samples_in_directory(
            input_path
        ):
            elements, paths, image_info = parse_cvat_xml_for_image(
                xml_path, image_filename
            )
            roots = build_containment_tree(elements)
            path_to_elements = map_path_points_to_elements(paths, elements)
            path_to_container = associate_reading_order_paths_to_containers(
                path_to_elements, roots
            )
            samples.append(
                (
                    sample_name,
                    elements,
                    paths,
                    roots,
                    path_to_elements,
                    path_to_container,
                )
            )
    else:
        # Single XML file, process all <image> elements
        tree = ET.parse(input_path)
        root = tree.getroot()
        for image_el in root.findall(".//image"):
            image_filename = image_el.attrib["name"]
            elements, paths, image_info = parse_cvat_xml_for_image(
                input_path, image_filename
            )
            roots = build_containment_tree(elements)
            path_to_elements = map_path_points_to_elements(paths, elements)
            path_to_container = associate_reading_order_paths_to_containers(
                path_to_elements, roots
            )
            samples.append(
                (
                    image_filename,
                    elements,
                    paths,
                    roots,
                    path_to_elements,
                    path_to_container,
                )
            )
    report = validate_and_report(samples)
    print(json.dumps(report.model_dump(), indent=2))


if __name__ == "__main__":
    main()
