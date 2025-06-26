"""Analysis and visualization tools for CVAT annotations.

This module provides functions for visualizing CVAT annotations,
including printing elements and paths, and containment trees.
"""

from typing import List

from .models import CVATAnnotationPath, CVATElement, CVATImageInfo
from .tree import TreeNode


def print_elements_and_paths(
    elements: List[CVATElement],
    paths: List[CVATAnnotationPath],
    image_info: CVATImageInfo,
) -> None:
    """Print a simple tree of elements and paths for debugging."""
    print(f"Image: {image_info.name} ({image_info.width}x{image_info.height})\n")
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


def print_containment_tree(
    tree_roots: List[TreeNode],
    image_info: CVATImageInfo,
) -> None:
    """Print the containment tree indented."""
    print(
        f"Containment tree for {image_info.name} ({image_info.width}x{image_info.height}):\n"
    )

    def print_tree(node: TreeNode, indent: int = 0) -> None:
        el = node.element
        print(
            "  " * indent
            + f"[{el.label} id={el.id}] bbox=({el.bbox.l:.1f},{el.bbox.t:.1f},{el.bbox.r:.1f},{el.bbox.b:.1f})"
        )
        for child in node.children:
            print_tree(child, indent + 1)

    for root in tree_roots:
        print_tree(root)
