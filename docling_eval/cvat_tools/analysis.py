"""Analysis tools for CVAT annotations.

This module provides functions for analyzing and visualizing CVAT annotations,
including printing elements and paths, containment trees, and reading order.
"""

from typing import Dict, List

from .models import CVATAnnotationPath, Element, ImageInfo
from .tree import TreeNode, find_node_by_element_id


def print_elements_and_paths(
    elements: List[Element],
    paths: List[CVATAnnotationPath],
    image_info: ImageInfo,
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
    image_info: ImageInfo,
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


def apply_reading_order_to_tree(
    tree_roots: List[TreeNode],
    global_order: List[int],
) -> None:
    """Reorder the children of each container node in the tree to match the global reading order.

    This does not modify the global_order or the tree structure, only the order of children lists.
    """

    def collect_all_nodes(roots: List[TreeNode]) -> List[TreeNode]:
        stack = list(roots)
        all_nodes = []
        while stack:
            node = stack.pop()
            all_nodes.append(node)
            stack.extend(node.children)
        return all_nodes

    id_to_node = {node.element.id: node for node in collect_all_nodes(tree_roots)}

    # First, reorder the tree roots themselves
    ordered_roots = [
        id_to_node[root_id]
        for root_id in global_order
        if any(root.element.id == root_id for root in tree_roots)
    ]
    remaining_roots = [root for root in tree_roots if root not in ordered_roots]
    tree_roots[:] = ordered_roots + remaining_roots

    # Then reorder children of each node
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
