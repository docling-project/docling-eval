"""Tree data structures and algorithms for CVAT annotations.

This module provides the TreeNode class for representing containment hierarchies
and various functions for building and manipulating these trees, including
reading order processing and ancestor searching.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Set

from .geometry import bbox_fraction_inside
from .models import CVATAnnotationPath, CVATElement
from .utils import is_container_element


@dataclass
class TreeNode:
    """Node in the containment tree. Holds an Element, parent, and children."""

    element: CVATElement
    parent: Optional["TreeNode"] = None
    children: List["TreeNode"] = field(default_factory=list)

    def add_child(self, child: "TreeNode") -> None:
        """Add a child node to this node."""
        self.children.append(child)
        child.parent = self

    def get_descendant_ids(self) -> Set[int]:
        """Get all descendant element IDs."""
        ids = {self.element.id}
        for child in self.children:
            ids.update(child.get_descendant_ids())
        return ids


def iter_tree_nodes(roots: Iterable["TreeNode"]) -> Iterator["TreeNode"]:
    """Yield all nodes contained in ``roots`` using depth-first traversal."""

    stack = list(roots)
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def iter_ancestors(
    node: "TreeNode", include_self: bool = False
) -> Iterator["TreeNode"]:
    """Yield ancestors for ``node`` starting from its parent."""

    current = node if include_self else node.parent
    while current is not None:
        yield current
        current = current.parent


def get_ancestors(node: "TreeNode") -> List["TreeNode"]:
    """Return a list of ancestors from closest to root."""

    return list(iter_ancestors(node))


def find_ancestor(
    node: "TreeNode", predicate: Callable[["TreeNode"], bool]
) -> Optional["TreeNode"]:
    """Return the first ancestor matching ``predicate`` or ``None``."""

    for ancestor in iter_ancestors(node):
        if predicate(ancestor):
            return ancestor
    return None


def index_tree_by_element_id(
    roots: Iterable["TreeNode"],
) -> Dict[int, "TreeNode"]:
    """Map every element identifier to its ``TreeNode``."""

    return {node.element.id: node for node in iter_tree_nodes(roots)}


def contains(parent: CVATElement, child: CVATElement, iou_thresh: float = 0.7) -> bool:
    """Check if parent element contains child element based on IoU threshold."""

    if parent.bbox.area() <= child.bbox.area():
        return False

    return bbox_fraction_inside(child.bbox, parent.bbox) > iou_thresh


def build_containment_tree(elements: List[CVATElement]) -> List[TreeNode]:
    """Build a containment tree from elements based on spatial containment and content_layer."""
    nodes = [TreeNode(el) for el in elements]

    for i, node in enumerate(nodes):
        best_parent = None
        best_area = None

        for j, candidate in enumerate(nodes):
            if i == j:
                continue
            if node.element.content_layer != candidate.element.content_layer:
                continue
            if contains(candidate.element, node.element):
                area = candidate.element.bbox.area()
                if best_area is None or area < best_area:
                    best_parent = candidate
                    best_area = area

        if best_parent:
            best_parent.add_child(node)

    return [node for node in nodes if node.parent is None]


def find_node_by_element_id(
    tree_roots: List[TreeNode], element_id: int
) -> Optional[TreeNode]:
    """Find the TreeNode for a given element_id in the tree."""
    stack = list(tree_roots)
    while stack:
        node = stack.pop()
        if node.element.id == element_id:
            return node
        stack.extend(node.children)
    return None


def closest_common_ancestor(nodes: List[TreeNode]) -> Optional[TreeNode]:
    """Find the closest common ancestor of a list of nodes."""
    if not nodes:
        return None

    ancestor_lists = [[node] + get_ancestors(node) for node in nodes]
    # Find the first common node in all ancestor lists
    for candidate in ancestor_lists[0]:
        if all(candidate in lst for lst in ancestor_lists[1:]):
            return candidate
    return None


def apply_reading_order_to_tree(
    tree_roots: List[TreeNode],
    global_order: List[int],
) -> List[TreeNode]:
    """Reorder the children of each container node in the tree to match the global reading order.

    This returns a new reordered copy of tree_roots and modifies the children lists in-place.

    Args:
        tree_roots: List of root nodes to reorder
        global_order: List of element IDs in reading order

    Returns:
        New list of reordered tree roots
    """
    if not tree_roots or not global_order:
        return tree_roots[:]

    id_to_node = {node.element.id: node for node in iter_tree_nodes(tree_roots)}

    # First, reorder the tree roots themselves
    ordered_roots = []
    remaining_roots = []

    # Process elements in global order
    for element_id in global_order:
        # Find the node for this element
        node = id_to_node.get(element_id)
        if not node:
            continue

        # If this is a root node, add it to ordered_roots
        if node in tree_roots:
            if node not in ordered_roots:
                ordered_roots.append(node)
        # If this is a child node, reorder its parent's children
        elif node.parent:
            parent = node.parent
            if parent.children:
                # Only keep children that are in global_order, in the right order
                ordered_children = [
                    id_to_node[child_id]
                    for child_id in global_order
                    if any(child.element.id == child_id for child in parent.children)
                ]
                # Append any children not in global_order
                remaining = [
                    child for child in parent.children if child not in ordered_children
                ]
                parent.children = ordered_children + remaining

    # Add any remaining root nodes that weren't in global_order
    remaining_roots = [root for root in tree_roots if root not in ordered_roots]
    new_tree_roots = ordered_roots + remaining_roots

    return new_tree_roots


def build_global_reading_order(
    paths: List[CVATAnnotationPath],
    path_to_elements: Dict[int, List[int]],
    path_to_container: Dict[int, TreeNode],
    tree_roots: List[TreeNode],
) -> List[int]:
    """Build a global reading order from all reading-order paths.

    This function merges all reading-order paths into a single global reading order.
    It handles nested reading orders by recursively processing level-2+ paths.

    Returns:
        List of element IDs in reading order.
    """
    # Gather all level-1 reading order paths; a multi-page document can legitimately
    # have more than one (one per page).
    level1_paths = [
        p
        for p in paths
        if p.label.startswith("reading_order") and (p.level == 1 or p.level is None)
    ]
    if not level1_paths:
        return []

    visited = set()
    result = []
    processed_paths: set[int] = set()

    def insert_with_ancestors(
        eid: int,
        path_container_id: Optional[int],
        position: Optional[int] = None,
        elements_in_path: Optional[List[int]] = None,
    ) -> None:
        node = find_node_by_element_id(tree_roots, eid)
        if not node:
            return

        # Collect ancestors up to (but not including) the path's associated container
        ancestors = []
        current = node.parent
        while current and current.element.id != path_container_id:
            ancestors.append(current)
            current = current.parent

        future_ids: Set[int] = set()
        if (
            position is not None
            and elements_in_path is not None
            and position + 1 < len(elements_in_path)
        ):
            future_ids = set(elements_in_path[position + 1 :])

        before_ancestors: List[TreeNode] = []
        after_ancestors: List[TreeNode] = []

        for ancestor in reversed(ancestors):  # outermost -> innermost
            ancestor_id = ancestor.element.id
            if ancestor_id in visited:
                continue
            if future_ids and ancestor_id in future_ids:
                if not is_container_element(ancestor.element):
                    # Path will insert this non-container ancestor explicitly later
                    continue
            if is_container_element(ancestor.element):
                before_ancestors.append(ancestor)
            else:
                after_ancestors.append(ancestor)

        for ancestor in before_ancestors:
            ancestor_id = ancestor.element.id
            if ancestor_id not in visited:
                result.append(ancestor_id)
                visited.add(ancestor_id)

        # Insert the element itself
        if eid not in visited:
            result.append(eid)
            visited.add(eid)

        for ancestor in reversed(after_ancestors):  # innermost -> outermost
            ancestor_id = ancestor.element.id
            if ancestor_id not in visited:
                result.append(ancestor_id)
                visited.add(ancestor_id)

    def insert_path(path_id: int) -> None:
        if path_id in processed_paths:
            return
        processed_paths.add(path_id)
        path_container = path_to_container.get(path_id)
        path_container_id = path_container.element.id if path_container else None

        # Insert the path's associated container first (if not already visited)
        if path_container and path_container.element.id not in visited:
            result.append(path_container.element.id)
            visited.add(path_container.element.id)

        elements_in_path = path_to_elements.get(path_id, [])
        element_positions = {eid: idx for idx, eid in enumerate(elements_in_path)}

        for idx, eid in enumerate(elements_in_path):
            if eid in visited:
                continue

            node = find_node_by_element_id(tree_roots, eid)
            if not node:
                continue

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
            insert_with_ancestors(
                eid,
                path_container_id,
                position=idx,
                elements_in_path=elements_in_path,
            )

            # Also insert all direct children of this container if they are touched by the path
            if node.children:
                child_ids = [child.element.id for child in node.children]
                for child_id in child_ids:
                    if (
                        child_id in path_to_elements.get(path_id, [])
                        and child_id not in visited
                    ):
                        child_position = element_positions.get(child_id)
                        insert_with_ancestors(
                            child_id,
                            path_container_id,
                            position=child_position,
                            elements_in_path=elements_in_path,
                        )

    # Process each level-1 reading order path in order of appearance (IDs are stable
    # because we rebuilt them sequentially while merging pages).
    for level1_path in sorted(level1_paths, key=lambda p: p.id):
        insert_path(level1_path.id)

    return result
