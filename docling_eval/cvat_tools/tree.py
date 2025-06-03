from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .models import CVATAnnotationPath, Element


@dataclass
class TreeNode:
    """Node in the containment tree. Holds an Element, parent, and children."""

    element: Element
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


def contains(parent: Element, child: Element, iou_thresh: float = 0.99) -> bool:
    """Check if parent element contains child element based on IOU threshold."""
    intersection = parent.bbox.intersection_area_with(child.bbox)
    return intersection / (child.bbox.area() + 1e-6) > iou_thresh


def build_containment_tree(elements: List[Element]) -> List[TreeNode]:
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


def map_path_points_to_elements(
    paths: List[CVATAnnotationPath],
    elements: List[Element],
    proximity_thresh: float = 5.0,
) -> Dict[int, List[int]]:
    """Map path control points to the deepest elements they touch."""
    path_to_elements: Dict[int, List[int]] = {}

    for path in paths:
        if not path.label.startswith("reading_order"):
            continue

        touched_elements = []
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
                if not touched_elements or touched_elements[-1] != eid:
                    touched_elements.append(eid)

        if touched_elements:
            path_to_elements[path.id] = touched_elements

    return path_to_elements


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


def get_ancestors(node: TreeNode) -> List[TreeNode]:
    """Return a list of ancestors from closest to root."""
    ancestors = []
    while node.parent:
        node = node.parent
        ancestors.append(node)
    return ancestors


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


def associate_reading_order_paths_to_containers(
    path_to_elements: Dict[int, List[int]], tree_roots: List[TreeNode]
) -> Dict[int, TreeNode]:
    """Associate reading-order paths to their closest parent containers."""
    path_to_container: Dict[int, TreeNode] = {}

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

    def insert_with_ancestors(eid: int, path_container_id: Optional[int]) -> None:
        node = find_node_by_element_id(tree_roots, eid)
        if not node:
            return

        # Collect ancestors up to (but not including) the path's associated container
        ancestors = []
        current = node.parent
        while current and current.element.id != path_container_id:
            ancestors.append(current)
            current = current.parent

        # Insert ancestors first
        for ancestor in reversed(ancestors):
            if ancestor.element.id not in visited:
                result.append(ancestor.element.id)
                visited.add(ancestor.element.id)

        # Insert the element itself
        if eid not in visited:
            result.append(eid)
            visited.add(eid)

    def insert_path(path_id: int) -> None:
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
            insert_with_ancestors(eid, path_container_id)

            # Also insert all direct children of this container if they are touched by the path
            if node.children:
                child_ids = [child.element.id for child in node.children]
                for child_id in child_ids:
                    if (
                        child_id in path_to_elements.get(path_id, [])
                        and child_id not in visited
                    ):
                        insert_with_ancestors(child_id, path_container_id)

    insert_path(level1_path.id)
    return result
