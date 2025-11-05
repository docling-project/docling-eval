"""Path mapping and validation for CVAT annotations.

This module provides functions for mapping different types of paths (reading order, merge, group, etc.)
to elements and validating their relationships.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Set, Tuple

from docling_core.types.doc.labels import DocItemLabel

from .models import CVATAnnotationPath, CVATElement
from .tree import (
    TreeNode,
    closest_common_ancestor,
    find_node_by_element_id,
    index_tree_by_element_id,
)
from .utils import (
    DEFAULT_PROXIMITY_THRESHOLD,
    find_elements_containing_point,
    get_deepest_element_at_point,
    is_caption_element,
    is_container_element,
    is_footnote_element,
)

logger = logging.getLogger(__name__)


@dataclass
class PathMappings:
    """Container for all path-to-element mappings."""

    reading_order: Dict[int, List[int]]  # path_id -> [element_id, ...]
    merge: Dict[int, List[int]]  # path_id -> [element_id, ...]
    group: Dict[int, List[int]]  # path_id -> [element_id, ...]
    to_caption: Dict[int, Tuple[int, int]]  # path_id -> (container_id, caption_id)
    to_footnote: Dict[int, Tuple[int, int]]  # path_id -> (container_id, footnote_id)
    to_value: Dict[int, Tuple[int, int]]  # path_id -> (key_id, value_id)


def map_path_points_to_elements(
    paths: List[CVATAnnotationPath],
    elements: List[CVATElement],
    proximity_thresh: float = DEFAULT_PROXIMITY_THRESHOLD,
) -> PathMappings:
    """Map all path types to their connected elements.

    Args:
        paths: List of CVAT annotation paths
        elements: List of elements to map paths to
        proximity_thresh: Distance threshold for point-to-element mapping

    Returns:
        PathMappings object containing all path-to-element mappings
    """
    from docling_core.types.doc.labels import GraphCellLabel

    reading_order: Dict[int, List[int]] = {}
    merge: Dict[int, List[int]] = {}
    group: Dict[int, List[int]] = {}
    to_caption: Dict[int, Tuple[int, int]] = {}
    to_footnote: Dict[int, Tuple[int, int]] = {}
    to_value: Dict[int, Tuple[int, int]] = {}

    for path in paths:
        # For group/merge/reading_order paths, skip GraphCellLabel elements (key/value)
        # They should never be in groups, merges, or reading order directly
        # For to_value paths, we need to hit GraphCellLabel elements
        skip_graph_cells = path.label in ["group", "merge"] or path.label.startswith(
            "reading_order"
        )

        touched_elements: List[int] = []
        for pt in path.points:
            deepest = get_deepest_element_at_point(
                pt, elements, proximity_thresh, skip_graph_cells=skip_graph_cells
            )
            if deepest:
                eid = deepest.id
                # Only add if not a duplicate in sequence
                if not touched_elements or touched_elements[-1] != eid:
                    touched_elements.append(eid)

        if not touched_elements:
            continue

        # Map based on path label
        if path.label.startswith("reading_order"):
            reading_order[path.id] = touched_elements
        elif path.label == "merge":
            merge[path.id] = touched_elements
        elif path.label == "group":
            group[path.id] = touched_elements
        elif path.label == "to_caption" and len(touched_elements) == 2:
            # First element should be container, second should be caption
            container_id, caption_id = touched_elements[0], touched_elements[1]

            # Get elements to check their types
            container_el = next((el for el in elements if el.id == container_id), None)
            caption_el = next((el for el in elements if el.id == caption_id), None)

            # Check if the relationship is backwards and auto-correct
            if (
                container_el
                and caption_el
                and is_caption_element(container_el)
                and is_container_element(caption_el)
            ):
                logger.debug(
                    f"Caption path {path.id}: Backwards annotation detected, auto-correcting"
                )
                container_id, caption_id = caption_id, container_id

            to_caption[path.id] = (container_id, caption_id)
        elif path.label == "to_footnote" and len(touched_elements) == 2:
            # First element should be container, second should be footnote
            container_id, footnote_id = touched_elements[0], touched_elements[1]

            # Get elements to check their types
            container_el = next((el for el in elements if el.id == container_id), None)
            footnote_el = next((el for el in elements if el.id == footnote_id), None)

            # Check if the relationship is backwards and auto-correct
            if (
                container_el
                and footnote_el
                and is_footnote_element(container_el)
                and is_container_element(footnote_el)
            ):
                logger.debug(
                    f"Footnote path {path.id}: Backwards annotation detected, auto-correcting"
                )
                container_id, footnote_id = footnote_id, container_id

            to_footnote[path.id] = (container_id, footnote_id)
        elif path.label == "to_value":
            if len(touched_elements) == 2:
                # Simple case: exactly 2 elements
                to_value[path.id] = (touched_elements[0], touched_elements[1])
            elif len(touched_elements) > 2:
                # Check if elements can be reduced to 2 logical elements via merges
                # This will be resolved after all paths are parsed
                to_value[path.id] = tuple(touched_elements)  # type: ignore[assignment]

    # Resolve reading order conflicts before returning
    reading_order = _resolve_reading_order_conflicts(reading_order, paths, elements)

    # Resolve to_value paths that touch more than 2 elements via merges
    to_value = _resolve_to_value_with_merges(dict(to_value), merge, elements)

    return PathMappings(
        reading_order=reading_order,
        merge=merge,
        group=group,
        to_caption=to_caption,
        to_footnote=to_footnote,
        to_value=to_value,
    )


def _resolve_to_value_with_merges(
    to_value: Dict[int, Tuple[int, ...]],
    merge: Dict[int, List[int]],
    elements: List[CVATElement],
) -> Dict[int, Tuple[int, int]]:
    """Resolve to_value paths by grouping elements via merge relationships."""
    # Build element-to-merge-group mapping
    element_to_group: Dict[int, Set[int]] = {}
    for merge_elements in merge.values():
        group = set(merge_elements)
        for el_id in merge_elements:
            element_to_group[el_id] = group

    resolved: Dict[int, Tuple[int, int]] = {}

    for path_id, element_tuple in to_value.items():
        if len(element_tuple) == 2:
            resolved[path_id] = element_tuple  # type: ignore[assignment]
            continue

        # Group elements by merge relationships
        groups: List[Set[int]] = []
        seen = set()

        for el_id in element_tuple:
            if el_id in seen:
                continue
            group = element_to_group.get(el_id, {el_id})
            groups.append(group)
            seen.update(group)

        if len(groups) == 2:
            # Valid: 2 logical elements (use min ID as representative)
            resolved[path_id] = (min(groups[0]), min(groups[1]))
            logger.debug(
                f"to_value path {path_id}: Resolved {len(element_tuple)} elements to 2 groups via merges"
            )
        else:
            # Invalid: will be caught by validation
            logger.debug(
                f"to_value path {path_id}: {len(element_tuple)} elements → {len(groups)} groups (expected 2). Ignored."
            )

    return resolved


def _find_container_for_conflicted_path(
    path: CVATAnnotationPath,
    lost_entries: List[Tuple[int, int]],
    elements: List[CVATElement],
) -> List[Tuple[CVATElement, int]]:
    """Return container elements to reinsert for a conflicted reading-order path.

    The function groups lost elements by the smallest enclosing container whose
    boundary is crossed by the reading-order polyline. Containers are never merged
    across independent regions—each returned container strictly corresponds to the
    lost elements it contains, unless containers are nested.
    """
    from .tree import contains
    from .utils import is_container_element

    if not lost_entries:
        return []

    id_to_element: Dict[int, CVATElement] = {
        element.id: element for element in elements
    }
    lost_element_ids = [element_id for element_id, _ in lost_entries]

    def _point_inside(element: CVATElement, point: Tuple[float, float]) -> bool:
        x, y = point
        bbox = element.bbox
        x_min, x_max = (bbox.l, bbox.r) if bbox.l <= bbox.r else (bbox.r, bbox.l)
        y_min, y_max = (bbox.t, bbox.b) if bbox.t <= bbox.b else (bbox.b, bbox.t)
        return x_min <= x <= x_max and y_min <= y <= y_max

    container_info: Dict[int, Tuple[CVATElement, Set[int]]] = {}

    for element in elements:
        if not is_container_element(element):
            continue

        if path.points and all(_point_inside(element, point) for point in path.points):
            continue

        covered_ids: Set[int] = set()
        for lost_id in lost_element_ids:
            lost_element = id_to_element.get(lost_id)
            if lost_element and contains(element, lost_element):
                covered_ids.add(lost_id)

        if covered_ids:
            container_info[element.id] = (element, covered_ids)

    if not container_info:
        return []

    container_assignments: Dict[int, Set[int]] = {}
    for element_id, _ in lost_entries:
        candidate_ids = [
            container_id
            for container_id, (_, covered) in container_info.items()
            if element_id in covered
        ]
        if not candidate_ids:
            continue

        candidate_ids.sort(
            key=lambda container_id: container_info[container_id][0].bbox.area()
        )
        chosen_id = candidate_ids[0]
        container_assignments.setdefault(chosen_id, set()).add(element_id)

    if not container_assignments:
        return []

    lost_index_by_id = {element_id: index for element_id, index in lost_entries}

    containers_to_insert: List[Tuple[CVATElement, int]] = []
    for container_id, assigned_ids in container_assignments.items():
        container_element, _ = container_info[container_id]
        insert_index = min(lost_index_by_id[element_id] for element_id in assigned_ids)
        containers_to_insert.append((container_element, insert_index))

    containers_to_insert.sort(key=lambda item: item[1])
    return containers_to_insert


def _resolve_reading_order_conflicts(
    reading_order: Dict[int, List[int]],
    paths: List[CVATAnnotationPath],
    elements: List[CVATElement],
) -> Dict[int, List[int]]:
    """Resolve conflicts where elements appear in multiple reading order paths."""
    # Build path mappings (reuse existing pattern from associate_paths_to_containers)
    path_levels = {
        p.id: p.level or 1 for p in paths if p.label.startswith("reading_order")
    }
    path_by_id = {p.id: p for p in paths if p.label.startswith("reading_order")}

    # Find element conflicts
    element_to_paths: Dict[int, List[Tuple[int, int]]] = {}
    for path_id, element_ids in reading_order.items():
        level = path_levels.get(path_id, 1)
        for element_id in element_ids:
            element_to_paths.setdefault(element_id, []).append((path_id, level))

    conflicts = {
        eid: paths for eid, paths in element_to_paths.items() if len(paths) > 1
    }
    if conflicts:
        logger.debug(f"Resolving {len(conflicts)} reading order conflicts")

    # Resolve conflicts: assign to deepest level, find containers for emptied paths
    emptied_paths: Dict[int, List[Tuple[int, int]]] = {}
    for element_id, path_level_pairs in conflicts.items():
        keep_path_id = max(path_level_pairs, key=lambda x: x[1])[0]
        for path_id, _ in path_level_pairs:
            if path_id != keep_path_id:
                if element_id in reading_order[path_id]:
                    removal_index = reading_order[path_id].index(element_id)
                    reading_order[path_id].pop(removal_index)
                    emptied_paths.setdefault(path_id, []).append(
                        (element_id, removal_index)
                    )

    # Add containers for paths that lost elements
    for path_id, lost_entries in emptied_paths.items():
        path = path_by_id.get(path_id)
        if not path:
            continue

        containers = _find_container_for_conflicted_path(path, lost_entries, elements)
        if not containers:
            continue

        inserted_positions: List[int] = []
        for container_element, insert_idx in containers:
            if container_element.id in reading_order[path_id]:
                continue

            offset = sum(1 for position in inserted_positions if position <= insert_idx)
            insert_at = min(insert_idx + offset, len(reading_order[path_id]))
            reading_order[path_id].insert(insert_at, container_element.id)
            inserted_positions.append(insert_idx)
            logger.debug(
                "Inserted container element %s (%s) into reading order path %s at position %s",
                container_element.id,
                container_element.label,
                path_id,
                insert_at,
            )

    return reading_order


def associate_paths_to_containers(
    mappings: PathMappings,
    tree_roots: List[TreeNode],
    paths: List[CVATAnnotationPath],
) -> Tuple[PathMappings, Dict[int, TreeNode]]:
    """Associate paths to their closest parent containers.

    Args:
        mappings: PathMappings object containing path-to-element mappings
        tree_roots: List of root nodes in the containment tree
        paths: List of all paths to check levels

    Returns:
        Tuple of (PathMappings, Dict[int, TreeNode]) where the dict maps path_id to container node
    """
    path_to_container: Dict[int, TreeNode] = {}

    # Create a mapping of path_id to path level for reading order paths
    path_levels = {p.id: p.level for p in paths if p.label.startswith("reading_order")}

    # Helper function to find common ancestor
    def find_common_ancestor(element_ids: List[int]) -> Optional[TreeNode]:
        touched_nodes: List[TreeNode] = [
            n
            for n in [find_node_by_element_id(tree_roots, eid) for eid in element_ids]
            if n is not None
        ]
        if not touched_nodes:
            return None
        return closest_common_ancestor(touched_nodes)

    # Associate reading order paths
    for path_id, el_ids in mappings.reading_order.items():
        # Skip level 1 reading order paths - they don't need containers
        if path_id in path_levels and (
            path_levels[path_id] == 1 or path_levels[path_id] is None
        ):
            continue

        ancestor = find_common_ancestor(el_ids)
        if ancestor is not None:
            path_to_container[path_id] = ancestor
        else:
            # fallback: parent of first touched element
            node = find_node_by_element_id(tree_roots, el_ids[0])
            if node and node.parent:
                path_to_container[path_id] = node.parent

    # Associate merge paths
    for path_id, el_ids in mappings.merge.items():
        ancestor = find_common_ancestor(el_ids)
        if ancestor is not None:
            path_to_container[path_id] = ancestor

    # Associate group paths
    for path_id, el_ids in mappings.group.items():
        ancestor = find_common_ancestor(el_ids)
        if ancestor is not None:
            path_to_container[path_id] = ancestor

    # Associate to_caption and to_footnote paths
    for path_id, (container_id, _) in mappings.to_caption.items():
        node = find_node_by_element_id(tree_roots, container_id)
        if node:
            path_to_container[path_id] = node

    for path_id, (container_id, _) in mappings.to_footnote.items():
        node = find_node_by_element_id(tree_roots, container_id)
        if node:
            path_to_container[path_id] = node

    # Associate to_value paths
    for path_id, (key_id, _) in mappings.to_value.items():
        node = find_node_by_element_id(tree_roots, key_id)
        if node:
            path_to_container[path_id] = node

    return mappings, path_to_container


def promote_table_cross_boundary_reading_order(
    mappings: PathMappings,
    paths: List[CVATAnnotationPath],
    tree_roots: List[TreeNode],
    tolerance: float = DEFAULT_PROXIMITY_THRESHOLD,
) -> PathMappings:
    """Promote table containers for reading-order paths that cross table boundaries.

    When a reading-order path starts or ends outside a table but touches elements inside it,
    this inserts the table element itself into the mapped order to anchor the container in
    the global reading order while keeping existing descendants.
    """

    if not mappings.reading_order:
        return mappings

    id_to_path = {
        path.id: path for path in paths if path.label.startswith("reading_order")
    }
    id_to_node = index_tree_by_element_id(tree_roots)

    for path_id, touched_ids in list(mappings.reading_order.items()):
        path = id_to_path.get(path_id)
        if not path or not touched_ids:
            continue

        table_states: Dict[int, Tuple[TreeNode, bool]] = {}

        for element_id in touched_ids:
            node = id_to_node.get(element_id)
            if node is None:
                continue

            table_node = _find_table_ancestor(node)
            if table_node is None:
                continue

            table_id = table_node.element.id
            if table_id not in table_states:
                crosses_boundary = _path_crosses_table_boundary(
                    path.points, table_node.element, tolerance
                )
                table_states[table_id] = (table_node, crosses_boundary)

        if not table_states:
            continue

        updated_order: List[int] = []

        for element_id in touched_ids:
            node = id_to_node.get(element_id)
            table_node = _find_table_ancestor(node) if node else None

            if table_node is not None:
                table_id = table_node.element.id
                _, crosses_boundary = table_states.get(table_id, (table_node, False))
                if crosses_boundary:
                    _append_unique(updated_order, table_id)

            _append_unique(updated_order, element_id)

        mappings.reading_order[path_id] = updated_order

    return mappings


def _find_table_ancestor(node: Optional[TreeNode]) -> Optional[TreeNode]:
    """Return the closest ancestor whose label is DocItemLabel.TABLE."""
    current = node
    while current is not None:
        label = current.element.label
        if isinstance(label, DocItemLabel) and label == DocItemLabel.TABLE:
            return current
        current = current.parent
    return None


def _append_unique(sequence: List[int], element_id: int) -> None:
    if element_id not in sequence:
        sequence.append(element_id)


def _path_crosses_table_boundary(
    points: List[Tuple[float, float]], table_element: CVATElement, tolerance: float
) -> bool:
    """Return True when the polyline has at least one point inside and one point outside the table."""
    if not points:
        return False

    inside_flags = []
    for point in points:
        hits = find_elements_containing_point(
            point, [table_element], proximity_thresh=tolerance
        )
        inside_flags.append(bool(hits))

    any_inside = any(inside_flags)
    any_outside = any(not flag for flag in inside_flags)
    return any_inside and any_outside
