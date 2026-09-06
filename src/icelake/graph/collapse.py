"""Identity collapse for the relation graph.

An entity with ``linked_user_id`` *is* that member. Read paths rewrite twin
endpoints to user nodes so person queries stay person queries. Pure functions;
the store lookup lives at the call site. No Discord types — only graph models.
"""

from __future__ import annotations

from icelake.models.graph import NodeType, RelationEdge
from icelake.ports.store import NodeRef


def twin_refs(user_id: str, linked: dict[str, str]) -> tuple[NodeRef, ...]:
    """The user node plus every entity twin bridged to them (one query set)."""
    twins = tuple((NodeType.ENTITY, slug) for slug, uid in linked.items() if uid == user_id)
    return ((NodeType.USER, user_id), *twins)


def collapse_edges(
    edges: tuple[RelationEdge, ...], linked: dict[str, str]
) -> tuple[RelationEdge, ...]:
    """Rewrite linked entity endpoints to user nodes; drop self-loops and dupes."""
    if not edges or not linked:
        return edges
    seen: set[tuple[str, str, str, str, str]] = set()
    out: list[RelationEdge] = []
    for edge in edges:
        src_type, src_id = edge.src_type, edge.src_id
        dst_type, dst_id = edge.dst_type, edge.dst_id
        if src_type is NodeType.ENTITY and src_id in linked:
            src_type, src_id = NodeType.USER, linked[src_id]
        if dst_type is NodeType.ENTITY and dst_id in linked:
            dst_type, dst_id = NodeType.USER, linked[dst_id]
        if (src_type, src_id) == (dst_type, dst_id):
            continue  # self-loop created by collapsing both ends
        key = (src_type.value, src_id, dst_type.value, dst_id, edge.verb)
        if key in seen:
            continue
        seen.add(key)
        if (src_type, src_id) != (edge.src_type, edge.src_id) or (dst_type, dst_id) != (
            edge.dst_type,
            edge.dst_id,
        ):
            edge = edge.model_copy(
                update={
                    "src_type": src_type,
                    "src_id": src_id,
                    "dst_type": dst_type,
                    "dst_id": dst_id,
                }
            )
        out.append(edge)
    return tuple(out)
