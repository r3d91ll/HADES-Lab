"""AQL builders for hierarchical narrowing (HiRAG)."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class HierarchyTraversalConfig:
    """Configuration for hierarchical topic narrowing."""

    topic_k: int
    beam_widths: Iterable[int]
    candidate_cap: int
    deg_alpha: float


def build_hierarchy_candidate_query(config: HierarchyTraversalConfig) -> str:
    """Return an AQL query that narrows candidates via hierarchy beams."""

    beam_widths = tuple(config.beam_widths)
    if not beam_widths:
        raise ValueError("beam_widths must contain at least one level")
    max_beam = max(beam_widths)
    depth = len(beam_widths)
    query = f"""
    LET topics = (
      FOR c IN clusters
        FILTER c.level IN [1, 2]
        FILTER COSINE_DISTANCE(c.centroid, @query_embedding) < @topic_threshold
        SORT COSINE_DISTANCE(c.centroid, @query_embedding) ASC
        LIMIT {config.topic_k}
        RETURN c._id
    )

    LET candidates = (
      FOR topic IN topics
        FOR v, e, p IN 1..{depth} OUTBOUND topic GRAPH @graph_name
          FILTER v.type IN @allowed_types
          LET hscore = (1 - COSINE_DISTANCE(v.embedding, @query_embedding))
                       * (1.0 / (1 + {config.deg_alpha} * NVL(v.deg_out, 0)))
          SORT hscore DESC
          LIMIT {max_beam}
          RETURN DISTINCT {{
            node: v,
            score: hscore
          }}
    )

    RETURN SLICE(
      (FOR doc IN candidates SORT doc.score DESC RETURN doc.node),
      0,
      {config.candidate_cap}
    )
    """
    return textwrap.dedent(query)
