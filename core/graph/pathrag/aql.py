"""AQL builders for PathRAG weighted path extraction."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass


@dataclass(frozen=True)
class PathExtractionConfig:
    """Configuration for flow-pruned path extraction."""

    max_paths: int
    max_hops: int
    lambda_term: float


def build_weighted_path_query(config: PathExtractionConfig) -> str:
    """Return an AQL query that extracts weighted paths."""

    query = f"""
    FOR v, e, p IN OUTBOUND SHORTEST_PATH @source TO @target
      GRAPH @graph_name
      OPTIONS {{ weightAttribute: "weight", defaultWeight: 0.1 }}
      FILTER LENGTH(p.edges) <= {config.max_hops}
      LET raw = SUM(
        FOR ed IN p.edges
          LET rel_boost = @relation_boost[ed.type] ? @relation_boost[ed.type] : 1.0
          RETURN LOG(GREATEST(ed.weight * rel_boost, @epsilon))
      )
      LET score = raw / (1 + {config.lambda_term} * (LENGTH(p.edges) - 2))
      SORT score DESC
      LIMIT {config.max_paths}
      RETURN {{
        nodes: p.vertices,
        edges: p.edges,
        score: score
      }}
    """
    return textwrap.dedent(query)
