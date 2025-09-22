from core.graph.hirag import aql as hirag_aql
from core.graph.pathrag import aql as pathrag_aql


def test_build_hierarchy_candidate_query_valid():
    config = hirag_aql.HierarchyTraversalConfig(
        topic_k=3,
        beam_widths=[4, 2],
        candidate_cap=100,
        deg_alpha=0.01,
    )
    query = hirag_aql.build_hierarchy_candidate_query(config)
    assert "LIMIT 3" in query
    assert "FOR doc IN candidates" in query


def test_build_hierarchy_candidate_query_requires_beam():
    config = hirag_aql.HierarchyTraversalConfig(
        topic_k=1,
        beam_widths=[],
        candidate_cap=10,
        deg_alpha=0.01,
    )
    try:
        hirag_aql.build_hierarchy_candidate_query(config)
    except ValueError as exc:
        assert "beam_widths" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for empty beam widths")


def test_build_weighted_path_query_basic():
    config = pathrag_aql.PathExtractionConfig(max_paths=5, max_hops=4, lambda_term=0.3)
    query = pathrag_aql.build_weighted_path_query(config)
    assert "LIMIT 5" in query
    assert "weightAttribute" in query
