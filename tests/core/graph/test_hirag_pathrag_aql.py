from core.graph.hirag_pathrag import aql


def test_build_hierarchy_candidate_query_valid():
    config = aql.HierarchyTraversalConfig(
        topic_k=3,
        beam_widths=[4, 2],
        candidate_cap=100,
        deg_alpha=0.01,
    )
    query = aql.build_hierarchy_candidate_query(config)
    assert "LIMIT 3" in query
    assert "FOR doc IN candidates" in query


def test_build_hierarchy_candidate_query_requires_beam():
    config = aql.HierarchyTraversalConfig(
        topic_k=1,
        beam_widths=[],
        candidate_cap=10,
        deg_alpha=0.01,
    )
    try:
        aql.build_hierarchy_candidate_query(config)
    except ValueError as exc:
        assert "beam_widths" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for empty beam widths")


def test_build_weighted_path_query_basic():
    config = aql.PathExtractionConfig(max_paths=5, max_hops=4, lambda_term=0.3)
    query = aql.build_weighted_path_query(config)
    assert "LIMIT 5" in query
    assert "weightAttribute" in query
