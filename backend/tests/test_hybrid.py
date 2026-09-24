"""混合检索测试。"""

from app.rag.hybrid import reciprocal_rank_fusion


def test_rrf_prefers_record_present_in_multiple_sources() -> None:
    vector = [
        {"recipe_id": "a", "retrieval_source": "vector"},
        {"recipe_id": "b", "retrieval_source": "vector"},
    ]
    bm25 = [
        {"recipe_id": "b", "retrieval_source": "bm25"},
        {"recipe_id": "c", "retrieval_source": "bm25"},
    ]

    result = reciprocal_rank_fusion([vector, bm25], top_k=3)

    assert result[0]["recipe_id"] == "b"
    assert result[0]["retrieval_sources"] == ["bm25", "vector"]
