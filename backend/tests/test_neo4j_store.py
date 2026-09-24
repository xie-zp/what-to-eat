"""Neo4j 存储工具的纯 Python 测试。"""

from app.rag.neo4j_store import build_search_groups, build_search_terms, chunked


def test_chunked_splits_values() -> None:
    values = [{"id": index} for index in range(5)]
    batches = list(chunked(values, 2))

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert batches[0][0]["id"] == 0
    assert batches[-1][0]["id"] == 4


def test_search_terms_remove_question_words_and_expand_concepts() -> None:
    terms = build_search_terms("鸡肉配什么蔬菜好？")
    groups = build_search_groups(terms)

    assert terms == ["鸡肉", "蔬菜"]
    assert "鸡胸肉" in groups[0]
    assert "西兰花" in groups[1]
