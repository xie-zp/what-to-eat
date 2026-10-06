"""混合检索测试。"""

from pathlib import Path

from app.rag.documents import RecipeDocument, build_recipe_document
from app.rag.hybrid import HybridRecipeSearcher, reciprocal_rank_fusion

from scripts.parse_howtocook import parse_directory

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "howtocook"


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


class _FakeVectorIndex:
    """只返回单个命中片段，用于验证父文档回填。"""

    def __init__(self, recipe_id: str) -> None:
        self.recipe_id = recipe_id

    def search(self, _query: str, limit: int = 10):
        return [
            {
                "recipe_id": self.recipe_id,
                "chunk_id": f"{self.recipe_id}#steps",
                "chunk_type": "steps",
                "text": "菜名：咖喱炒蟹\n步骤：\n1. 处理青蟹",
                "score": 0.9,
                "name": "咖喱炒蟹",
            }
        ]


def test_hybrid_search_backfills_parent_document_from_chunk() -> None:
    recipe = parse_directory(FIXTURE_ROOT)[0]
    document = build_recipe_document(recipe)
    searcher = HybridRecipeSearcher(
        documents=[document],
        vector_index=_FakeVectorIndex(document.recipe_id),
    )

    records = searcher.search("咖喱炒蟹怎么做", limit=1)

    assert len(records) == 1
    # 命中的是步骤片段，但返回的是完整父菜谱文本。
    assert records[0]["recipe_id"] == document.recipe_id
    assert records[0]["parent_id"] == document.recipe_id
    assert records[0]["text"] == document.text
    assert records[0]["chunk_text"].startswith("菜名：咖喱炒蟹")


class _FailingVectorIndex:
    """模拟向量库不可用。"""

    def search(self, *_args, **_kwargs):
        raise RuntimeError("milvus down")


def test_hybrid_search_degrades_to_bm25_when_vector_fails() -> None:
    # 单文档语料下 BM25 的 IDF 为负，这里补足三篇让关键词检索可用。
    documents = [
        _make_document("r1", "咖喱炒蟹", "菜名：咖喱炒蟹\n分类：水产\n食材：青蟹、咖喱块"),
        _make_document("r2", "西红柿炒鸡蛋", "菜名：西红柿炒鸡蛋\n分类：素菜\n食材：西红柿、鸡蛋"),
        _make_document("r3", "红烧肉", "菜名：红烧肉\n分类：荤菜\n食材：五花肉、冰糖"),
    ]
    searcher = HybridRecipeSearcher(
        documents=documents,
        vector_index=_FailingVectorIndex(),
    )

    records = searcher.search("咖喱炒蟹", limit=3)

    assert searcher.degraded_channels == ["vector"]
    assert "r1" in [item["recipe_id"] for item in records]


def _make_document(recipe_id: str, name: str, text: str) -> RecipeDocument:
    return RecipeDocument(
        recipe_id=recipe_id,
        name=name,
        text=text,
        category="水产",
        difficulty=2,
        source_path="fixture.md",
    )
