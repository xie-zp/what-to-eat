"""搭配类问题专项实验：图谱多跳 vs 文本检索。

问题形如“鸡肉配什么蔬菜”，答案是搭配关系而非菜谱列表，
因此用另一套指标：返回的搭配里有多少真的与主食材共现过。

标准答案同样独立计算：先扫出含该主食材的菜谱，再收集这些菜谱里的蔬菜类食材。
主食材别名表在本脚本内硬编码，不复用检索侧的别名逻辑。
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

# 主食材的常见写法，仅用于独立计算标准答案。
ANCHOR_ALIASES: dict[str, tuple[str, ...]] = {
    "鸡肉": ("鸡肉", "鸡腿", "鸡胸", "鸡翅", "鸡丁", "鸡块", "鸡丝", "鸡爪"),
    "牛肉": ("牛肉", "牛腩", "牛排", "肥牛"),
    "猪肉": ("猪肉", "五花肉", "里脊", "排骨", "肉末", "肉丝", "肉片", "猪排"),
    "豆腐": ("豆腐",),
    "虾": ("虾",),
    "鸡蛋": ("鸡蛋", "蛋液"),
    "羊肉": ("羊肉", "羊排", "羊腩"),
    "鱼": ("鱼",),
}

PAIRING_QUERIES = ["鸡肉", "牛肉", "猪肉", "豆腐", "虾", "鸡蛋", "羊肉", "鱼"]
TOP_K = 8


def load_recipes(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def pairing_truth(
    recipes: list[dict[str, Any]],
    anchor: str,
    is_vegetable,
) -> set[str]:
    """独立计算：与主食材共现过的蔬菜集合。"""

    aliases = ANCHOR_ALIASES[anchor]
    counter: Counter[str] = Counter()
    for recipe in recipes:
        names = [str(item.get("name") or "") for item in recipe.get("ingredients") or []]
        if not any(any(alias in name for alias in aliases) for name in names):
            continue
        for name in names:
            if is_vegetable(name):
                counter[name] += 1
    return {name for name, _ in counter.most_common()}


def precision_at_k(returned: list[str], expected: set[str], k: int = TOP_K) -> float:
    top = [name for name in returned[:k] if name]
    if not top:
        return 0.0
    return sum(1 for name in top if name in expected) / len(top)


def run_experiment(recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from app.config import get_settings
    from app.knowledge_graph.ingredient_classifier import classify_ingredient
    from app.rag.documents import load_recipe_documents
    from app.rag.hybrid import HybridRecipeSearcher
    from app.rag.neo4j_store import Neo4jKnowledgeGraph
    from app.rag.vector_store import MilvusVectorIndex, SentenceTransformerEmbedder

    def is_vegetable(name: str) -> bool:
        return classify_ingredient(name) == "蔬菜"

    settings = get_settings()
    documents = load_recipe_documents(Path(settings.recipe_data_path))
    document_by_id = {doc.recipe_id: doc for doc in documents}
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    vector_index = MilvusVectorIndex(
        uri=settings.milvus_uri,
        collection_name=settings.milvus_collection,
        dimension=settings.embedding_dimension,
        embedder=embedder,
    )
    graph = Neo4jKnowledgeGraph(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    searcher = HybridRecipeSearcher(documents=documents, vector_index=vector_index)

    def _vegetables_from_recipes(records: list[dict[str, Any]]) -> list[str]:
        """模拟传统 RAG：先检索菜谱，再从菜谱食材里筛出蔬菜。"""

        counter: Counter[str] = Counter()
        for record in records:
            for name in record.get("ingredients") or []:
                text = str(name)
                if is_vegetable(text):
                    counter[text] += 1
        return [name for name, _ in counter.most_common(TOP_K)]

    def _vector_based(anchor: str) -> list[str]:
        vector = embedder.encode_query(anchor)
        results = vector_index.client.search(
            collection_name=settings.milvus_collection,
            data=[vector],
            limit=30,
            filter="node_type == 'recipe'",
            output_fields=["parent_id", "name"],
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        )
        recipe_ids = [str(hit["entity"].get("parent_id") or hit["id"]) for hit in results[0]]
        records = [
            {"ingredients": document_by_id[rid].ingredients}
            for rid in recipe_ids
            if rid in document_by_id
        ]
        return _vegetables_from_recipes(records)

    def _bm25_based(anchor: str) -> list[str]:
        records = searcher._bm25_search(anchor, 30)
        return _vegetables_from_recipes(
            [{"ingredients": record.get("ingredients") or []} for record in records]
        )

    def _graph_cooccurrence(anchor: str) -> list[str]:
        rows = graph.recommend_pairings(
            anchor_terms=list(ANCHOR_ALIASES[anchor]),
            target_keywords=["蔬菜"],
            limit=TOP_K,
        )
        return [str(row.get("ingredient") or "") for row in rows]

    def _graph_multi_hop(anchor: str) -> list[str]:
        # 用展开后的别名分组，和检索侧调用方式保持一致。
        from app.tools.unified_recipe_search import _expand_entity_terms

        rows = graph.find_pairing_paths(
            anchor_terms=_expand_entity_terms([anchor]),
            target_category="蔬菜",
            limit=TOP_K,
            max_depth=3,
        )
        # 同一食材可能对应多条路径，按首次出现去重。
        seen: list[str] = []
        for row in rows:
            name = str(row.get("ingredient") or "")
            if name and name not in seen:
                seen.append(name)
        return seen

    configs = [
        ("纯向量 + 食材提取", _vector_based),
        ("纯 BM25 + 食材提取", _bm25_based),
        ("图谱共现统计", _graph_cooccurrence),
        ("图谱多跳路径（本方案）", _graph_multi_hop),
    ]

    summary = []
    for label, fn in configs:
        scores = []
        elapsed = 0.0
        for anchor in PAIRING_QUERIES:
            expected = pairing_truth(recipes, anchor, is_vegetable)
            if not expected:
                continue
            started = time.perf_counter()
            try:
                returned = fn(anchor)
            except Exception as exc:  # noqa: BLE001 - 实验脚本记录失败即可
                print(f"  [{label}] {anchor} 失败：{type(exc).__name__} {exc}")
                returned = []
            elapsed += time.perf_counter() - started
            scores.append(precision_at_k(returned, expected))
        row = {
            "配置": label,
            "Precision@8": sum(scores) / len(scores) if scores else 0.0,
            "查询数": len(scores),
            "平均延迟": elapsed / len(scores) if scores else 0.0,
        }
        summary.append(row)
        print(
            f"{label:24} P@8={row['Precision@8']:.3f} "
            f"查询 {row['查询数']} 延迟 {row['平均延迟'] * 1000:.0f}ms"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/generated/recipes.json"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    recipes = load_recipes(args.input)
    print(f"菜谱 {len(recipes)} 道 | 搭配查询 {len(PAIRING_QUERIES)} 条 | Top-{TOP_K}\n")
    summary = run_experiment(recipes)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n结果已写入 {args.output}")


if __name__ == "__main__":
    main()
