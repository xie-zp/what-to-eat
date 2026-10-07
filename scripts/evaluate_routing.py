"""查询路由增益实验。

对比三种召回策略在同一批查询上的精度与延迟：
1. 固定只走混合检索（向量 + BM25）
2. 固定走混合 + 图谱
3. 按查询复杂度自动路由（本项目方案）

查询集与标准答案复用 evaluate_retrieval.py，保证跨实验可比。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

# 补充的复杂查询：这些问法会触发图谱与组合检索，用于检验路由是否真的分流。
# 标准答案仍由菜谱字段独立推导。
COMPLEX_QUERIES: list[dict[str, Any]] = [
    {
        "goal": "如果我只有鸡肉和蔬菜，能做什么菜，最好是不同菜系的",
        "match": {"ingredient": "鸡"},
    },
    {
        "goal": "哪些菜可以用豆腐替代肉类，并且保持相似的口感",
        "match": {"ingredient": "豆腐"},
    },
    {"goal": "为什么川菜用花椒而不是胡椒", "match": {"cuisine": "川菜"}},
    {"goal": "推荐低糖高蛋白的晚餐", "match": {"nutrition_tag": "高蛋白"}},
    {
        "goal": "30分钟内、难度不超过2星的素菜",
        "match": {"category": "素菜", "max_minutes": 30, "max_difficulty": 2},
    },
    {"goal": "有什么菜可以用豆腐替代肉类", "match": {"ingredient": "豆腐"}},
    {"goal": "有哪些适合新手的快手菜", "match": {"max_difficulty": 2, "max_minutes": 20}},
    {"goal": "推荐几道简单的水产，最好不超过30分钟", "match": {"category": "水产"}},
]


def run_experiment(limit: int = 5) -> list[dict[str, Any]]:
    from app.agent.query_plan import build_query_plan
    from app.agent.query_router import route_query
    from app.config import get_settings
    from app.rag.documents import load_recipe_documents
    from app.rag.hybrid import HybridRecipeSearcher
    from app.rag.neo4j_store import Neo4jKnowledgeGraph
    from app.rag.vector_store import MilvusVectorIndex, SentenceTransformerEmbedder
    from app.tools.unified_recipe_search import UnifiedRecipeSearchTool

    from scripts.evaluate_retrieval import QUERIES, build_truth, load_recipes, precision_at_k

    settings = get_settings()
    recipes = load_recipes(Path(settings.recipe_data_path))
    cases = build_truth(recipes, [*QUERIES, *COMPLEX_QUERIES])

    documents = load_recipe_documents(Path(settings.recipe_data_path))
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
    tool = UnifiedRecipeSearchTool(searcher=searcher, graph=graph)

    def _params(goal: str) -> dict[str, Any]:
        constraints = build_query_plan(goal).get("constraints") or {}
        return {
            "query": goal,
            "category": constraints.get("category"),
            "cuisine": constraints.get("cuisine"),
            "max_minutes": constraints.get("max_minutes"),
            "max_difficulty": constraints.get("max_difficulty"),
            "ingredient_terms": constraints.get("search_ingredients") or [],
        }

    def _search(params: dict[str, Any], sources: list[str], strategy: str) -> list[str]:
        result = tool.run({**params, "sources": sources, "strategy": strategy, "limit": limit})
        return [item["name"] for item in result.data.get("recipes", [])]

    def fixed_hybrid(params: dict[str, Any]) -> list[str]:
        return _search(params, ["hybrid"], "hybrid_traditional")

    def fixed_combined(params: dict[str, Any]) -> list[str]:
        return _search(params, ["hybrid", "graph"], "combined")

    def routed(params: dict[str, Any]) -> list[str]:
        analysis = route_query(
            params["query"], build_query_plan(params["query"]).get("constraints")
        )
        strategy = analysis.strategy.value
        sources = ["hybrid"] if strategy == "hybrid_traditional" else ["hybrid", "graph"]
        return _search(params, sources, strategy)

    configs = [
        ("固定混合检索", fixed_hybrid),
        ("固定混合 + 图谱", fixed_combined),
        ("按策略路由（本方案）", routed),
    ]

    summary = []
    route_distribution: dict[str, int] = {}
    for case in cases:
        analysis = route_query(case["goal"])
        route_distribution[analysis.strategy.value] = (
            route_distribution.get(analysis.strategy.value, 0) + 1
        )

    for label, fn in configs:
        scores = []
        elapsed = 0.0
        for case in cases:
            params = _params(case["goal"])
            started = time.perf_counter()
            try:
                names = fn(params)
            except Exception as exc:  # noqa: BLE001 - 实验脚本记录失败即可
                print(f"  [{label}] {case['goal']} 失败：{type(exc).__name__} {exc}")
                names = []
            elapsed += time.perf_counter() - started
            scores.append(precision_at_k(names, case["expected"], limit))
        row = {
            "配置": label,
            f"Precision@{limit}": sum(scores) / len(scores) if scores else 0.0,
            "查询数": len(cases),
            "平均延迟": elapsed / len(cases) if cases else 0.0,
        }
        summary.append(row)
        print(
            f"{label:22} P@{limit}={row[f'Precision@{limit}']:.3f} "
            f"延迟 {row['平均延迟'] * 1000:.0f}ms"
        )
    print(f"\n路由决策分布：{route_distribution}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = run_experiment(args.limit)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入 {args.output}")


if __name__ == "__main__":
    main()
