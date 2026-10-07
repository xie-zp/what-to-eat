"""检索降级鲁棒性实验。

模拟向量库（Milvus）不可用，验证系统是否仍能返回结果，以及召回质量的下降幅度。
对比：正常模式（向量 + BM25 + 图谱）与降级模式（向量不可用，仅 BM25 + 图谱）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


class BrokenVectorIndex:
    """模拟向量库不可用。"""

    @staticmethod
    def search(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("milvus unavailable")


def run_experiment(limit: int = 5) -> list[dict[str, Any]]:
    from app.agent.query_plan import build_query_plan
    from app.config import get_settings
    from app.rag.documents import load_recipe_documents
    from app.rag.hybrid import HybridRecipeSearcher
    from app.rag.neo4j_store import Neo4jKnowledgeGraph
    from app.rag.vector_store import MilvusVectorIndex, SentenceTransformerEmbedder
    from app.tools.unified_recipe_search import UnifiedRecipeSearchTool

    from scripts.evaluate_retrieval import build_truth, load_recipes, precision_at_k

    settings = get_settings()
    recipes = load_recipes(Path(settings.recipe_data_path))
    cases = build_truth(recipes)
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

    configs = [
        ("正常模式", vector_index),
        ("降级模式（向量库不可用）", BrokenVectorIndex()),
    ]

    summary = []
    for label, index in configs:
        searcher = HybridRecipeSearcher(documents=documents, vector_index=index)
        tool = UnifiedRecipeSearchTool(searcher=searcher, graph=graph)
        scores = []
        served = 0
        elapsed = 0.0
        for case in cases:
            params = _params(case["goal"])
            started = time.perf_counter()
            try:
                result = tool.run(
                    {
                        **params,
                        "sources": ["hybrid", "graph"],
                        "strategy": "combined",
                        "limit": limit,
                    }
                )
                names = [item["name"] for item in (result.data or {}).get("recipes", [])]
            except Exception as exc:  # noqa: BLE001 - 实验脚本记录失败即可
                print(f"  [{label}] {case['goal']} 失败：{type(exc).__name__} {exc}")
                names = []
            elapsed += time.perf_counter() - started
            if names:
                served += 1
            scores.append(precision_at_k(names, case["expected"], limit))
        row = {
            "配置": label,
            "可用率": served / len(cases) if cases else 0.0,
            f"Precision@{limit}": sum(scores) / len(scores) if scores else 0.0,
            "查询数": len(cases),
            "平均延迟": elapsed / len(cases) if cases else 0.0,
        }
        summary.append(row)
        print(
            f"{label:26} 可用率={row['可用率']:.1%} "
            f"P@{limit}={row[f'Precision@{limit}']:.3f} 延迟 {row['平均延迟'] * 1000:.0f}ms"
        )
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
