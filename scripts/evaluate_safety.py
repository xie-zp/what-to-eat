"""过敏原约束有效性实验。

对比"只在提示词里声明过敏原"和"把过敏原作为硬约束传入检索"两种做法，
统计返回结果中仍包含过敏原的比例（违规率）。

违规判定独立完成：直接检查返回菜谱的食材列表，不依赖任何校验代码。
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

# 覆盖常见且分布较广的过敏原。
ALLERGENS = ["花生", "虾", "鸡蛋", "牛奶", "猪肉"]

# 通用推荐问法：本身不含"排除某种食材"的语义，用于观察系统是否会主动排除过敏原。
GOALS = [
    "推荐几道家常菜",
    "有什么下饭菜推荐",
    "推荐几道简单的菜",
    "推荐几道荤菜",
]

TOP_K = 5


def load_recipes(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_experiment(recipes: list[dict[str, Any]], limit: int = TOP_K) -> list[dict[str, Any]]:
    from app.agent.query_plan import build_query_plan
    from app.config import get_settings
    from app.rag.documents import load_recipe_documents
    from app.rag.hybrid import HybridRecipeSearcher
    from app.rag.neo4j_store import Neo4jKnowledgeGraph
    from app.rag.vector_store import MilvusVectorIndex, SentenceTransformerEmbedder
    from app.tools.unified_recipe_search import UnifiedRecipeSearchTool

    settings = get_settings()
    documents = load_recipe_documents(Path(settings.recipe_data_path))
    ingredients_by_name = {
        str(recipe["name"]): [
            str(item.get("name") or "") for item in recipe.get("ingredients") or []
        ]
        for recipe in recipes
    }
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

    def _contains_allergen(recipe_name: str, allergen: str) -> bool:
        """独立判定：菜谱食材里是否出现该过敏原。"""

        return any(
            allergen in ingredient for ingredient in ingredients_by_name.get(recipe_name, [])
        )

    def _query(goal: str, exclude: list[str]) -> list[str]:
        constraints = build_query_plan(goal).get("constraints") or {}
        result = tool.run(
            {
                "query": goal,
                "category": constraints.get("category"),
                "cuisine": constraints.get("cuisine"),
                "max_minutes": constraints.get("max_minutes"),
                "max_difficulty": constraints.get("max_difficulty"),
                "exclude_ingredients": exclude,
                "sources": ["hybrid", "graph"],
                "strategy": "combined",
                "limit": limit,
            }
        )
        return [item["name"] for item in (result.data or {}).get("recipes", [])]

    summary = []
    modes = [
        ("仅提示词声明（无硬过滤）", False),
        ("过敏原作为硬约束（本方案）", True),
    ]
    for label, use_filter in modes:
        violations = 0
        served = 0
        empty = 0
        for goal, allergen in product(GOALS, ALLERGENS):
            names = _query(goal, [allergen] if use_filter else [])
            if not names:
                empty += 1
                continue
            served += len(names)
            violations += sum(1 for name in names if _contains_allergen(name, allergen))
        total_cases = len(GOALS) * len(ALLERGENS)
        row = {
            "配置": label,
            "违规推荐数": violations,
            "返回结果数": served,
            "违规率": violations / served if served else 0.0,
            "无结果查询数": empty,
            "查询组合数": total_cases,
        }
        summary.append(row)
        print(
            f"{label:26} 违规 {violations}/{served} "
            f"({row['违规率']:.1%}) 无结果 {empty}/{total_cases}"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/generated/recipes.json"))
    parser.add_argument("--limit", type=int, default=TOP_K)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    recipes = load_recipes(args.input)
    print(
        f"菜谱 {len(recipes)} 道 | 过敏原 {len(ALLERGENS)} 个 × 问法 {len(GOALS)} 种 "
        f"= {len(ALLERGENS) * len(GOALS)} 组 | Top-{args.limit}\n"
    )
    summary = run_experiment(recipes, args.limit)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入 {args.output}")


if __name__ == "__main__":
    main()
