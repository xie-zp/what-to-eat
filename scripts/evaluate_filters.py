"""元数据过滤增益实验。

对比“只做语义检索”和“语义检索 + 元数据硬过滤”在带约束查询上的表现。
标准答案独立计算：直接按菜谱字段筛选，不调用检索代码。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

# 带明确元数据条件的查询；答案 = 同时满足所有字段条件的菜谱。
FILTER_QUERIES: list[dict[str, Any]] = [
    {"goal": "推荐30分钟内能做完的素菜", "match": {"category": "素菜", "max_minutes": 30}},
    {"goal": "推荐40分钟内能做好的荤菜", "match": {"category": "荤菜", "max_minutes": 40}},
    {"goal": "有没有难度不超过2星的素菜", "match": {"category": "素菜", "max_difficulty": 2}},
    {"goal": "推荐几道简单的水产", "match": {"category": "水产", "max_difficulty": 2}},
    {"goal": "推荐几道简单的汤类", "match": {"category": "汤类", "max_difficulty": 2}},
    {"goal": "推荐几道甜品", "match": {"category": "甜品"}},
    {"goal": "推荐几道主食", "match": {"category": "主食"}},
    {"goal": "有什么素菜推荐", "match": {"category": "素菜"}},
    {"goal": "推荐几道水产", "match": {"category": "水产"}},
    {"goal": "湘菜有哪些", "match": {"cuisine": "湘菜"}},
    {"goal": "粤菜有什么推荐", "match": {"cuisine": "粤菜"}},
    {"goal": "川菜有哪些特色菜", "match": {"cuisine": "川菜"}},
]

TOP_K = 5


def load_recipes(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def matches(recipe: dict[str, Any], spec: dict[str, Any]) -> bool:
    if "category" in spec and recipe.get("category") != spec["category"]:
        return False
    if "cuisine" in spec and recipe.get("cuisine_type") != spec["cuisine"]:
        return False
    if "max_minutes" in spec:
        minutes = recipe.get("estimated_minutes")
        if minutes is None or int(minutes) > int(spec["max_minutes"]):
            return False
    if "max_difficulty" in spec:
        if int(recipe.get("difficulty") or 0) > int(spec["max_difficulty"]):
            return False
    return True


def run_experiment(recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from app.agent.query_plan import build_query_plan
    from app.config import get_settings
    from app.rag.documents import load_recipe_documents
    from app.rag.hybrid import HybridRecipeSearcher
    from app.rag.vector_store import MilvusVectorIndex, SentenceTransformerEmbedder

    settings = get_settings()
    documents = load_recipe_documents(Path(settings.recipe_data_path))
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    vector_index = MilvusVectorIndex(
        uri=settings.milvus_uri,
        collection_name=settings.milvus_collection,
        dimension=settings.embedding_dimension,
        embedder=embedder,
    )
    searcher = HybridRecipeSearcher(documents=documents, vector_index=vector_index)

    def without_filter(goal: str, _constraints: dict[str, Any]) -> list[str]:
        """只做混合检索，不施加任何元数据过滤。"""

        records = searcher.search(query=goal, limit=TOP_K)
        return [record["name"] for record in records]

    def with_filter(goal: str, constraints: dict[str, Any]) -> list[str]:
        """混合检索 + 分类/菜系/时间/难度的硬过滤。"""

        records = searcher.search(
            query=goal,
            limit=TOP_K,
            category=constraints.get("category"),
            cuisine=constraints.get("cuisine"),
            max_minutes=constraints.get("max_minutes"),
            max_difficulty=constraints.get("max_difficulty"),
        )
        return [record["name"] for record in records]

    truth_by_goal = {
        query["goal"]: {recipe["name"] for recipe in recipes if matches(recipe, query["match"])}
        for query in FILTER_QUERIES
    }

    summary = []
    for label, fn in (("仅语义检索", without_filter), ("语义检索 + 元数据过滤", with_filter)):
        rates = []
        counts = []
        elapsed = 0.0
        for query in FILTER_QUERIES:
            goal = query["goal"]
            expected = truth_by_goal[goal]
            constraints = build_query_plan(goal).get("constraints") or {}
            started = time.perf_counter()
            try:
                names = fn(goal, constraints)
            except Exception as exc:  # noqa: BLE001 - 实验脚本记录失败即可
                print(f"  [{label}] {goal} 失败：{type(exc).__name__} {exc}")
                names = []
            elapsed += time.perf_counter() - started
            if not names:
                rates.append(0.0)
                counts.append(0)
                continue
            hit = sum(1 for name in names if name in expected)
            rates.append(hit / len(names))
            counts.append(hit)
        row = {
            "配置": label,
            "约束满足率": sum(rates) / len(rates) if rates else 0.0,
            "平均合规条数": sum(counts) / len(counts) if counts else 0.0,
            "查询数": len(FILTER_QUERIES),
            "平均延迟": elapsed / len(FILTER_QUERIES) if FILTER_QUERIES else 0.0,
        }
        summary.append(row)
        print(
            f"{label:22} 约束满足率={row['约束满足率']:.3f} "
            f"平均合规 {row['平均合规条数']:.2f}/{TOP_K} 延迟 {row['平均延迟'] * 1000:.0f}ms"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/generated/recipes.json"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    recipes = load_recipes(args.input)
    print(f"菜谱 {len(recipes)} 道 | 带约束查询 {len(FILTER_QUERIES)} 条 | Top-{TOP_K}\n")
    summary = run_experiment(recipes)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入 {args.output}")


if __name__ == "__main__":
    main()
