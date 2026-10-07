"""检索方案消融实验。

与 evaluate_agent.py 不同，这里不跑完整 Agent 循环，只对比检索层：
同一批真实问法，在不同召回通道组合下 Top-5 的精度和延迟。

答案是**独立于检索实现**计算出来的：直接扫标准化菜谱数据，
不调用任何检索代码，避免"用被测系统评判被测系统"。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

# 手写查询，覆盖五类有区分度的问法；答案由菜谱数据独立推导。
QUERIES: list[dict[str, Any]] = [
    # 食材检索：答案 = 食材列表里含该词的菜谱
    {"goal": "有什么菜可以用西兰花", "match": {"ingredient": "西兰花"}},
    {"goal": "推荐用豆腐做的菜", "match": {"ingredient": "豆腐"}},
    {"goal": "有哪些菜用到了土豆", "match": {"ingredient": "土豆"}},
    {"goal": "推荐用鸡腿做的菜", "match": {"ingredient": "鸡腿"}},
    {"goal": "有什么菜可以用香菇", "match": {"ingredient": "香菇"}},
    {"goal": "推荐几道用虾做的菜", "match": {"ingredient": "虾"}},
    {"goal": "有哪些菜用到了茄子", "match": {"ingredient": "茄子"}},
    {"goal": "用鸡蛋能做什么", "match": {"ingredient": "鸡蛋"}},
    {"goal": "推荐用牛肉做的菜", "match": {"ingredient": "牛肉"}},
    {"goal": "有哪些菜用到了青椒", "match": {"ingredient": "青椒"}},
    {"goal": "推荐用面条做的菜", "match": {"ingredient": "面"}},
    # 分类检索：答案 = 该分类下的菜谱
    {"goal": "推荐几道水产", "match": {"category": "水产"}},
    {"goal": "有什么素菜推荐", "match": {"category": "素菜"}},
    {"goal": "推荐几道汤类", "match": {"category": "汤类"}},
    {"goal": "早餐吃什么", "match": {"category": "早餐"}},
    {"goal": "推荐几道甜品", "match": {"category": "甜品"}},
    {"goal": "推荐几道主食", "match": {"category": "主食"}},
    # 菜系检索：答案 = 该菜系的菜谱
    {"goal": "川菜有哪些特色菜", "match": {"cuisine": "川菜"}},
    {"goal": "粤菜有什么推荐", "match": {"cuisine": "粤菜"}},
    {"goal": "湘菜有哪些", "match": {"cuisine": "湘菜"}},
    {"goal": "鲁菜有哪些特色菜", "match": {"cuisine": "鲁菜"}},
    # 约束组合：答案 = 同时满足分类和时间限制的菜谱
    {"goal": "推荐30分钟内能做完的素菜", "match": {"category": "素菜", "max_minutes": 30}},
    {"goal": "推荐几道简单的水产", "match": {"category": "水产", "max_difficulty": 2}},
    {"goal": "推荐简单的甜品", "match": {"category": "甜品", "max_difficulty": 2}},
    {"goal": "有没有难度不超过2星的素菜", "match": {"category": "素菜", "max_difficulty": 2}},
    {"goal": "推荐40分钟内能做好的荤菜", "match": {"category": "荤菜", "max_minutes": 40}},
    {"goal": "30分钟内能做完的荤菜", "match": {"category": "荤菜", "max_minutes": 30}},
    {"goal": "推荐几道简单的汤类", "match": {"category": "汤类", "max_difficulty": 2}},
    # 模糊语义：答案 = 符合该难度/营养条件的菜谱
    {"goal": "推荐适合新手做的菜", "match": {"max_difficulty": 2, "category": "素菜"}},
    {"goal": "有什么简单快手的菜", "match": {"max_difficulty": 2, "max_minutes": 20}},
    {"goal": "推荐低糖高蛋白的晚餐", "match": {"nutrition_tag": "高蛋白"}},
    {"goal": "适合减脂的菜有哪些", "match": {"nutrition_tag": "低脂"}},
    {"goal": "难度不超过1星的菜", "match": {"max_difficulty": 1}},
]


def load_recipes(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_answer(recipe: dict[str, Any], spec: dict[str, Any]) -> bool:
    """判断单条菜谱是否满足查询的客观条件。"""

    ingredients = [str(item.get("name") or "") for item in recipe.get("ingredients") or []]
    if "ingredient" in spec:
        keyword = str(spec["ingredient"])
        if not any(keyword in name for name in ingredients):
            return False
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
    if "nutrition_tag" in spec:
        tags = [str(tag) for tag in recipe.get("nutrition_tags") or []]
        if spec["nutrition_tag"] not in tags:
            return False
    return True


def build_truth(
    recipes: list[dict[str, Any]],
    queries: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """为每个查询计算标准答案集合。"""

    cases = []
    for query in queries if queries is not None else QUERIES:
        expected = {
            str(recipe["name"]) for recipe in recipes if build_answer(recipe, query["match"])
        }
        cases.append({**query, "expected": expected})
    return cases


TOP_K = 5


def precision_at_k(names: list[str], expected: set[str], k: int = TOP_K) -> float:
    """Top-K 结果里命中标准答案的比例。"""

    top = names[:k]
    if not top:
        return 0.0
    return sum(1 for name in top if name in expected) / len(top)


def _query_params(goal: str) -> dict[str, Any]:
    """用规则解析器把问法转成检索参数，保证实验可复现。"""

    from app.agent.query_plan import build_query_plan

    plan = build_query_plan(goal)
    constraints = plan.get("constraints") or {}
    return {
        "query": goal,
        "category": constraints.get("category"),
        "cuisine": constraints.get("cuisine"),
        "max_minutes": constraints.get("max_minutes"),
        "max_difficulty": constraints.get("max_difficulty"),
        "search_ingredients": constraints.get("search_ingredients") or [],
    }


def run_experiment(cases: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """跑五种召回配置，返回每个配置的指标。"""

    from app.config import get_settings
    from app.rag.documents import load_recipe_documents
    from app.rag.hybrid import HybridRecipeSearcher
    from app.rag.neo4j_store import Neo4jKnowledgeGraph
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
    graph = Neo4jKnowledgeGraph(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    searcher = HybridRecipeSearcher(documents=documents, vector_index=vector_index)

    def vector_only(params: dict[str, Any]) -> list[str]:
        records = vector_index.search(params["query"], limit=limit)
        return [record["name"] for record in records]

    def bm25_only(params: dict[str, Any]) -> list[str]:
        records = searcher._bm25_search(params["query"], limit)
        return [record["name"] for record in records]

    def graph_only(params: dict[str, Any]) -> list[str]:
        records = graph.find_recipes(
            query=params["query"],
            category=params["category"],
            cuisine=params["cuisine"],
            max_minutes=params["max_minutes"],
            max_difficulty=params["max_difficulty"],
            limit=limit,
        )
        return [record["name"] for record in records]

    def hybrid(params: dict[str, Any]) -> list[str]:
        records = searcher.search(
            query=params["query"],
            limit=limit,
            category=params["category"],
            cuisine=params["cuisine"],
            max_minutes=params["max_minutes"],
            max_difficulty=params["max_difficulty"],
        )
        return [record["name"] for record in records]

    def full(params: dict[str, Any]) -> list[str]:
        from app.tools.unified_recipe_search import UnifiedRecipeSearchTool

        tool = UnifiedRecipeSearchTool(searcher=searcher, graph=graph)
        result = tool.run(
            {
                "query": params["query"],
                "category": params["category"],
                "cuisine": params["cuisine"],
                "max_minutes": params["max_minutes"],
                "max_difficulty": params["max_difficulty"],
                "ingredient_terms": params["search_ingredients"],
                "strategy": "combined",
                "sources": ["hybrid", "graph"],
                "limit": limit,
            }
        )
        return [item["name"] for item in result.data.get("recipes", [])]

    configs = [
        ("纯 BM25", bm25_only),
        ("纯向量", vector_only),
        ("纯图谱", graph_only),
        ("向量 + BM25", hybrid),
        ("五路混合（本方案）", full),
    ]

    summary = []
    for label, fn in configs:
        scores = []
        hits = 0
        elapsed = 0.0
        for case in cases:
            params = _query_params(case["goal"])
            started = time.perf_counter()
            try:
                names = fn(params)
            except Exception as exc:  # noqa: BLE001 - 实验脚本记录失败即可
                print(f"  [{label}] {case['goal']} 失败：{type(exc).__name__} {exc}")
                names = []
            elapsed += time.perf_counter() - started
            score = precision_at_k(names, case["expected"])
            scores.append(score)
            if score > 0:
                hits += 1
        summary.append(
            {
                "配置": label,
                "Precision@5": sum(scores) / len(scores) if scores else 0.0,
                "命中查询数": hits,
                "查询总数": len(cases),
                "平均延迟": elapsed / len(cases) if cases else 0.0,
            }
        )
        print(
            f"{label:18} P@5={summary[-1]['Precision@5']:.3f} "
            f"命中 {hits}/{len(cases)} 延迟 {summary[-1]['平均延迟'] * 1000:.0f}ms"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/generated/recipes.json"))
    parser.add_argument("--limit", type=int, default=TOP_K)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    recipes = load_recipes(args.input)
    cases = build_truth(recipes)
    print(f"菜谱 {len(recipes)} 道 | 查询 {len(cases)} 条 | Top-{args.limit}\n")
    empty = [c["goal"] for c in cases if not c["expected"]]
    if empty:
        print(f"警告：{len(empty)} 条查询没有标准答案：{empty}\n")

    summary = run_experiment(cases, args.limit)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n结果已写入 {args.output}")


if __name__ == "__main__":
    main()
