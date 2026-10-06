"""统一菜谱检索：Hybrid 召回、图谱召回、硬约束过滤和 RRF 融合。"""

from __future__ import annotations

import logging
from typing import Any

from app.domain.cuisine import classify_recipe_cuisine
from app.domain.ingredients import contains_high_sugar_ingredient
from app.domain.inventory import (
    expand_ingredient_aliases,
    recipe_contains_any_ingredient,
    recipe_contains_required_ingredients,
    recipe_matches_inventory,
)
from app.state import ToolResult

from .base import BaseTool
from .recipe_search import RecipeSearchInput

logger = logging.getLogger(__name__)


class UnifiedRecipeSearchTool(BaseTool):
    """组合 Hybrid 与图谱的候选，再做确定性过滤和 RRF 排序。"""

    name = "search_recipes"
    description = "使用向量、BM25、图谱和硬约束统一召回菜谱。"
    input_model = RecipeSearchInput

    def __init__(self, searcher=None, graph=None) -> None:
        self.searcher = searcher
        self.graph = graph

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        params = self.input_model.model_validate(arguments)
        candidate_lists: list[tuple[str, float, list[dict[str, Any]]]] = []
        errors: list[str] = []
        degraded_channels: list[str] = []
        sources = set(params.sources or ["hybrid", "graph"])
        weights = _channel_weights(params.strategy)

        if self.searcher is not None and "hybrid" in sources:
            try:
                hybrid_records = self.searcher.search(
                    query=params.query,
                    limit=max(params.limit * 4, 20),
                    category=params.category,
                    max_difficulty=params.max_difficulty,
                    max_minutes=params.max_minutes,
                    exclude_ingredients=params.exclude_ingredients,
                    available_ingredients=params.available_ingredients,
                    inventory_strict=params.inventory_strict,
                    exclude_categories=params.exclude_categories,
                    cuisine=params.cuisine,
                )
                candidate_lists.append(
                    ("hybrid", weights["hybrid"], list(hybrid_records)),
                )
                # 混合检索内部会在向量库不可用时降级为仅 BM25，这里向上暴露。
                degraded_channels.extend(
                    f"hybrid:{name}" for name in getattr(self.searcher, "degraded_channels", [])
                )
            except Exception as exc:
                logger.warning("Hybrid 召回失败：%s", exc)
                errors.append(f"hybrid: {exc}")

        if self.graph is not None and "graph" in sources:
            try:
                graph_records = self.graph.find_recipes(
                    query=params.query,
                    category=params.category,
                    cuisine=params.cuisine,
                    max_difficulty=params.max_difficulty,
                    max_minutes=params.max_minutes,
                    exclude_ingredients=params.exclude_ingredients,
                    available_ingredients=params.available_ingredients,
                    inventory_strict=params.inventory_strict,
                    limit=max(params.limit * 4, 20),
                )
                candidate_lists.append(
                    ("graph", weights["graph"], list(graph_records)),
                )
            except Exception as exc:
                logger.warning("图谱召回失败：%s", exc)
                errors.append(f"graph: {exc}")
                degraded_channels.append("graph")

        if self.graph is not None and "graph" in sources and (params.cuisine or params.category):
            try:
                theme_records = self.graph.find_by_theme(
                    cuisine=params.cuisine or "",
                    category=params.category or "",
                    limit=max(params.limit * 4, 20),
                )
                candidate_lists.append(
                    ("theme", weights["theme"], list(theme_records)),
                )
            except Exception as exc:
                logger.warning("主题级召回失败：%s", exc)
                errors.append(f"theme: {exc}")
                degraded_channels.append("theme")

        if self.graph is not None and self._use_entity_channel(params, sources):
            try:
                entity_records = self.graph.expand_from_ingredients(
                    ingredient_groups=[
                        _expand_entity_terms([term]) for term in params.ingredient_terms
                    ],
                    limit=max(params.limit * 4, 20),
                )
                candidate_lists.append(
                    ("entity", weights["entity"], list(entity_records)),
                )
            except Exception as exc:
                logger.warning("实体级召回失败：%s", exc)
                errors.append(f"entity: {exc}")
                degraded_channels.append("entity")

        if not candidate_lists:
            return ToolResult(
                ok=False,
                error=(
                    "所有召回通道都失败（混合检索内部已尝试降级为 BM25）：" + "; ".join(errors)
                    if errors
                    else "没有可用的召回通道"
                ),
            )

        normalized_lists = []
        for source, weight, records in candidate_lists:
            normalized = []
            for record in records:
                item = _normalize_record(record, source)
                if _passes_constraints(item, params):
                    normalized.append(item)
            normalized_lists.append((source, weight, normalized))

        normalized_lists = _focus_name_matches(normalized_lists, params.required_ingredients)
        fused = _fuse(
            normalized_lists,
            params.limit,
            dietary_goals=params.dietary_goals,
        )
        return ToolResult(
            ok=True,
            data={
                "recipes": fused,
                **({"degraded_channels": degraded_channels} if degraded_channels else {}),
            },
            evidence_ids=[item["recipe_id"] for item in fused],
        )

    @staticmethod
    def _use_entity_channel(params: RecipeSearchInput, sources: set[str]) -> bool:
        """走向图检索且查询里有明确食材词时，额外做一次实体级召回。"""

        if not params.ingredient_terms:
            return False
        if "graph" not in sources:
            return False
        return params.strategy in {None, "graph_rag", "combined"}


def _channel_weights(strategy: str | None) -> dict[str, float]:
    """按检索策略调整各路召回在 RRF 融合中的权重。

    实测向量检索是语义上最稳的一路，图谱、实体、主题召回属于精确信号，
    更适合作为补充。因此任何策略下都让混合检索保持主导，只用权重体现差异。
    """

    if strategy == "graph_rag":
        return {"hybrid": 1.6, "graph": 1.0, "entity": 1.0, "theme": 1.1}
    if strategy == "combined":
        return {"hybrid": 1.5, "graph": 1.0, "entity": 0.9, "theme": 1.0}
    return {"hybrid": 1.2, "graph": 1.0, "entity": 0.9, "theme": 1.0}


def _expand_entity_terms(terms: list[str], max_terms: int = 40) -> list[str]:
    """把食材词展开成图里可能存在的节点名，例如鸡肉 → 鸡腿、鸡胸肉。"""

    expanded = [
        alias
        for term in terms
        for alias in expand_ingredient_aliases(term)
        if alias and alias.strip()
    ]
    return list(dict.fromkeys(expanded))[:max_terms]


def _normalize_record(record: dict[str, Any], source: str) -> dict[str, Any]:
    categories = record.get("categories") or []
    category = record.get("category") or (categories[0] if categories else "未知")
    recipe_id = str(record.get("recipe_id") or "")
    ingredients = list(record.get("ingredients") or [])
    cuisine_type = str(record.get("cuisine_type") or "")
    if not cuisine_type:
        cuisine_type = classify_recipe_cuisine(
            str(record.get("name") or ""),
            str(record.get("description") or ""),
            ingredients,
        )
    score = float(record.get("rrf_score") or record.get("search_score") or record.get("score") or 0)
    return {
        "recipe_id": recipe_id,
        "name": str(record.get("name") or ""),
        "ingredients": ingredients,
        "steps": list(record.get("steps") or []),
        "nutrition_tags": list(record.get("nutrition_tags") or []),
        "cuisine_type": cuisine_type,
        "category": str(category or "未知"),
        "difficulty": int(record.get("difficulty") or 0),
        "estimated_minutes": record.get("estimated_minutes"),
        "prep_minutes": int(record.get("prep_minutes") or 0),
        "cook_minutes": int(record.get("cook_minutes") or 0),
        "source": str(record.get("source") or f"{source}:{recipe_id}"),
        "score": score,
    }


def _passes_constraints(item: dict[str, Any], params: RecipeSearchInput) -> bool:
    ingredients = item.get("ingredients") or []
    if not recipe_contains_required_ingredients(ingredients, params.required_ingredients):
        return False
    if "低糖" in params.dietary_goals and contains_high_sugar_ingredient(ingredients):
        return False
    if params.inventory_strict:
        if not recipe_matches_inventory(
            ingredients,
            params.available_ingredients,
        ):
            return False
        if params.available_ingredients and not recipe_contains_any_ingredient(
            ingredients,
            params.available_ingredients,
        ):
            return False
    if (
        params.available_ingredients
        and not params.inventory_strict
        and not recipe_contains_any_ingredient(ingredients, params.available_ingredients)
    ):
        return False
    if params.category and item.get("category") != params.category:
        return False
    if params.cuisine and params.cuisine != item.get("cuisine_type"):
        return False
    if item.get("category") in set(params.exclude_categories):
        return False
    if params.max_difficulty is not None and item.get("difficulty", 0) > params.max_difficulty:
        return False
    if params.max_minutes is not None:
        minutes = item.get("estimated_minutes")
        if minutes is None or int(minutes) > params.max_minutes:
            return False
    return not any(
        excluded in ingredient
        for excluded in params.exclude_ingredients
        for ingredient in ingredients
        if excluded
    )


def _fuse(
    candidate_lists: list[tuple[str, float, list[dict[str, Any]]]],
    limit: int,
    dietary_goals: list[str] | None = None,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    records: dict[str, dict[str, Any]] = {}
    for _source, weight, candidates in candidate_lists:
        for rank, item in enumerate(candidates, start=1):
            recipe_id = item.get("recipe_id") or ""
            if not recipe_id:
                continue
            scores[recipe_id] = scores.get(recipe_id, 0.0) + weight / (60 + rank)
            if recipe_id not in records:
                records[recipe_id] = dict(item)
            else:
                records[recipe_id] = _merge_record(records[recipe_id], item)

    result = []
    goals = [goal for goal in (dietary_goals or []) if goal]
    ordered = sorted(
        scores,
        key=lambda recipe_id: (
            sum(tag in (records[recipe_id].get("nutrition_tags") or []) for tag in goals),
            scores[recipe_id],
        ),
        reverse=True,
    )
    for recipe_id in ordered[:limit]:
        item = records[recipe_id]
        item["score"] = round(scores[recipe_id], 6)
        result.append(item)
    return result


def _focus_name_matches(
    candidate_lists: list[tuple[str, float, list[dict[str, Any]]]],
    required_ingredients: list[str],
) -> list[tuple[str, float, list[dict[str, Any]]]]:
    """菜名直接命中目标食材时，排除只在配料中出现该食材的宽泛结果。"""

    if not required_ingredients:
        return candidate_lists

    def name_matches(item: dict[str, Any]) -> bool:
        name = str(item.get("name") or "")
        if not name:
            return False
        return any(
            alias in name or name in alias
            for ingredient in required_ingredients
            for alias in expand_ingredient_aliases(ingredient)
        )

    all_items = [item for _, _, candidates in candidate_lists for item in candidates]
    if not any(name_matches(item) for item in all_items):
        return candidate_lists
    return [
        (source, weight, [item for item in candidates if name_matches(item)])
        for source, weight, candidates in candidate_lists
    ]


def _merge_record(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if key in {"score", "source"}:
            continue
        if value not in (None, "", [], {}):
            merged[key] = value
    sources = [str(left.get("source") or ""), str(right.get("source") or "")]
    merged["source"] = "、".join(dict.fromkeys(source for source in sources if source))
    return merged
