"""统一菜谱检索：Hybrid 召回、图谱召回、硬约束过滤和 RRF 融合。"""

from __future__ import annotations

import logging
from typing import Any

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
        sources = set(params.sources or ["hybrid", "graph"])

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
                )
                candidate_lists.append(
                    ("hybrid", 1.2, list(hybrid_records)),
                )
            except Exception as exc:
                logger.warning("Hybrid 召回失败：%s", exc)
                errors.append(f"hybrid: {exc}")

        if self.graph is not None and "graph" in sources:
            try:
                graph_records = self.graph.find_recipes(
                    query=params.query,
                    category=params.category,
                    max_difficulty=params.max_difficulty,
                    max_minutes=params.max_minutes,
                    exclude_ingredients=params.exclude_ingredients,
                    available_ingredients=params.available_ingredients,
                    inventory_strict=params.inventory_strict,
                    limit=max(params.limit * 4, 20),
                )
                candidate_lists.append(
                    ("graph", 1.0, list(graph_records)),
                )
            except Exception as exc:
                logger.warning("图谱召回失败：%s", exc)
                errors.append(f"graph: {exc}")

        if not candidate_lists:
            return ToolResult(
                ok=False,
                error="所有召回通道都失败：" + "; ".join(errors) if errors else "没有召回通道",
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
        fused = _fuse(normalized_lists, params.limit)
        return ToolResult(
            ok=True,
            data={"recipes": fused},
            evidence_ids=[item["recipe_id"] for item in fused],
        )


def _normalize_record(record: dict[str, Any], source: str) -> dict[str, Any]:
    categories = record.get("categories") or []
    category = record.get("category") or (categories[0] if categories else "未知")
    recipe_id = str(record.get("recipe_id") or "")
    score = float(record.get("rrf_score") or record.get("search_score") or record.get("score") or 0)
    return {
        "recipe_id": recipe_id,
        "name": str(record.get("name") or ""),
        "ingredients": list(record.get("ingredients") or []),
        "steps": list(record.get("steps") or []),
        "nutrition_tags": list(record.get("nutrition_tags") or []),
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

    ordered = sorted(scores, key=lambda recipe_id: scores[recipe_id], reverse=True)
    result = []
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
