"""主食材与候选食材的图搭配推荐工具。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from app.domain.ingredients import clean_ingredient_name
from app.domain.inventory import expand_ingredient_aliases
from app.knowledge_graph.ingredient_classifier import classify_ingredient
from app.rag.neo4j_store import SEARCH_SYNONYM_GROUPS
from app.state import ToolResult

from .base import BaseTool
from .catalog import RECIPES


class PairingRecommendationInput(BaseModel):
    """搭配推荐输入。"""

    anchor: str = Field(min_length=1)
    target_type: str = "蔬菜"
    candidates: list[str] = Field(default_factory=list)
    limit: int = Field(default=8, ge=1, le=20)


def expand_anchor_terms(anchor: str) -> list[str]:
    """展开主食材的同义词。"""

    for trigger, synonyms in SEARCH_SYNONYM_GROUPS.items():
        if trigger in anchor or anchor in trigger:
            return list(synonyms)
    return [anchor]


class PairingRecommendationTool(BaseTool):
    """通过共现关系返回主食材适合搭配的候选食材。"""

    name = "recommend_pairings"
    description = "从知识图谱中统计主食材与其他食材的共现关系，适合回答搭配问题。"
    input_model = PairingRecommendationInput

    def __init__(self, graph=None) -> None:
        self.graph = graph

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """统计与主食材共同出现的蔬菜并保留菜谱来源。"""

        params = self.input_model.model_validate(arguments)
        if params.target_type != "蔬菜":
            return ToolResult(
                ok=False,
                error=f"暂不支持搭配类型：{params.target_type}",
            )

        anchor_terms = expand_anchor_terms(params.anchor)
        candidate_terms = (
            list(
                dict.fromkeys(
                    alias
                    for candidate in params.candidates
                    for alias in expand_ingredient_aliases(candidate)
                )
            )
            if params.candidates
            else list(SEARCH_SYNONYM_GROUPS["蔬菜"])
        )
        target_category = "" if params.candidates else params.target_type
        path_records: list[dict[str, Any]] = []
        if self.graph is not None:
            records = self.graph.recommend_pairings(
                anchor_terms=anchor_terms,
                candidate_terms=candidate_terms,
                target_category=target_category,
                limit=params.limit,
            )
            if target_category:
                # 变长路径查询会额外给出“主食材→菜谱→食材→食材类别”的可解释推理链。
                path_records = self.graph.find_pairing_paths(
                    anchor_terms=anchor_terms,
                    target_category=target_category,
                    limit=params.limit * 2,
                    max_depth=3,
                )
        else:
            records = _recommend_from_catalog(
                anchor_terms=anchor_terms,
                candidate_terms=candidate_terms,
                target_category=target_category,
                limit=params.limit,
            )
        paths_by_ingredient: dict[str, list[str]] = defaultdict(list)
        for path in path_records:
            ingredient = str(path.get("ingredient") or "").strip()
            names = [str(item) for item in (path.get("path_names") or []) if item]
            if not ingredient or len(names) < 2:
                continue
            description = " → ".join(names)
            if description not in paths_by_ingredient[ingredient]:
                paths_by_ingredient[ingredient].append(description)
        enriched_records = []
        for record in records:
            ingredient_name = str(record.get("ingredient") or "")
            enriched_records.append(
                {
                    **record,
                    "graph_paths": paths_by_ingredient.get(ingredient_name, [])[:2],
                }
            )
        records = enriched_records
        records = [
            {**record, "ingredient": cleaned}
            for record in records
            if (cleaned := clean_ingredient_name(str(record.get("ingredient") or "")))
        ]

        source_ids = list(
            dict.fromkeys(
                example["recipe_id"] for record in records for example in record.get("examples", [])
            )
        )
        return ToolResult(
            ok=True,
            data={
                "anchor": params.anchor,
                "target_type": params.target_type,
                "pairings": records,
            },
            evidence_ids=source_ids,
        )


def _recommend_from_catalog(
    anchor_terms: list[str],
    candidate_terms: list[str],
    target_category: str,
    limit: int,
) -> list[dict[str, Any]]:
    """内存菜谱库的搭配统计，保持与 Neo4j 工具相同的输出结构。"""

    examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    frequency: dict[str, int] = defaultdict(int)
    for recipe in RECIPES:
        text = "、".join(recipe.ingredients)
        if not any(term in text for term in anchor_terms):
            continue
        for ingredient in recipe.ingredients:
            if "、" in ingredient or "，" in ingredient:
                continue
            if (target_category and classify_ingredient(ingredient) == target_category) or any(
                term in ingredient for term in candidate_terms
            ):
                frequency[ingredient] += 1
                if len(examples[ingredient]) < 3:
                    examples[ingredient].append(
                        {
                            "recipe_id": recipe.recipe_id,
                            "name": recipe.name,
                            "source": recipe.source,
                        }
                    )

    records = [
        {
            "ingredient": ingredient,
            "ingredient_category": classify_ingredient(ingredient),
            "frequency": count,
            "examples": examples[ingredient],
        }
        for ingredient, count in frequency.items()
    ]
    records.sort(key=lambda item: (-item["frequency"], item["ingredient"]))
    return records[:limit]
