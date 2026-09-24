"""向量检索、BM25 与 RRF 融合。"""

from __future__ import annotations

from typing import Any

import jieba
from rank_bm25 import BM25Okapi

from app.domain.inventory import expand_ingredient_aliases, recipe_matches_inventory

from .documents import RecipeDocument

CHINESE_STOPWORDS = set(
    """
    的 了 和 是 在 我 有 就 不 也 都 还 这 那 一 个 与 及 等
    什么 怎么 如何 哪些 哪个 请 请问 推荐 一道 可以 需要
    """.split()
)


def tokenize(text: str) -> list[str]:
    """使用 jieba 分词并过滤停用词。"""

    return [
        token
        for token in jieba.lcut(text)
        if token.strip() and token not in CHINESE_STOPWORDS and not token.isspace()
    ]


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    top_k: int,
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    """使用 RRF 融合多路检索结果。"""

    scores: dict[str, float] = {}
    records: dict[str, dict[str, Any]] = {}
    sources: dict[str, set[str]] = {}

    for ranked in ranked_lists:
        for rank, record in enumerate(ranked, start=1):
            recipe_id = record["recipe_id"]
            source = record.get("retrieval_source", "unknown")
            scores[recipe_id] = scores.get(recipe_id, 0.0) + 1.0 / (rank_constant + rank)
            records.setdefault(recipe_id, record)
            sources.setdefault(recipe_id, set()).add(source)

    ordered_ids = sorted(
        scores,
        key=lambda recipe_id: scores[recipe_id],
        reverse=True,
    )
    result = []
    for recipe_id in ordered_ids[:top_k]:
        record = dict(records[recipe_id])
        record["rrf_score"] = scores[recipe_id]
        record["retrieval_sources"] = sorted(sources[recipe_id])
        result.append(record)
    return result


class HybridRecipeSearcher:
    """组合向量检索与 BM25 检索。"""

    def __init__(
        self,
        documents: list[RecipeDocument],
        vector_index: Any,
    ) -> None:
        self.documents = documents
        self.vector_index = vector_index
        self.document_by_id = {document.recipe_id: document for document in documents}
        self.bm25 = BM25Okapi([tokenize(document.text) for document in documents])

    def search(
        self,
        query: str,
        limit: int = 5,
        category: str | None = None,
        max_difficulty: int | None = None,
        max_minutes: int | None = None,
        exclude_ingredients: list[str] | None = None,
        available_ingredients: list[str] | None = None,
        inventory_strict: bool = False,
    ) -> list[dict[str, Any]]:
        """执行混合检索。"""

        candidate_k = 1000 if inventory_strict else max(limit * 4, 20)
        excluded = [item.strip() for item in (exclude_ingredients or []) if item and item.strip()]

        vector_records = self.vector_index.search(query, limit=candidate_k)
        for record in vector_records:
            record["retrieval_source"] = "vector"

        bm25_records = self._bm25_search(query, candidate_k)

        vector_records = self._filter_records(
            vector_records,
            category,
            max_difficulty,
            max_minutes,
            excluded,
            available_ingredients or [],
            inventory_strict,
        )
        bm25_records = self._filter_records(
            bm25_records,
            category,
            max_difficulty,
            max_minutes,
            excluded,
            available_ingredients or [],
            inventory_strict,
        )

        return reciprocal_rank_fusion(
            [vector_records, bm25_records],
            top_k=limit,
        )

    def _bm25_search(
        self,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        tokens = list(
            dict.fromkeys(alias for token in tokens for alias in expand_ingredient_aliases(token))
        )

        scores = self.bm25.get_scores(tokens)
        indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:limit]
        records = []
        for index in indexes:
            if scores[index] <= 0:
                continue
            document = self.documents[index]
            records.append(
                {
                    "recipe_id": document.recipe_id,
                    "score": float(scores[index]),
                    "name": document.name,
                    "text": document.text,
                    "category": document.category,
                    "difficulty": document.difficulty,
                    "estimated_minutes": document.estimated_minutes,
                    "ingredients": document.ingredients,
                    "source_path": document.source_path,
                    "retrieval_source": "bm25",
                }
            )
        return records

    @staticmethod
    def _filter_records(
        records: list[dict[str, Any]],
        category: str | None,
        max_difficulty: int | None,
        max_minutes: int | None,
        excluded: list[str],
        available_ingredients: list[str],
        inventory_strict: bool,
    ) -> list[dict[str, Any]]:
        filtered = []
        for record in records:
            if category and record.get("category") != category:
                continue
            if max_difficulty is not None and int(record.get("difficulty") or 0) > max_difficulty:
                continue
            if max_minutes is not None:
                estimated_minutes = record.get("estimated_minutes")
                if estimated_minutes is None or int(estimated_minutes) > max_minutes:
                    continue
            ingredients = record.get("ingredients") or []
            if any(allergy in ingredient for allergy in excluded for ingredient in ingredients):
                continue
            if inventory_strict and not recipe_matches_inventory(
                ingredients,
                available_ingredients,
            ):
                continue
            filtered.append(record)
        return filtered
