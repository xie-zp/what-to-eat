"""将标准化菜谱转换为可检索文档。"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from app.domain.cuisine import classify_recipe_cuisine
from app.domain.recipe import ParsedRecipe
from app.knowledge_graph.ingredient_classifier import classify_ingredient

MAX_INGREDIENT_NAME_LENGTH = 12


class RecipeChunk(BaseModel):
    """菜谱文档的一个可检索片段，保留指向父菜谱的引用。"""

    chunk_id: str
    parent_id: str
    chunk_type: str
    text: str


class KnowledgeNodeDocument(BaseModel):
    """图谱节点级别的检索文档，用于食材这类非菜谱节点。"""

    node_id: str
    node_type: str
    name: str
    text: str
    category: str = ""
    source_path: str = ""
    related_recipe_ids: list[str] = Field(default_factory=list)


class RecipeDocument(BaseModel):
    """向量检索和关键词检索使用的菜谱文档。"""

    recipe_id: str
    name: str
    text: str
    category: str
    difficulty: int
    estimated_minutes: int | None = None
    ingredients: list[str] = Field(default_factory=list)
    nutrition_tags: list[str] = Field(default_factory=list)
    cuisine_type: str = ""
    source_path: str
    chunks: list[RecipeChunk] = Field(default_factory=list)


def build_recipe_document(recipe: ParsedRecipe) -> RecipeDocument:
    """将一道菜谱转换为检索文本。"""

    ingredient_text = "、".join(item.name for item in recipe.ingredients)
    step_text = "\n".join(f"{index}. {step}" for index, step in enumerate(recipe.steps, start=1))
    overview = (
        f"菜名：{recipe.name}\n"
        f"分类：{recipe.category}\n"
        f"难度：{recipe.difficulty}星\n"
        f"预计时间：{recipe.estimated_minutes or '未标注'} 分钟\n"
        f"热量：{recipe.estimated_calories or '未标注'} 千卡\n"
        f"营养标签：{'、'.join(recipe.nutrition_tags) or '未标注'}\n"
        f"菜系：{recipe.cuisine_type or '未标注'}"
    )
    text = f"{overview}\n食材：{ingredient_text}\n步骤：\n{step_text}"
    ingredients = [item.name for item in recipe.ingredients]
    cuisine_type = recipe.cuisine_type or classify_recipe_cuisine(
        recipe.name,
        recipe.description,
        ingredients,
    )
    return RecipeDocument(
        recipe_id=recipe.recipe_id,
        name=recipe.name,
        text=text,
        category=recipe.category,
        difficulty=recipe.difficulty,
        estimated_minutes=recipe.estimated_minutes,
        ingredients=ingredients,
        nutrition_tags=recipe.nutrition_tags,
        cuisine_type=cuisine_type,
        source_path=recipe.source_path,
        chunks=build_recipe_chunks(
            recipe_id=recipe.recipe_id,
            name=recipe.name,
            overview=overview,
            ingredient_text=ingredient_text,
            step_text=step_text,
        ),
    )


def build_recipe_chunks(
    recipe_id: str,
    name: str,
    overview: str,
    ingredient_text: str,
    step_text: str,
) -> list[RecipeChunk]:
    """把一道菜拆成概述、食材、步骤三个检索片段。"""

    return [
        RecipeChunk(
            chunk_id=f"{recipe_id}#overview",
            parent_id=recipe_id,
            chunk_type="overview",
            text=overview,
        ),
        RecipeChunk(
            chunk_id=f"{recipe_id}#ingredients",
            parent_id=recipe_id,
            chunk_type="ingredients",
            text=f"菜名：{name}\n食材：{ingredient_text}",
        ),
        RecipeChunk(
            chunk_id=f"{recipe_id}#steps",
            parent_id=recipe_id,
            chunk_type="steps",
            text=f"菜名：{name}\n步骤：\n{step_text}",
        ),
    ]


def load_recipe_documents(path: Path) -> list[RecipeDocument]:
    """从标准化菜谱 JSON 加载检索文档。"""

    data = json.loads(path.read_text(encoding="utf-8"))
    return [build_recipe_document(ParsedRecipe.model_validate(item)) for item in data]


def build_ingredient_nodes(
    documents: list[RecipeDocument],
) -> list[KnowledgeNodeDocument]:
    """把菜谱里的食材聚合成图谱节点文档，供食材级检索使用。"""

    grouped: dict[str, list[RecipeDocument]] = {}
    for document in documents:
        for ingredient in document.ingredients:
            name = ingredient.strip()
            if not name or len(name) > MAX_INGREDIENT_NAME_LENGTH:
                continue
            if any(marker in name for marker in ("、", "，", ",", "或", "等")):
                continue
            grouped.setdefault(name, []).append(document)

    nodes = []
    for name, related in grouped.items():
        category = classify_ingredient(name)
        if category == "工具":
            continue
        recipe_names = list(dict.fromkeys(item.name for item in related))
        nodes.append(
            KnowledgeNodeDocument(
                node_id=f"ingredient:{name}",
                node_type="ingredient",
                name=name,
                category=category,
                text=(
                    f"食材：{name}\n"
                    f"类别：{category}\n"
                    f"出现于 {len(recipe_names)} 道菜谱：{'、'.join(recipe_names[:20])}"
                ),
                source_path=related[0].source_path,
                related_recipe_ids=[item.recipe_id for item in related],
            )
        )
    return nodes
