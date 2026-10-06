"""将标准化菜谱转换为知识图谱数据。"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from app.domain.ingredients import clean_ingredient_name
from app.domain.recipe import ParsedRecipe

from .ingredient_classifier import classify_ingredient
from .models import GraphDataset, GraphNode, GraphRelationship

# 烹饪方法按词典匹配，避免用模型自由发挥导致结果不可复现。
COOKING_METHOD_MARKERS = {
    "炒": ("爆炒", "翻炒", "煸炒", "小炒", "炒"),
    "炖": ("炖",),
    "焖": ("焖",),
    "蒸": ("蒸",),
    "煮": ("焯", "汆", "煮"),
    "烤": ("烘烤", "烤"),
    "炸": ("油炸", "炸"),
    "煎": ("煎",),
    "烧": ("红烧", "烧"),
    "拌": ("凉拌", "拌匀", "拌"),
    "腌": ("腌制", "腌"),
    "烘焙": ("烘焙", "烤箱"),
}
MINUTES_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(分钟|min|minutes?)", re.IGNORECASE)
HOURS_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(小时|hour|hours?)", re.IGNORECASE)


def stable_id(prefix: str, value: str) -> str:
    """根据业务名称生成稳定且可重复的节点 ID。"""

    digest = hashlib.sha1(value.strip().encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def detect_cooking_methods(text: str) -> list[str]:
    """从步骤文本里识别烹饪方法。"""

    # “烧热”“烧开”是加热动作而不是“烧”这种烹饪方式，先剔除再匹配。
    normalized = text.replace("烧热", "").replace("烧开", "").replace("烧水", "")
    return [
        method
        for method, markers in COOKING_METHOD_MARKERS.items()
        if any(marker in normalized for marker in markers)
    ]


def extract_step_minutes(text: str) -> int:
    """从步骤文本里提取耗时（分钟），无法识别时返回 -1。"""

    hours = HOURS_PATTERN.search(text)
    if hours:
        return int(float(hours.group(1)) * 60)
    minutes = MINUTES_PATTERN.search(text)
    if minutes:
        return int(float(minutes.group(1)))
    return -1


def build_graph(recipes: Iterable[ParsedRecipe]) -> GraphDataset:
    """将菜谱集合转换为节点和关系。"""

    nodes: dict[str, GraphNode] = {}
    relationships: dict[tuple[str, str, str], GraphRelationship] = {}

    for recipe in recipes:
        recipe_node_id = recipe.recipe_id
        upsert_node(
            nodes,
            GraphNode(
                node_id=recipe_node_id,
                label="Recipe",
                name=recipe.name,
                properties={
                    "description": recipe.description,
                    "difficulty": recipe.difficulty,
                    "estimated_calories": recipe.estimated_calories,
                    "estimated_minutes": recipe.estimated_minutes,
                    "nutrition_tags": recipe.nutrition_tags,
                    "cuisineType": recipe.cuisine_type,
                    "source_path": recipe.source_path,
                },
            ),
        )
        if recipe.cuisine_type:
            cuisine_id = stable_id("cuisine", recipe.cuisine_type)
            upsert_node(
                nodes,
                GraphNode(
                    node_id=cuisine_id,
                    label="Cuisine",
                    name=recipe.cuisine_type,
                ),
            )
            add_relationship(
                relationships,
                GraphRelationship(
                    source_id=recipe_node_id,
                    target_id=cuisine_id,
                    relation_type="BELONGS_TO_CUISINE",
                ),
            )

        category_id = stable_id("category", recipe.category)
        upsert_node(
            nodes,
            GraphNode(
                node_id=category_id,
                label="Category",
                name=recipe.category,
            ),
        )
        add_relationship(
            relationships,
            GraphRelationship(
                source_id=recipe_node_id,
                target_id=category_id,
                relation_type="BELONGS_TO_CATEGORY",
            ),
        )

        difficulty_id = stable_id("difficulty", str(recipe.difficulty))
        upsert_node(
            nodes,
            GraphNode(
                node_id=difficulty_id,
                label="DifficultyLevel",
                name=f"{recipe.difficulty}星",
                properties={"level": recipe.difficulty},
            ),
        )
        add_relationship(
            relationships,
            GraphRelationship(
                source_id=recipe_node_id,
                target_id=difficulty_id,
                relation_type="HAS_DIFFICULTY",
            ),
        )

        for ingredient in recipe.ingredients:
            ingredient_name = clean_ingredient_name(ingredient.name)
            if not ingredient_name:
                continue
            ingredient_id = stable_id("ingredient", ingredient_name)
            ingredient_category = classify_ingredient(ingredient_name)
            upsert_node(
                nodes,
                GraphNode(
                    node_id=ingredient_id,
                    label="Ingredient",
                    name=ingredient_name,
                    properties={"category": ingredient_category},
                ),
            )
            category_node_id = stable_id("ingredient_category", ingredient_category)
            upsert_node(
                nodes,
                GraphNode(
                    node_id=category_node_id,
                    label="IngredientCategory",
                    name=ingredient_category,
                ),
            )
            add_relationship(
                relationships,
                GraphRelationship(
                    source_id=ingredient_id,
                    target_id=category_node_id,
                    relation_type="BELONGS_TO_INGREDIENT_CATEGORY",
                ),
            )
            add_relationship(
                relationships,
                GraphRelationship(
                    source_id=recipe_node_id,
                    target_id=ingredient_id,
                    relation_type="REQUIRES",
                    properties={
                        "raw_text": ingredient.raw_text,
                        "amount_text": ingredient.amount_text,
                    },
                ),
            )

        for index, step in enumerate(recipe.steps, start=1):
            step_id = stable_id("step", f"{recipe.recipe_id}:{index}")
            methods = detect_cooking_methods(step)
            upsert_node(
                nodes,
                GraphNode(
                    node_id=step_id,
                    label="CookingStep",
                    name=f"步骤{index}",
                    properties={
                        "order": index,
                        "description": step,
                        "methods": "、".join(methods),
                        "time_estimate": extract_step_minutes(step),
                    },
                ),
            )
            add_relationship(
                relationships,
                GraphRelationship(
                    source_id=recipe_node_id,
                    target_id=step_id,
                    relation_type="CONTAINS_STEP",
                    properties={"order": index},
                ),
            )

    return GraphDataset(
        nodes=sorted(nodes.values(), key=lambda item: item.node_id),
        relationships=sorted(
            relationships.values(),
            key=lambda item: (
                item.source_id,
                item.relation_type,
                item.target_id,
            ),
        ),
    )


def upsert_node(nodes: dict[str, GraphNode], node: GraphNode) -> None:
    """插入节点，已存在时保留原有节点。"""

    nodes.setdefault(node.node_id, node)


def add_relationship(
    relationships: dict[tuple[str, str, str], GraphRelationship],
    relationship: GraphRelationship,
) -> None:
    """按源节点、关系和目标节点去重。"""

    key = (
        relationship.source_id,
        relationship.relation_type,
        relationship.target_id,
    )
    relationships.setdefault(key, relationship)
