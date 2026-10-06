"""知识图谱构建测试。"""

from pathlib import Path

from app.knowledge_graph import build_graph
from app.knowledge_graph.builder import (
    detect_cooking_methods,
    detect_cooking_tools,
    extract_step_minutes,
)
from app.knowledge_graph.ingredient_classifier import classify_ingredient

from scripts.parse_howtocook import parse_directory

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "howtocook"


def test_build_graph_creates_expected_node_types() -> None:
    recipes = parse_directory(FIXTURE_ROOT)
    graph = build_graph(recipes)

    labels = {node.label for node in graph.nodes}
    assert labels == {
        "Recipe",
        "Ingredient",
        "CookingStep",
        "Category",
        "IngredientCategory",
        "DifficultyLevel",
    }

    assert len([node for node in graph.nodes if node.label == "Recipe"]) == 1
    assert len([node for node in graph.nodes if node.label == "Ingredient"]) == 5
    assert len([node for node in graph.nodes if node.label == "CookingStep"]) == 3


def test_build_graph_creates_expected_relations() -> None:
    graph = build_graph(parse_directory(FIXTURE_ROOT))
    relation_types = [item.relation_type for item in graph.relationships]

    assert relation_types.count("REQUIRES") == 5
    assert relation_types.count("CONTAINS_STEP") == 3
    assert relation_types.count("BELONGS_TO_CATEGORY") == 1
    assert relation_types.count("BELONGS_TO_INGREDIENT_CATEGORY") == 5
    assert relation_types.count("HAS_DIFFICULTY") == 1


def test_ingredient_classifier_covers_graph_reasoning_categories() -> None:
    assert classify_ingredient("鸡胸肉") == "肉类"
    assert classify_ingredient("西兰花") == "蔬菜"
    assert classify_ingredient("微波炉") == "工具"
    assert classify_ingredient("生抽") == "调料"


def test_detect_cooking_methods_ignores_heating_actions() -> None:
    assert detect_cooking_methods("起锅，大火烧热转小火") == []
    assert detect_cooking_methods("倒入植物油，下入鸡丁翻炒") == ["炒"]
    assert detect_cooking_methods("盖上锅盖焖 2 分钟") == ["焖"]
    assert set(detect_cooking_methods("鸡丁加入盐和淀粉搅拌均匀腌制")) == {"拌", "腌"}


def test_extract_step_minutes_handles_minutes_and_hours() -> None:
    assert extract_step_minutes("泡于清水 10 分钟") == 10
    assert extract_step_minutes("小火炖 1 小时") == 60
    assert extract_step_minutes("切成小块") == -1


def test_detect_cooking_tools_extracts_step_equipment() -> None:
    assert detect_cooking_tools("放入干辣椒，用炒锅翻炒") == ["炒锅"]
    assert detect_cooking_tools("用菜刀切成小块，放在案板上") == ["菜刀", "案板"]
    # “起锅”是动作描述，不是具体工具。
    assert detect_cooking_tools("起锅，大火烧热转小火") == []
