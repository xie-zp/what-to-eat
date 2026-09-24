"""知识图谱构建测试。"""

from pathlib import Path

from app.knowledge_graph import build_graph
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
