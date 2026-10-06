"""工具契约测试。"""

from app.domain.inventory import (
    recipe_contains_required_ingredients,
    recipe_matches_inventory,
)
from app.tools import (
    MealPlanTool,
    PairingRecommendationTool,
    RecipeDetailTool,
    RecipeSearchTool,
    ReviseMealPlanTool,
    ReviseShoppingListTool,
    ShoppingListTool,
    ToolRegistry,
)
from app.tools.unified_recipe_search import UnifiedRecipeSearchTool


def test_recipe_search_excludes_allergy_ingredient() -> None:
    result = RecipeSearchTool().run(
        {
            "query": "鸡",
            "exclude_ingredients": ["花生"],
            "limit": 10,
        }
    )

    assert result.ok is True
    names = [recipe["name"] for recipe in result.data["recipes"]]
    assert "宫保鸡丁" not in names
    assert "青椒鸡胸肉" in names


def test_recipe_search_respects_total_time() -> None:
    result = RecipeSearchTool().run(
        {
            "query": "低糖",
            "max_minutes": 20,
            "limit": 10,
        }
    )

    assert result.ok is True
    assert all(
        recipe["prep_minutes"] + recipe["cook_minutes"] <= 20 for recipe in result.data["recipes"]
    )


def test_recipe_search_can_enforce_current_inventory() -> None:
    result = RecipeSearchTool().run(
        {
            "query": "",
            "available_ingredients": ["鸡蛋", "西红柿", "食用油", "盐"],
            "inventory_strict": True,
            "limit": 10,
        }
    )

    assert result.ok is True
    assert [recipe["name"] for recipe in result.data["recipes"]] == ["西红柿炒鸡蛋"]


def test_inventory_does_not_treat_fresh_aromatics_as_pantry_staples() -> None:
    assert recipe_matches_inventory(
        ["鸡蛋", "西红柿", "食用油", "盐"],
        ["鸡蛋", "西红柿", "食用油", "盐"],
    )
    assert not recipe_matches_inventory(
        ["鸡蛋", "洋葱", "食用油", "盐"],
        ["鸡蛋", "西红柿", "食用油", "盐"],
    )
    assert not recipe_matches_inventory(
        ["白砂糖：必须水油炒，不加水融化会很慢", "水", "食用油"],
        ["鸡蛋", "西红柿", "食用油", "盐"],
    )


def test_inventory_expands_common_ingredient_aliases() -> None:
    assert recipe_matches_inventory(["花椰菜", "盐"], ["花菜"])
    assert recipe_contains_required_ingredients(["大鸡腿"], ["鸡肉"])


def test_unified_search_merges_hybrid_and_graph_before_rrf() -> None:
    class FakeSearcher:
        def search(self, **kwargs):
            return [
                {
                    "recipe_id": "cauliflower",
                    "name": "清炒花菜",
                    "ingredients": ["花菜", "盐"],
                    "category": "素菜",
                    "difficulty": 2,
                    "estimated_minutes": 15,
                    "source_path": "测试",
                },
                {
                    "recipe_id": "unrelated",
                    "name": "糖拌西红柿",
                    "ingredients": ["西红柿", "糖"],
                    "category": "素菜",
                    "difficulty": 1,
                    "estimated_minutes": 5,
                    "source_path": "测试",
                },
                {
                    "recipe_id": "curry",
                    "name": "日式咖喱饭",
                    "ingredients": ["花椰菜", "土豆", "肉"],
                    "category": "主食",
                    "difficulty": 4,
                    "estimated_minutes": 40,
                    "source_path": "测试",
                },
            ]

    class FakeGraph:
        def find_recipes(self, **kwargs):
            return [
                {
                    "recipe_id": "cauliflower",
                    "name": "清炒花菜",
                    "ingredients": ["花椰菜", "盐"],
                    "categories": ["素菜"],
                    "difficulty": 2,
                    "estimated_minutes": 15,
                    "search_score": 10,
                }
            ]

    result = UnifiedRecipeSearchTool(searcher=FakeSearcher(), graph=FakeGraph()).run(
        {
            "query": "花菜",
            "required_ingredients": ["花菜"],
            "available_ingredients": ["花菜"],
            "limit": 5,
        }
    )

    assert result.ok is True
    assert [item["recipe_id"] for item in result.data["recipes"]] == ["cauliflower"]
    assert "hybrid" in result.data["recipes"][0]["source"]
    assert "graph" in result.data["recipes"][0]["source"]


def test_unified_search_returns_empty_when_hard_constraints_exclude_all() -> None:
    class FakeSearcher:
        def search(self, **kwargs):
            return [
                {
                    "recipe_id": "pork-cauliflower",
                    "name": "干锅花菜",
                    "ingredients": ["花菜", "五花肉"],
                    "category": "荤菜",
                    "difficulty": 3,
                    "estimated_minutes": 30,
                }
            ]

    result = UnifiedRecipeSearchTool(searcher=FakeSearcher()).run(
        {
            "query": "花菜",
            "required_ingredients": ["花菜"],
            "available_ingredients": ["花菜"],
            "inventory_strict": True,
            "limit": 5,
        }
    )

    assert result.ok is True
    assert result.data["recipes"] == []


def test_registry_returns_structured_unknown_tool_error() -> None:
    registry = ToolRegistry()
    result = registry.execute("missing_tool", {})
    assert result.ok is False
    assert "未知工具" in (result.error or "")


def test_meal_plan_repeats_recipes_to_fill_days() -> None:
    result = MealPlanTool().run(
        {
            "days": 3,
            "meals_per_day": 2,
            "recipes": [
                {
                    "recipe_id": "r001",
                    "name": "青椒鸡胸肉",
                    "ingredients": ["鸡胸肉", "青椒"],
                },
                {
                    "recipe_id": "r002",
                    "name": "西红柿炒鸡蛋",
                    "ingredients": ["西红柿", "鸡蛋"],
                },
            ],
        }
    )

    assert result.ok is True
    assert len(result.data["meal_plan"]["days"]) == 3
    assert len(result.data["meal_plan"]["days"][0]["meals"]) == 2


def test_shopping_list_removes_available_ingredients() -> None:
    result = ShoppingListTool().run(
        {
            "required_ingredients": ["鸡胸肉", "青椒", "鸡蛋"],
            "available_ingredients": ["鸡胸肉"],
        }
    )

    assert result.ok is True
    assert [item["name"] for item in result.data["shopping_list"]] == [
        "青椒",
        "鸡蛋",
    ]


def test_shopping_list_excludes_pantry_staples() -> None:
    result = ShoppingListTool().run(
        {
            "required_ingredients": ["鸡胸肉", "青椒", "盐", "食用油", "生抽", "水"],
            "available_ingredients": [],
        }
    )

    assert result.ok is True
    assert [item["name"] for item in result.data["shopping_list"]] == [
        "鸡胸肉",
        "青椒",
    ]


def test_revise_meal_plan_replaces_target_day_with_vegetarian_dish() -> None:
    result = ReviseMealPlanTool().run(
        {
            "instruction": "把第二天换成素菜",
            "meal_plan": {
                "days": [
                    {
                        "day": 1,
                        "meals": [
                            {
                                "recipe_id": "r001",
                                "name": "青椒鸡胸肉",
                                "ingredients": ["鸡胸肉", "青椒"],
                            }
                        ],
                    },
                    {
                        "day": 2,
                        "meals": [
                            {
                                "recipe_id": "r003",
                                "name": "宫保鸡丁",
                                "ingredients": ["鸡胸肉", "花生"],
                            }
                        ],
                    },
                ]
            },
            "recipes": [
                {
                    "recipe_id": "r004",
                    "name": "蒜蓉空心菜",
                    "ingredients": ["空心菜", "大蒜"],
                },
                {
                    "recipe_id": "r005",
                    "name": "咖喱炒蟹",
                    "ingredients": ["青蟹", "咖喱块"],
                },
            ],
            "available_ingredients": ["大蒜"],
        }
    )

    assert result.ok is True
    plan = result.data["meal_plan"]
    assert plan["days"][0]["meals"][0]["name"] == "青椒鸡胸肉"
    assert plan["days"][1]["meals"][0]["name"] == "蒜蓉空心菜"
    shopping_names = [item["name"] for item in result.data["shopping_list"]]
    assert "空心菜" in shopping_names
    assert "大蒜" not in shopping_names


def test_revise_shopping_list_removes_named_ingredient() -> None:
    result = ReviseShoppingListTool().run(
        {
            "instruction": "购物清单里去掉鸡蛋",
            "shopping_list": [
                {"name": "鸡蛋", "reason": "当前库存中没有"},
                {"name": "青椒", "reason": "当前库存中没有"},
            ],
        }
    )

    assert result.ok is True
    assert [item["name"] for item in result.data["shopping_list"]] == ["青椒"]


def test_pairing_recommendation_returns_vegetables_with_evidence() -> None:
    result = PairingRecommendationTool().run(
        {
            "anchor": "鸡肉",
            "target_type": "蔬菜",
            "limit": 5,
        }
    )

    assert result.ok is True
    assert result.data["pairings"]
    assert result.data["pairings"][0]["examples"]


def test_recipe_detail_returns_steps() -> None:
    result = RecipeDetailTool().run({"recipe_id": "r001"})

    assert result.ok is True
    assert result.data["recipe"]["name"] == "青椒鸡胸肉"
    assert result.data["recipe"]["steps"]
