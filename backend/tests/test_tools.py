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
from app.tools.unified_recipe_search import UnifiedRecipeSearchTool, _fuse


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


def test_inventory_matches_broad_ingredient_categories() -> None:
    assert recipe_matches_inventory(["西兰花", "食用油", "盐"], ["蔬菜"])
    assert recipe_matches_inventory(["鸡胸肉", "青椒", "生抽"], ["鸡肉", "蔬菜"])
    assert recipe_contains_required_ingredients(["西兰花", "鸡胸肉"], ["鸡肉", "蔬菜"])
    assert not recipe_matches_inventory(["牛排", "食用油", "盐"], ["蔬菜"])


def test_pairing_tool_attaches_multi_hop_paths() -> None:
    class FakeGraph:
        def recommend_pairings(self, **_kwargs):
            return [
                {
                    "ingredient": "青椒",
                    "ingredient_category": "蔬菜",
                    "frequency": 2,
                    "examples": [
                        {"recipe_id": "r1", "name": "黄焖鸡", "source": "neo4j:r1"},
                    ],
                }
            ]

        def find_pairing_paths(self, **_kwargs):
            return [
                {
                    "anchor": "鸡腿",
                    "target": "蔬菜",
                    "ingredient": "青椒",
                    "path_len": 3,
                    "relevance": 3.2,
                    "path_names": ["鸡腿", "黄焖鸡", "青椒", "蔬菜"],
                    "path_relations": [
                        "REQUIRES",
                        "REQUIRES",
                        "BELONGS_TO_INGREDIENT_CATEGORY",
                    ],
                }
            ]

    result = PairingRecommendationTool(graph=FakeGraph()).run(
        {"anchor": "鸡肉", "target_type": "蔬菜", "limit": 5}
    )

    assert result.ok is True
    pairing = result.data["pairings"][0]
    assert pairing["graph_paths"] == ["鸡腿 → 黄焖鸡 → 青椒 → 蔬菜"]


def test_unified_search_uses_entity_channel_for_graph_strategy() -> None:
    class FakeGraph:
        def __init__(self) -> None:
            self.entity_calls: list[dict] = []

        def find_recipes(self, **_kwargs):
            return []

        def expand_from_ingredients(self, **kwargs):
            self.entity_calls.append(kwargs)
            return [
                {
                    "recipe_id": "r1",
                    "name": "黄焖鸡",
                    "ingredients": ["鸡腿", "土豆", "青椒"],
                    "category": "荤菜",
                    "difficulty": 2,
                    "cuisine_type": "",
                    "source": "neo4j:r1",
                }
            ]

    graph = FakeGraph()
    result = UnifiedRecipeSearchTool(searcher=None, graph=graph).run(
        {
            "query": "鸡肉 蔬菜",
            "sources": ["hybrid", "graph"],
            "strategy": "graph_rag",
            "ingredient_terms": ["鸡肉"],
            "limit": 5,
        }
    )

    assert result.ok is True
    assert graph.entity_calls, "走向图检索且有食材词时应触发实体级召回"
    assert [item["name"] for item in result.data["recipes"]] == ["黄焖鸡"]


def test_unified_search_skips_entity_channel_without_ingredient_terms() -> None:
    class FakeGraph:
        def __init__(self) -> None:
            self.entity_calls: list[dict] = []

        def find_recipes(self, **_kwargs):
            return []

        def expand_from_ingredients(self, **kwargs):
            self.entity_calls.append(kwargs)
            return []

    graph = FakeGraph()
    UnifiedRecipeSearchTool(searcher=None, graph=graph).run(
        {
            "query": "推荐一道菜",
            "sources": ["hybrid", "graph"],
            "strategy": "graph_rag",
            "limit": 5,
        }
    )

    assert graph.entity_calls == []


def test_fusion_rewards_recipe_hit_by_multiple_chunks() -> None:
    """同一道菜被多个片段命中时，累计分数应高于只命中一次的菜。"""

    candidates = [
        {"recipe_id": "a", "name": "菜A"},
        {"recipe_id": "b", "name": "菜B"},
        {"recipe_id": "b", "name": "菜B"},
    ]

    fused = _fuse([("hybrid", 1.0, candidates)], limit=2)

    assert [item["recipe_id"] for item in fused] == ["b", "a"]


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


def test_unified_search_excludes_non_meal_categories_and_boosts_goals() -> None:
    class FakeSearcher:
        def search(self, **kwargs):
            return [
                {
                    "recipe_id": "sauce",
                    "name": "糖醋汁",
                    "ingredients": ["糖", "醋"],
                    "category": "调料",
                    "difficulty": 1,
                    "nutrition_tags": [],
                },
                {
                    "recipe_id": "drink",
                    "name": "长岛冰茶",
                    "ingredients": ["酒", "可乐"],
                    "category": "饮料",
                    "difficulty": 2,
                    "nutrition_tags": [],
                },
                {
                    "recipe_id": "dish",
                    "name": "香煎鸡胸肉",
                    "ingredients": ["鸡胸肉"],
                    "category": "荤菜",
                    "difficulty": 2,
                    "nutrition_tags": ["高蛋白", "低糖"],
                },
                {
                    "recipe_id": "sweet",
                    "name": "醪糟小汤圆",
                    "ingredients": ["小汤圆", "醪糟", "白砂糖"],
                    "category": "主食",
                    "difficulty": 1,
                    "nutrition_tags": [],
                },
            ]

    result = UnifiedRecipeSearchTool(searcher=FakeSearcher()).run(
        {
            "query": "低糖 高蛋白 晚餐",
            "dietary_goals": ["低糖", "高蛋白"],
            "exclude_categories": ["调料", "饮料", "甜品"],
            "limit": 5,
        }
    )

    assert result.ok is True
    assert [item["name"] for item in result.data["recipes"]] == ["香煎鸡胸肉"]


def test_unified_search_filters_by_cuisine() -> None:
    class FakeSearcher:
        def search(self, **kwargs):
            return [
                {
                    "recipe_id": "mapo",
                    "name": "麻婆豆腐",
                    "ingredients": ["豆腐", "豆瓣酱"],
                    "category": "素菜",
                    "difficulty": 2,
                    "cuisine_type": "川菜",
                },
                {
                    "recipe_id": "shrimp",
                    "name": "白灼虾",
                    "ingredients": ["虾", "葱", "姜"],
                    "category": "水产",
                    "difficulty": 1,
                    "cuisine_type": "粤菜",
                },
            ]

    result = UnifiedRecipeSearchTool(searcher=FakeSearcher()).run(
        {
            "query": "粤菜",
            "cuisine": "粤菜",
            "limit": 5,
        }
    )

    assert result.ok is True
    assert [item["name"] for item in result.data["recipes"]] == ["白灼虾"]
    assert result.data["recipes"][0]["cuisine_type"] == "粤菜"


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


def test_shopping_list_keeps_seasonings_and_excludes_equipment() -> None:
    result = ShoppingListTool().run(
        {
            "required_ingredients": [
                "鸡胸肉",
                "青椒",
                "盐",
                "食用油",
                "生抽",
                "水",
                "秒表",
                "菜刀一个",
            ],
            "available_ingredients": [],
        }
    )

    assert result.ok is True
    assert [item["name"] for item in result.data["shopping_list"]] == [
        "鸡胸肉",
        "青椒",
        "盐",
        "食用油",
        "酱油",
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
