"""Agent 图回归测试。"""

import app.agent.context as context_module
import app.agent.planner as planner_module
from app.agent import build_recipe_agent
from app.agent.nodes import respond_node
from app.agent.planner import LLMRecipePlanner, RuleBasedPlanner
from app.state import RecipeEvidence, create_initial_state
from app.tools import (
    MealPlanTool,
    NutritionTool,
    PairingRecommendationTool,
    RecipeDetailTool,
    RecipeSearchTool,
    ReviseMealPlanTool,
    ReviseShoppingListTool,
    ShoppingListTool,
    ToolRegistry,
)
from app.tools.unified_recipe_search import UnifiedRecipeSearchTool


def build_test_graph():
    registry = ToolRegistry()
    registry.register(RecipeSearchTool())
    registry.register(RecipeDetailTool())
    registry.register(MealPlanTool())
    registry.register(ShoppingListTool())
    registry.register(ReviseMealPlanTool())
    registry.register(ReviseShoppingListTool())
    registry.register(PairingRecommendationTool())
    registry.register(NutritionTool())
    return build_recipe_agent(registry)


def test_agent_returns_evidence_grounded_answer() -> None:
    graph = build_test_graph()
    state = create_initial_state(
        "推荐低糖晚餐",
        {"max_minutes": 30, "allergies": ["花生"]},
    )
    result = graph.invoke(state)

    assert result["status"] == "completed"
    assert result["evidence"]
    assert {event.name for event in result["trace_events"]} >= {
        "plan",
        "execute",
        "validate",
    }
    assert "工具结果" in result["final_answer"]
    assert "宫保鸡丁" not in result["final_answer"]


def test_agent_refuses_when_hard_constraints_have_no_match() -> None:
    graph = build_test_graph()
    state = create_initial_state(
        "推荐低糖晚餐",
        {"max_minutes": 1, "allergies": ["花生"]},
    )
    result = graph.invoke(state)

    assert result["status"] == "completed"
    assert "没有找到" in result["final_answer"]


def test_agent_can_plan_menu_and_shopping_list() -> None:
    graph = build_test_graph()
    state = create_initial_state(
        "规划两天晚餐菜单并生成购物清单",
        {
            "max_minutes": 30,
            "allergies": ["花生"],
            "days": 2,
            "meals_per_day": 1,
            "inventory": ["鸡胸肉"],
        },
    )
    result = graph.invoke(state)

    assert result["status"] == "completed"
    assert result["meal_plan"]
    assert result["shopping_list"]
    assert "菜单规划" in result["final_answer"]
    assert "购物清单" in result["final_answer"]


def test_agent_loads_full_steps_for_how_to_question() -> None:
    graph = build_test_graph()
    result = graph.invoke(
        create_initial_state(
            "咖喱炒蟹怎么做",
            {"allergies": ["花生"]},
        )
    )

    assert result["status"] == "completed"
    assert result["recipe_details"]
    assert "制作步骤" in result["final_answer"]
    assert "青蟹处理干净" in result["final_answer"]


def test_agent_revises_existing_menu_from_follow_up() -> None:
    graph = build_test_graph()
    result = graph.invoke(
        create_initial_state(
            "把第二天换成素菜",
            {"allergies": ["花生"], "inventory": []},
            meal_plan={
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
            shopping_list=[{"name": "花生", "reason": "当前库存中没有"}],
        )
    )

    assert result["status"] == "completed"
    assert result["meal_plan"]["days"][0]["meals"][0]["name"] == "青椒鸡胸肉"
    assert result["meal_plan"]["days"][1]["meals"][0]["name"] == "西红柿炒鸡蛋"


def test_agent_pauses_before_confirmation_tool() -> None:
    graph = build_test_graph()
    result = graph.invoke(
        create_initial_state(
            "规划两天晚餐菜单并生成购物清单",
            {"days": 2, "meals_per_day": 1},
            confirmation_required=True,
        )
    )

    assert result["status"] == "awaiting_confirmation"
    assert result["current_task"]["tool_name"] == "create_meal_plan"
    assert all(call.tool_name != "create_meal_plan" for call in result["tool_calls"])


def test_agent_answers_pairing_question_with_vegetables() -> None:
    graph = build_test_graph()
    result = graph.invoke(create_initial_state("鸡肉配什么蔬菜好？"))

    assert result["status"] == "completed"
    assert result["pairing_recommendations"]
    assert "适合搭配这些蔬菜" in result["final_answer"]


def test_agent_compares_pairing_candidates() -> None:
    graph = build_test_graph()
    result = graph.invoke(create_initial_state("鸡肉配青椒还是黄瓜？"))

    assert result["status"] == "completed"
    assert "共现证据" in result["final_answer"]
    assert "青椒" in result["final_answer"]
    assert "黄瓜" in result["final_answer"]


def test_agent_uses_structured_nutrition_data() -> None:
    graph = build_test_graph()
    result = graph.invoke(create_initial_state("青椒鸡胸肉有多少卡路里？"))

    assert result["status"] == "completed"
    assert "320 千卡" in result["final_answer"]
    assert "高蛋白" in result["final_answer"]


def test_recipe_detail_without_name_asks_for_clarification() -> None:
    graph = build_test_graph()
    result = graph.invoke(
        create_initial_state(
            "这道菜怎么做",
            query_plan={"intent": "recipe_detail", "entities": []},
        )
    )

    assert result["status"] == "completed"
    assert "菜名" in result["final_answer"]


def test_agent_applies_allergies_and_dislikes_together() -> None:
    graph = build_test_graph()
    result = graph.invoke(
        create_initial_state(
            "我对花生过敏，不要鸡蛋，推荐一道鸡肉菜",
        )
    )

    assert result["status"] == "completed"
    ingredients = [ingredient for item in result["evidence"] for ingredient in item.ingredients]
    assert "花生" not in ingredients
    assert "鸡蛋" not in ingredients


def test_agent_can_ask_before_guessing_pairing_anchor() -> None:
    graph = build_test_graph()
    result = graph.invoke(create_initial_state("配什么蔬菜好？"))

    assert result["status"] == "completed"
    assert "主食材" in result["final_answer"]


def test_agent_asks_which_recipe_after_ambiguous_follow_up() -> None:
    graph = build_test_graph()
    result = graph.invoke(
        create_initial_state(
            "怎么做",
            history=[
                {
                    "role": "assistant",
                    "content": (
                        "为你找到以下候选菜谱：\n- 青椒鸡胸肉：20 分钟\n- 西红柿炒鸡蛋：15 分钟"
                    ),
                }
            ],
        )
    )

    assert result["status"] == "completed"
    assert "哪一道" in result["final_answer"]
    assert "青椒鸡胸肉" in result["final_answer"]


def test_agent_strict_inventory_only_returns_available_recipe() -> None:
    graph = build_test_graph()
    result = graph.invoke(
        create_initial_state(
            "冰箱只有鸡蛋、西红柿、食用油和盐，推荐一道菜",
        )
    )

    assert result["status"] == "completed"
    assert [item.name for item in result["evidence"]] == ["西红柿炒鸡蛋"]


def test_respond_node_ignores_malformed_meal_plan_entries() -> None:
    state = create_initial_state(
        "规划菜单",
        meal_plan={
            "days": [
                None,
                {"day": 1, "meals": [None, {}, {"name": "青椒鸡胸肉"}]},
            ]
        },
    )
    state["status"] = "validated"
    state["evidence"] = [
        RecipeEvidence(
            recipe_id="r001",
            name="青椒鸡胸肉",
            source="memory:r001",
        )
    ]
    result = respond_node(state)

    assert result["status"] == "completed"
    assert "青椒鸡胸肉" in result["final_answer"]


def test_respond_node_explains_missing_cuisine_coverage() -> None:
    state = create_initial_state("浙菜有哪些特色菜？")
    state["status"] = "validated"
    state["evidence"] = []

    result = respond_node(state)

    assert result["status"] == "completed"
    assert "浙菜" in result["final_answer"]
    assert "没有收录" in result["final_answer"]


def test_respond_prefers_current_evidence_over_stale_meal_plan() -> None:
    """多轮追问时，不能用上一轮的菜单盖掉本轮检索结果。"""

    state = create_initial_state("有什么菜可以用西兰花")
    state["status"] = "validated"
    state["meal_plan"] = {
        "days": [{"day": 1, "meals": [{"name": "农家一碗香", "recipe_id": "r-old"}]}]
    }
    state["evidence"] = [
        RecipeEvidence(recipe_id="r-new", name="蒜蓉西兰花", source="hybrid:r-new")
    ]

    result = respond_node(state)

    assert "蒜蓉西兰花" in result["final_answer"]
    assert "农家一碗香" not in result["final_answer"]


def test_planner_uses_linked_inventory_as_search_query() -> None:
    state = create_initial_state("冰箱现有花菜，推荐一道菜")

    action = RuleBasedPlanner().plan(state)

    assert action.tool_name == "search_recipes"
    assert action.arguments["query"] == "花菜"
    assert action.arguments["required_ingredients"] == ["花菜"]
    assert action.arguments["available_ingredients"] == ["花菜"]
    assert action.arguments["sources"] == ["hybrid", "graph"]


def test_planner_relaxes_strict_inventory_for_broad_categories() -> None:
    action = RuleBasedPlanner().plan(create_initial_state("如果我只有鸡肉和蔬菜，能做什么菜"))

    assert action.tool_name == "search_recipes"
    assert action.arguments["inventory_strict"] is False
    assert action.arguments["limit"] == 5
    assert "蔬菜" in action.arguments["available_ingredients"]


def test_planner_passes_dinner_goals_and_excludes_non_meal_categories() -> None:
    action = RuleBasedPlanner().plan(create_initial_state("推荐低糖高蛋白晚餐"))

    assert action.tool_name == "search_recipes"
    assert action.arguments["dietary_goals"] == ["低糖", "高蛋白"]
    assert action.arguments["exclude_categories"] == [
        "调料",
        "饮料",
        "甜品",
        "早餐",
        "半成品",
    ]


def test_planner_passes_cuisine_filter() -> None:
    action = RuleBasedPlanner().plan(create_initial_state("粤菜有哪些特色菜？"))

    assert action.tool_name == "search_recipes"
    assert action.arguments["cuisine"] == "粤菜"


def test_explicit_recipe_ignores_ingredient_and_category_filters(monkeypatch) -> None:
    recipe_names = ("奶油蘑菇汤",)
    monkeypatch.setattr(context_module, "known_recipe_names", lambda: recipe_names)
    monkeypatch.setattr(planner_module, "known_recipe_names", lambda: recipe_names)
    state = create_initial_state(
        "我对牛奶过敏，推荐奶油蘑菇汤",
        query_plan={
            "intent": "recommendation",
            "entities": ["牛奶", "奶油蘑菇汤"],
            "constraints": {
                "allergies": ["牛奶"],
                "search_ingredients": ["黄油", "蘑菇"],
                "category": "甜品",
            },
        },
    )

    action = RuleBasedPlanner().plan(state)

    assert action.tool_name == "search_recipes"
    assert action.arguments["query"] == "奶油蘑菇汤"
    assert action.arguments["category"] is None
    assert action.arguments["required_ingredients"] == []
    assert action.arguments["exclude_ingredients"] == []


def test_strict_inventory_does_not_require_every_stock_item() -> None:
    graph = build_test_graph()
    result = graph.invoke(
        create_initial_state(
            "冰箱只有鸡蛋、西红柿、盐和食用油，推荐一道菜",
        )
    )

    assert result["status"] == "completed"
    assert [item.name for item in result["evidence"]] == ["西红柿炒鸡蛋"]


def test_unknown_food_term_does_not_return_unrelated_recipes() -> None:
    graph = build_test_graph()
    result = graph.invoke(create_initial_state("推荐一道用龙肉做的菜"))

    assert result["status"] == "completed"
    assert not result["tool_calls"]
    assert not result["evidence"]
    assert "没有找到" in result["final_answer"]


def test_llm_planner_marks_successful_and_fallback_actions() -> None:
    registry = ToolRegistry()
    registry.register(RecipeSearchTool())
    planner = LLMRecipePlanner(
        registry=registry,
        model_name="test-model",
        api_key="test-key",
        base_url="https://example.invalid",
    )

    class SuccessfulModel:
        model_name = "fake-model"

        def invoke(self, prompt):
            return type(
                "Response",
                (),
                {
                    "content": (
                        '{"action":"tool","tool_name":"search_recipes",'
                        '"arguments":{"query":"西红柿"},"intent":"检索"}'
                    )
                },
            )()

    planner.model = SuccessfulModel()
    action = planner.plan(create_initial_state("推荐西红柿炒鸡蛋"))
    assert action.planner == "llm"

    class BrokenModel(SuccessfulModel):
        def invoke(self, prompt):
            raise RuntimeError("model unavailable")

    planner.model = BrokenModel()
    fallback = planner.plan(create_initial_state("推荐西红柿炒鸡蛋"))
    assert fallback.planner == "llm_fallback"


def test_inventory_query_does_not_let_unrelated_candidates_reach_answer() -> None:
    class FakeSearcher:
        def search(self, **kwargs):
            return [
                {
                    "recipe_id": "cauliflower",
                    "name": "清炒花菜",
                    "ingredients": ["花菜", "盐", "大蒜"],
                    "category": "素菜",
                    "difficulty": 2,
                    "estimated_minutes": 15,
                },
                {
                    "recipe_id": "tomato",
                    "name": "糖拌西红柿",
                    "ingredients": ["西红柿", "白糖"],
                    "category": "素菜",
                    "difficulty": 1,
                    "estimated_minutes": 5,
                },
            ]

    registry = ToolRegistry()
    registry.register(UnifiedRecipeSearchTool(searcher=FakeSearcher()))
    graph = build_recipe_agent(registry)
    result = graph.invoke(create_initial_state("冰箱现有花菜，推荐一道菜"))

    assert result["status"] == "completed"
    assert [item.name for item in result["evidence"]] == ["清炒花菜"]
    assert "糖拌西红柿" not in result["final_answer"]


def test_explicit_allergy_conflict_is_blocked_and_explained() -> None:
    graph = build_test_graph()
    result = graph.invoke(
        create_initial_state(
            "我对花生过敏，推荐宫保鸡丁",
            query_plan={
                "intent": "recommendation",
                "entities": ["宫保鸡丁"],
                "constraints": {"allergies": ["花生"]},
            },
        )
    )

    assert result["status"] == "completed"
    assert "r003" in result["blocked_recipe_ids"]
    assert "不符合" in result["final_answer"]
    assert "memory:r003" not in result["final_answer"]
    assert "宫保鸡丁" not in [
        item.name
        for item in result["evidence"]
        if item.recipe_id not in result["blocked_recipe_ids"]
    ]
