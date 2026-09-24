"""语义查询解析器的同义表达和兜底测试。"""

from app.agent.query_analyzer import LLMQueryAnalyzer, merge_query_plans
from app.agent.query_plan import build_query_plan


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeModel:
    def __init__(self, content: str) -> None:
        self.content = content

    def invoke(self, prompt: str) -> FakeMessage:
        return FakeMessage(self.content)


def test_entity_lexicon_handles_unlisted_inventory_phrasing() -> None:
    plan = build_query_plan("花菜还能拿来做什么菜")
    inventory = build_query_plan("手头就剩点花菜")

    assert "花菜" in plan["constraints"]["search_ingredients"]
    assert inventory["constraints"]["inventory"] == ["花菜"]
    assert inventory["constraints"]["inventory_strict"] is True


def test_llm_plan_is_grounded_and_merged_with_rule_safety() -> None:
    model = FakeModel(
        """
        {
          "intent": "recommendation",
          "entities": [],
          "target_entities": [],
          "constraints": {
            "inventory": ["花椰菜", "不存在食材"],
            "search_ingredients": ["花椰菜", "不存在食材"],
            "allergies": ["花生"]
          },
          "requires_clarification": false
        }
        """
    )
    analyzer = LLMQueryAnalyzer(model)

    plan = analyzer.analyze("冰箱现有花菜，我对花生过敏")

    assert plan["constraints"]["inventory"] == ["花菜"]
    assert plan["constraints"]["search_ingredients"] == ["花菜"]
    assert plan["constraints"]["allergies"] == ["花生"]
    assert "不存在食材" not in plan["constraints"]["search_ingredients"]


def test_merge_query_plans_keeps_rule_inventory_strictness() -> None:
    merged = merge_query_plans(
        build_query_plan("冰箱只有花菜"),
        {
            "intent": "recommendation",
            "constraints": {"inventory_strict": False},
        },
        goal="冰箱只有花菜",
    )

    assert merged["constraints"]["inventory_strict"] is True


def test_merge_query_plans_drops_llm_meal_plan_entities_and_category() -> None:
    merged = merge_query_plans(
        build_query_plan("规划两天晚餐菜单并生成购物清单"),
        {
            "intent": "meal_plan",
            "entities": ["晚餐菜单", "购物清单"],
            "target_entities": ["晚餐菜单", "购物清单"],
            "constraints": {"category": "晚餐"},
        },
        goal="规划两天晚餐菜单并生成购物清单",
    )

    assert merged["entities"] == []
    assert merged["target_entities"] == []
    assert merged["constraints"]["category"] is None


def test_rule_comparison_intent_wins_over_llm_pairing() -> None:
    merged = merge_query_plans(
        build_query_plan("鸡肉配西红柿还是土豆？"),
        {"intent": "pairing"},
        goal="鸡肉配西红柿还是土豆？",
    )

    assert merged["intent"] == "comparison"
