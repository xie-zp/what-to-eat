"""自然表达层的可信改写测试。"""

from app.agent.responder import LLMResponder
from app.state import RecipeEvidence, create_initial_state


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeModel:
    def __init__(self, content: str) -> None:
        self.content = content

    def invoke(self, prompt: str) -> FakeMessage:
        return FakeMessage(self.content)


def build_state():
    state = create_initial_state(
        "冰箱现有花菜，推荐一道菜",
        query_plan={
            "intent": "recommendation",
            "entities": ["花菜"],
            "constraints": {"inventory": ["花菜"], "search_ingredients": ["花菜"]},
        },
    )
    state["status"] = "ready_to_respond"
    state["evidence"] = [
        RecipeEvidence(
            recipe_id="cauliflower",
            name="清炒花菜",
            ingredients=["花菜", "盐", "大蒜"],
            source="neo4j:cauliflower",
        )
    ]
    return state


def test_llm_responder_rewrites_without_new_recipe() -> None:
    responder = LLMResponder(
        FakeModel("有花菜的话，可以先做一份清炒花菜，简单也稳妥。要不要我继续按两人份整理？")
    )

    result = responder.respond(build_state())

    assert "清炒花菜" in result["final_answer"]
    assert "两人份" in result["final_answer"]
    assert any(
        event.name == "respond" and event.data.get("responder") == "llm"
        for event in result["trace_events"]
    )


def test_llm_responder_rejects_unknown_recipe_and_falls_back() -> None:
    responder = LLMResponder(FakeModel("也可以做《佛跳墙》。"))

    result = responder.respond(build_state())

    assert "佛跳墙" not in result["final_answer"]
    assert "清炒花菜" in result["final_answer"]


def test_llm_responder_rejects_new_ingredient() -> None:
    responder = LLMResponder(FakeModel("清炒花菜还可以加花生一起炒。"))

    result = responder.respond(build_state())

    assert "花生" not in result["final_answer"]
    assert "清炒花菜" in result["final_answer"]


def test_llm_responder_rejects_changed_time_and_calories() -> None:
    wrong_time = LLMResponder(FakeModel("清炒花菜只要 8 分钟。"))
    wrong_calories = LLMResponder(FakeModel("清炒花菜大约 298 千卡。"))

    time_result = wrong_time.respond(build_state())
    calorie_result = wrong_calories.respond(build_state())

    assert "8 分钟" not in time_result["final_answer"]
    assert "298 千卡" not in calorie_result["final_answer"]
