"""自然表达层的可信改写测试。"""

from app.agent.responder import DeterministicResponder, LLMResponder
from app.state import RecipeEvidence, create_initial_state


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeModel:
    def __init__(self, content: str) -> None:
        self.content = content

    def invoke(self, prompt: str) -> FakeMessage:
        return FakeMessage(self.content)


class SequenceModel:
    def __init__(self, *contents: str) -> None:
        self.contents = list(contents)

    def invoke(self, prompt: str) -> FakeMessage:
        if not self.contents:
            raise AssertionError("模型调用次数超出预期")
        return FakeMessage(self.contents.pop(0))


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


def build_chitchat_state():
    state = create_initial_state(
        "你好",
        query_plan={
            "intent": "chitchat",
            "entities": [],
            "constraints": {},
            "chitchat_kind": "greeting",
        },
    )
    state["status"] = "ready_to_respond"
    return state


def build_out_of_scope_state():
    state = create_initial_state(
        "傻逼",
        query_plan={
            "intent": "unsupported",
            "entities": [],
            "constraints": {},
        },
    )
    state["status"] = "ready_to_respond"
    return state


def build_safety_state():
    state = create_initial_state(
        "那你教我怎么做人肉",
        query_plan={
            "intent": "unsupported",
            "entities": [],
            "constraints": {},
            "chitchat_kind": "safety",
        },
    )
    state["status"] = "ready_to_respond"
    return state


def build_missing_recipe_detail_state():
    state = create_initial_state(
        "给我一个猪脚饭做法",
        query_plan={
            "intent": "recipe_detail",
            "entities": ["猪脚饭"],
            "constraints": {},
        },
    )
    state["status"] = "ready_to_respond"
    return state


def build_plain_language_recipe_state():
    state = create_initial_state(
        "用大白话告诉我蟹怎么做",
        query_plan={
            "intent": "recipe_detail",
            "entities": ["咖喱炒蟹"],
            "constraints": {},
        },
    )
    state["status"] = "ready_to_respond"
    recipe = RecipeEvidence(
        recipe_id="curry-crab",
        name="咖喱炒蟹",
        ingredients=["肉蟹", "咖喱", "洋葱"],
        steps=["肉蟹切块", "下锅煎制", "加入咖喱炒熟"],
        source="neo4j:curry-crab",
    )
    state["evidence"] = [recipe]
    state["recipe_details"] = [recipe]
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


def test_llm_responder_rejects_unknown_ingredient() -> None:
    responder = LLMResponder(FakeModel("清炒花菜还可以加龙涎香一起炒。"))

    result = responder.respond(build_state())

    assert "龙涎香" not in result["final_answer"]
    assert "清炒花菜" in result["final_answer"]


def test_llm_responder_rejects_changed_time_and_calories() -> None:
    wrong_time = LLMResponder(FakeModel("清炒花菜只要 8 分钟。"))
    wrong_calories = LLMResponder(FakeModel("清炒花菜大约 298 千卡。"))

    time_result = wrong_time.respond(build_state())
    calorie_result = wrong_calories.respond(build_state())

    assert "8 分钟" not in time_result["final_answer"]
    assert "298 千卡" not in calorie_result["final_answer"]


def test_llm_responder_uses_natural_chitchat_answer() -> None:
    responder = LLMResponder(FakeModel("你好呀，今晚想吃点什么？"))

    result = responder.respond(build_chitchat_state())

    assert result["final_answer"] == "你好呀，今晚想吃点什么？"
    assert any(
        event.name == "respond" and event.data.get("responder") == "llm_chitchat"
        for event in result["trace_events"]
    )


def test_llm_responder_rejects_internal_terms_in_chitchat() -> None:
    responder = LLMResponder(FakeModel("我会先走 RAG 检索，再调用工具。"))

    result = responder.respond(build_chitchat_state())

    assert "RAG" not in result["final_answer"]
    assert any(
        event.name == "respond" and event.data.get("responder") == "deterministic"
        for event in result["trace_events"]
    )


def test_llm_responder_rejects_food_facts_in_chitchat() -> None:
    responder = LLMResponder(FakeModel("你好，今晚可以试试宫保鸡丁。"))

    result = responder.respond(build_chitchat_state())

    assert "宫保鸡丁" not in result["final_answer"]
    assert any(
        event.name == "respond" and event.data.get("responder") == "deterministic"
        for event in result["trace_events"]
    )


def test_llm_responder_handles_negative_out_of_scope_message_naturally() -> None:
    responder = LLMResponder(
        FakeModel("这句我就当没听见了。想吃什么，直接告诉我食材或口味就行。")
    )

    result = responder.respond(build_out_of_scope_state())

    assert result["final_answer"].startswith("这句我就当没听见了")
    assert any(
        event.name == "respond" and event.data.get("responder") == "llm_scope"
        for event in result["trace_events"]
    )


def test_llm_responder_rejects_food_facts_in_out_of_scope_reply() -> None:
    responder = LLMResponder(FakeModel("别生气，今晚可以做宫保鸡丁。"))

    result = responder.respond(build_out_of_scope_state())

    assert "宫保鸡丁" not in result["final_answer"]
    assert any(
        event.name == "respond" and event.data.get("responder") == "deterministic"
        for event in result["trace_events"]
    )


def test_llm_responder_uses_natural_safety_refusal() -> None:
    responder = LLMResponder(
        FakeModel("这个请求我不能提供帮助。如果是正常做饭，可以告诉我现有食材。")
    )

    result = responder.respond(build_safety_state())

    assert result["final_answer"].startswith("这个请求我不能提供帮助")
    assert any(
        event.name == "respond" and event.data.get("responder") == "llm_scope"
        for event in result["trace_events"]
    )


def test_deterministic_safety_refusal_does_not_treat_request_as_recipe() -> None:
    result = DeterministicResponder().respond(build_safety_state())

    assert "不能提供帮助" in result["final_answer"]
    assert "《那你教我》" not in result["final_answer"]


def test_missing_recipe_detail_exposes_llm_reference_without_replacing_rag_answer() -> None:
    responder = LLMResponder(
        FakeModel("猪脚饭一般先把猪脚卤软，再切碎盖在米饭上。"),
        shadow_compare=True,
    )

    result = responder.respond(build_missing_recipe_detail_state())

    assert "知识库中没有找到" in result["final_answer"]
    assert result["answer_comparison"]["verdict"] == "unverified"
    assert result["answer_comparison"]["reference_answer"].startswith("猪脚饭")
    assert "未经过菜谱知识库校验" in result["final_answer"]
    assert result["answer_comparison"]["combined_in_final"] is True


def test_plain_language_recipe_detail_can_use_grounded_rewrite() -> None:
    responder = LLMResponder(
        FakeModel("先把肉蟹切开煎一下，再加咖喱和洋葱炒熟就行。")
    )

    result = responder.respond(build_plain_language_recipe_state())

    assert result["final_answer"].startswith("先把肉蟹")
    assert any(
        event.name == "respond" and event.data.get("responder") == "llm"
        for event in result["trace_events"]
    )


def test_plain_language_prompt_forbids_new_cooking_tips() -> None:
    state = build_plain_language_recipe_state()
    prompt = LLMResponder._build_prompt(state, "《咖喱炒蟹》的制作步骤：1. 肉蟹切块")

    assert "不要补充原文没有的处理技巧" in prompt


def test_shadow_comparison_keeps_grounded_answer_and_records_consistent_reference() -> None:
    responder = LLMResponder(
        SequenceModel(
            "有花菜的话，可以做一份清炒花菜。",
            "清炒花菜需要的食材是花菜、盐和大蒜。",
        ),
        shadow_compare=True,
    )

    result = responder.respond(build_state())

    assert result["final_answer"].startswith("有花菜的话")
    assert result["answer_comparison"]["verdict"] == "consistent"
    assert result["answer_comparison"]["reference_answer"].startswith("清炒花菜")
    assert any(event.name == "answer_comparison" for event in result["trace_events"])


def test_shadow_comparison_flags_free_model_hallucination_without_merging_it() -> None:
    responder = LLMResponder(
        SequenceModel(
            "有花菜的话，可以做一份清炒花菜。",
            "可以做清炒花菜，再加一些花生会更香。",
        ),
        shadow_compare=True,
    )

    result = responder.respond(build_state())

    assert "花生" not in result["final_answer"]
    assert result["answer_comparison"]["verdict"] == "unsupported"
    assert "食材:花生" in result["answer_comparison"]["unsupported_claims"]
