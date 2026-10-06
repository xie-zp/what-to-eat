"""在确定性答案之上生成自然、对话式表达。"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Protocol

from langchain_openai import ChatOpenAI

from app.config import Settings
from app.domain.inventory import expand_ingredient_aliases
from app.observability import record_llm_response, trace_span
from app.state import RecipeAgentState, TraceEvent

from .lexicon import extract_known_ingredients, extract_known_recipe_names
from .nodes import respond_node

logger = logging.getLogger(__name__)


class Responder(Protocol):
    """生成最终回答的协议。"""

    def respond(self, state: RecipeAgentState) -> dict[str, Any]:
        """返回包含 final_answer 的状态更新。"""


class DeterministicResponder:
    """现有确定性回答，作为默认实现和模型失败兜底。"""

    def respond(self, state: RecipeAgentState) -> dict[str, Any]:
        started = time.perf_counter()
        result = respond_node(state)
        result["trace_events"] = [
            *result.get("trace_events", []),
            TraceEvent(
                name="respond",
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                data={"responder": "deterministic"},
            ),
        ]
        return result


class LLMResponder:
    """只改写已校验答案，不负责新增事实。"""

    def __init__(self, model: Any, fallback: Responder | None = None) -> None:
        self.model = model
        self.fallback = fallback or DeterministicResponder()

    def respond(self, state: RecipeAgentState) -> dict[str, Any]:
        started = time.perf_counter()
        baseline = self.fallback.respond(state)
        if not self._should_rewrite(state):
            logger.debug(
                "跳过自然表达：status=%s intent=%s evidence=%s blocked=%s",
                state.get("status"),
                (state.get("query_plan") or {}).get("intent"),
                len(state.get("evidence") or []),
                state.get("blocked_recipe_ids"),
            )
            return baseline

        baseline_answer = str(baseline.get("final_answer") or "")
        if not baseline_answer:
            return baseline
        try:
            with trace_span(
                "gen_ai.chat",
                {
                    "gen_ai.system": "openai-compatible",
                    "gen_ai.operation.name": "chat",
                    "gen_ai.request.model": str(getattr(self.model, "model_name", "unknown")),
                    "agent.component": "responder",
                },
                kind="client",
            ) as span:
                response = self.model.invoke(self._build_prompt(state, baseline_answer))
                record_llm_response(span, response)
            content = response.content if hasattr(response, "content") else response
            answer = str(content).strip()
            logger.debug("自然表达生成完成：%s", answer[:120])
            if not answer or _contains_unsupported_facts(answer, baseline_answer, state):
                raise ValueError("自然表达包含证据外事实")
            baseline_events = [
                event for event in baseline.get("trace_events", []) if event.name != "respond"
            ]
            return {
                **baseline,
                "final_answer": answer,
                "trace_events": [
                    *baseline_events,
                    TraceEvent(
                        name="respond",
                        duration_ms=round((time.perf_counter() - started) * 1000, 2),
                        data={"responder": "llm"},
                    ),
                ],
            }
        except Exception as exc:
            logger.warning("自然表达生成失败，使用确定性回答：%s", exc)
            return baseline

    @staticmethod
    def _should_rewrite(state: RecipeAgentState) -> bool:
        if state.get("status") not in {"completed", "validated", "ready_to_respond"}:
            return False
        if state.get("blocked_recipe_ids"):
            return False
        intent = (state.get("query_plan") or {}).get("intent")
        if intent not in {
            "recommendation",
            "meal_plan",
            "shopping",
            "pairing",
            "comparison",
            "substitution",
        }:
            return False
        return bool(
            state.get("evidence") or state.get("meal_plan") or state.get("pairing_recommendations")
        )

    @staticmethod
    def _build_prompt(state: RecipeAgentState, baseline_answer: str) -> str:
        context = {
            "goal": state.get("goal"),
            "history": (state.get("history") or [])[-6:],
            "constraints": state.get("constraints"),
            "evidence": [item.model_dump() for item in state.get("evidence", [])],
            "meal_plan": state.get("meal_plan"),
            "shopping_list": state.get("shopping_list"),
            "pairing_recommendations": state.get("pairing_recommendations"),
        }
        return f"""
你是“食序”，一个会做饭、说话自然但不啰嗦的中文食谱助手。
请把下面已经校验过的答案改写成更自然的对话回答。

硬性要求：
1. 只能使用“已校验答案”和“结构化证据”中的事实，不能新增菜谱、食材、营养或时间。
2. 不能改变菜名、食材、数量、时间、过敏或库存结论。
3. 不要提到工具、Planner、State、检索结果等内部词。
4. 先直接回应用户，再给最有用的解释；最后最多问一个自然的下一步问题。
5. 不要过度热情，不要说“作为 AI”。像熟悉厨房的人正常聊天即可。
6. 保留必要的来源信息，放在末尾一行即可。
7. 如果结构化证据里出现 graph_paths（图谱多跳路径），至少展示一条，用“→”保留完整节点顺序。

【结构化证据】
{json.dumps(context, ensure_ascii=False)}

【已校验答案】
{baseline_answer}

只输出最终回答。
""".strip()


def create_responder(settings: Settings) -> Responder:
    """根据配置创建自然表达层。"""

    if not settings.use_llm_responder or not settings.llm_api_key:
        return DeterministicResponder()
    model = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0.3,
    )
    return LLMResponder(model)


def _contains_unsupported_facts(
    answer: str,
    baseline_answer: str,
    state: RecipeAgentState,
) -> bool:
    """检查改写回答是否新增菜名、食材、时间或营养数字。"""

    allowed_recipes = (
        {item.name for item in state.get("evidence", [])}
        | {item.name for item in state.get("recipe_details", [])}
        | set(extract_known_recipe_names(baseline_answer))
    )
    meal_plan = state.get("meal_plan") or {}
    for day in meal_plan.get("days", []):
        for meal in day.get("meals", []):
            if meal.get("name"):
                allowed_recipes.add(str(meal["name"]))

    mentioned = re.findall(r"《([^》]+)》", answer)
    if any(name.strip() not in allowed_recipes for name in mentioned):
        return True
    if any(name not in allowed_recipes for name in extract_known_recipe_names(answer)):
        return True

    allowed_ingredients = set(extract_known_ingredients(baseline_answer))
    for item in state.get("evidence", []):
        allowed_ingredients.update(item.ingredients)
    for item in state.get("recipe_details", []):
        allowed_ingredients.update(item.ingredients)
    for day in meal_plan.get("days", []):
        for meal in day.get("meals", []):
            allowed_ingredients.update(meal.get("ingredients") or [])
    for item in state.get("shopping_list", []):
        if item.get("name"):
            allowed_ingredients.add(str(item["name"]))

    allowed_aliases = [
        expand_ingredient_aliases(str(item)) for item in allowed_ingredients if str(item).strip()
    ]
    for ingredient in extract_known_ingredients(answer):
        aliases = expand_ingredient_aliases(ingredient)
        if not any(aliases & allowed for allowed in allowed_aliases):
            return True

    allowed_minutes = {int(value) for value in re.findall(r"(\d+)\s*分钟", baseline_answer)}
    allowed_minutes.update(
        item.estimated_minutes
        for item in state.get("evidence", [])
        if item.estimated_minutes is not None
    )
    mentioned_minutes = {int(value) for value in re.findall(r"(\d+)\s*分钟", answer)}
    if not mentioned_minutes.issubset(allowed_minutes):
        return True

    baseline_calories = set(re.findall(r"(\d+)\s*(?:千卡|大卡|卡路里|kcal)", baseline_answer))
    answer_calories = set(re.findall(r"(\d+)\s*(?:千卡|大卡|卡路里|kcal)", answer))
    return not answer_calories.issubset(baseline_calories)
