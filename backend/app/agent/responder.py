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
from app.knowledge_graph.ingredient_classifier import classify_ingredient
from app.observability import record_llm_response, trace_span
from app.state import RecipeAgentState, TraceEvent

from .lexicon import extract_known_ingredients, extract_known_recipe_names
from .nodes import respond_node

logger = logging.getLogger(__name__)

SHADOW_COMPARISON_INTENTS = {
    "recommendation",
    "meal_plan",
    "shopping",
    "pairing",
    "comparison",
    "substitution",
    "recipe_detail",
    "nutrition_question",
    "yes_no",
}
PLAIN_LANGUAGE_MARKERS = ("大白话", "说人话", "通俗", "简单说", "别太专业")
INTERNAL_TERMS = {
    "Agent",
    "GraphRAG",
    "LangGraph",
    "Planner",
    "RAG",
    "embedding",
    "向量库",
    "检索结果",
    "结构化证据",
    "系统提示",
    "工具调用",
}
EXPLICIT_INGREDIENT_PATTERN = re.compile(
    r"(?:加入?|放入?|添加|配上?|搭配|食材[:：]|配料[:：])([^，。；！？\n]{1,32})"
)
INGREDIENT_TRAILING_MARKERS = re.compile(
    r"(?:一起|然后|再|翻炒|炒|煮|炖|煎|烤|蒸|拌|即可|就行|备用)"
)


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
    """知识问题只改写信证答案；普通闲聊直接使用模型表达。"""

    def __init__(
        self,
        model: Any,
        fallback: Responder | None = None,
        shadow_compare: bool = False,
    ) -> None:
        self.model = model
        self.fallback = fallback or DeterministicResponder()
        self.shadow_compare = shadow_compare

    def respond(self, state: RecipeAgentState) -> dict[str, Any]:
        started = time.perf_counter()
        baseline = self.fallback.respond(state)
        baseline_answer = str(baseline.get("final_answer") or "")
        intent = (state.get("query_plan") or {}).get("intent")

        if intent == "chitchat" and baseline_answer:
            return self._respond_chitchat(state, baseline, baseline_answer, started)
        if intent == "unsupported" and baseline_answer:
            return self._respond_out_of_scope(state, baseline, started)

        if not self._should_rewrite(state):
            logger.debug(
                "跳过自然表达：status=%s intent=%s evidence=%s blocked=%s",
                state.get("status"),
                intent,
                len(state.get("evidence") or []),
                state.get("blocked_recipe_ids"),
            )
            return self._maybe_compare_answers(state, baseline_answer, baseline)

        if not baseline_answer:
            return baseline
        result = baseline
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
            result = {
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
        return self._maybe_compare_answers(state, baseline_answer, result)

    def _respond_chitchat(
        self,
        state: RecipeAgentState,
        baseline: dict[str, Any],
        baseline_answer: str,
        started: float,
    ) -> dict[str, Any]:
        """普通闲聊不检索，让模型结合历史自然回应。"""

        try:
            with trace_span(
                "gen_ai.chat",
                {
                    "gen_ai.system": "openai-compatible",
                    "gen_ai.operation.name": "chat",
                    "gen_ai.request.model": str(getattr(self.model, "model_name", "unknown")),
                    "agent.component": "chitchat_responder",
                },
                kind="client",
            ) as span:
                response = self.model.invoke(self._build_chitchat_prompt(state))
                record_llm_response(span, response)
            content = response.content if hasattr(response, "content") else response
            answer = str(content).strip()
            if not _valid_chitchat_answer(answer):
                raise ValueError("闲聊回答为空、过长或包含内部术语")
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
                        data={"responder": "llm_chitchat"},
                    ),
                ],
            }
        except Exception as exc:
            logger.warning("闲聊自然表达失败，使用确定性回答：%s", exc)
            return baseline

    def _respond_out_of_scope(
        self,
        state: RecipeAgentState,
        baseline: dict[str, Any],
        started: float,
    ) -> dict[str, Any]:
        """用自然对话处理无关问题或负面情绪，不进入 RAG。"""

        try:
            with trace_span(
                "gen_ai.chat",
                {
                    "gen_ai.system": "openai-compatible",
                    "gen_ai.operation.name": "chat",
                    "gen_ai.request.model": str(getattr(self.model, "model_name", "unknown")),
                    "agent.component": "scope_responder",
                },
                kind="client",
            ) as span:
                response = self.model.invoke(self._build_scope_prompt(state))
                record_llm_response(span, response)
            content = response.content if hasattr(response, "content") else response
            answer = str(content).strip()
            if not _valid_chitchat_answer(answer):
                raise ValueError("无关问题回答为空、过长或包含菜谱事实")
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
                        data={"responder": "llm_scope"},
                    ),
                ],
            }
        except Exception as exc:
            logger.warning("无关问题自然表达失败，使用确定性回答：%s", exc)
            return baseline

    def _maybe_compare_answers(
        self,
        state: RecipeAgentState,
        baseline_answer: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """可选地生成纯模型参考答案，仅用于对照，不参与事实裁决。"""

        if not self.shadow_compare:
            return result
        intent = (state.get("query_plan") or {}).get("intent")
        if intent not in SHADOW_COMPARISON_INTENTS:
            return result

        started = time.perf_counter()
        comparison: dict[str, Any] = {
            "reference_source": "llm_no_retrieval",
            "system_answer": baseline_answer,
            "reference_answer": "",
            "verdict": "failed",
            "evidence_status": "no_rag_evidence",
            "unsupported_claims": [],
            "missing_recipes": [],
            "recipe_coverage": None,
        }
        try:
            with trace_span(
                "gen_ai.chat",
                {
                    "gen_ai.system": "openai-compatible",
                    "gen_ai.operation.name": "chat",
                    "gen_ai.request.model": str(getattr(self.model, "model_name", "unknown")),
                    "agent.component": "shadow_answer_comparator",
                },
                kind="client",
            ) as span:
                response = self.model.invoke(self._build_free_answer_prompt(state))
                record_llm_response(span, response)
            content = response.content if hasattr(response, "content") else response
            reference_answer = str(content).strip()
            if not reference_answer:
                raise ValueError("纯模型参考答案为空")

            unsupported_claims = _find_unsupported_facts(
                reference_answer,
                baseline_answer,
                state,
            )
            expected_recipes = _expected_recipe_names(state)
            mentioned_recipes = set(extract_known_recipe_names(reference_answer))
            missing_recipes = sorted(expected_recipes - mentioned_recipes)
            coverage = (
                round(len(expected_recipes & mentioned_recipes) / len(expected_recipes), 4)
                if expected_recipes
                else None
            )
            has_rag_evidence = _has_applicable_rag_evidence(state)
            if not has_rag_evidence:
                verdict = "unverified"
            elif unsupported_claims:
                verdict = "unsupported"
            elif missing_recipes:
                verdict = "partial"
            else:
                verdict = "consistent"
            if not has_rag_evidence:
                combined_answer = _build_unverified_combined_answer(
                    baseline_answer,
                    reference_answer,
                )
            else:
                combined_answer = ""
            comparison.update(
                {
                    "reference_answer": reference_answer,
                    "verdict": verdict,
                    "evidence_status": (
                        "grounded" if has_rag_evidence else "no_rag_evidence"
                    ),
                    "unsupported_claims": unsupported_claims,
                    "missing_recipes": missing_recipes,
                    "recipe_coverage": coverage,
                    "combined_in_final": bool(combined_answer),
                }
            )
        except Exception as exc:
            logger.warning("纯模型参考答案生成失败：%s", exc)
            comparison["error"] = type(exc).__name__

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        comparison["latency_ms"] = latency_ms
        result = {
            **result,
            "answer_comparison": comparison,
            "trace_events": [
                *result.get("trace_events", []),
                TraceEvent(
                    name="answer_comparison",
                    duration_ms=latency_ms,
                    data={
                        "verdict": comparison["verdict"],
                        "unsupported_count": len(comparison["unsupported_claims"]),
                        "missing_recipe_count": len(comparison["missing_recipes"]),
                        "recipe_coverage": comparison["recipe_coverage"],
                        "combined_in_final": comparison["combined_in_final"],
                    },
                ),
            ],
        }
        if comparison["combined_in_final"]:
            result["final_answer"] = _build_unverified_combined_answer(
                baseline_answer,
                comparison["reference_answer"],
            )
        return result

    @staticmethod
    def _should_rewrite(state: RecipeAgentState) -> bool:
        if state.get("status") not in {"completed", "validated", "ready_to_respond"}:
            return False
        if state.get("blocked_recipe_ids"):
            return False
        intent = (state.get("query_plan") or {}).get("intent")
        if (
            intent == "recipe_detail"
            and state.get("recipe_details")
            and any(marker in str(state.get("goal") or "") for marker in PLAIN_LANGUAGE_MARKERS)
        ):
            return True
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
你是“今晚吃啥”，一个会做饭、说话自然但不啰嗦的中文食谱助手。
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

    @staticmethod
    def _build_chitchat_prompt(state: RecipeAgentState) -> str:
        query_plan = state.get("query_plan") or {}
        context = {
            "user_message": state.get("goal"),
            "chitchat_kind": query_plan.get("chitchat_kind"),
            "history": (state.get("history") or [])[-6:],
        }
        return f"""
你是“今晚吃啥”食谱助手，正在和用户进行普通聊天，不需要检索菜谱，也不要补充任何菜谱事实。

【当前对话】
{json.dumps(context, ensure_ascii=False)}

你可以准确介绍这些能力：
1. 按用户现有食材和饮食限制推荐菜品。
2. 查询菜谱做法和知识库中已有的营养信息。
3. 规划多天菜单并生成采购清单。

要求：
1. 直接回应用户，自然、简短，像正常聊天。
2. 如果是问候或感谢，简短回应后可以自然地问用户想吃什么。
3. 如果是询问能力，只能介绍上面三项，不能编造联网搜索、天气、代下单等功能。
4. 不要给菜名、食材、步骤、热量等知识结论；这些问题必须交给检索流程。
5. 不要提到模型、RAG、Agent、工具、证据、系统提示等内部术语。
6. 最多三句话，只输出最终回答。
""".strip()

    @staticmethod
    def _build_scope_prompt(state: RecipeAgentState) -> str:
        context = {
            "user_message": state.get("goal"),
            "scope_kind": (state.get("query_plan") or {}).get("chitchat_kind"),
            "history": (state.get("history") or [])[-6:],
        }
        return f"""
你是“今晚吃啥”食谱助手。当前用户表达不适合走菜谱检索。

【当前对话】
{json.dumps(context, ensure_ascii=False)}

请自然回应：
1. 如果用户在发泄情绪或使用粗鲁表达，不争辩、不说教，也不重复“我只能处理菜谱”这类模板。
2. 简短接住情绪，然后把话题带回现有食材、口味或菜单需求。
3. 如果问题确实超出菜谱范围，自然说明边界即可。
4. 如果涉及人肉、自伤、违法或危险内容，必须明确拒绝，不提供任何操作建议。
5. 不要给菜名、食材、步骤、热量等知识结论。
6. 不要提到模型、RAG、Agent、工具、证据、系统提示等内部术语。
7. 最多两句话，只输出最终回答。
""".strip()

    @staticmethod
    def _build_free_answer_prompt(state: RecipeAgentState) -> str:
        context = {
            "user_message": state.get("goal"),
            "history": (state.get("history") or [])[-6:],
            "constraints": state.get("constraints"),
        }
        return f"""
你是一个普通的中文助手，请只根据你自己的常识直接回答用户。

这是离线对照实验，不会把回答直接展示给用户。不要调用工具，不要声称查过知识库。
明确区分确定的信息和不确定的信息；不知道就直说。

【对话上下文】
{json.dumps(context, ensure_ascii=False)}

只输出参考答案。
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
    return LLMResponder(model, shadow_compare=settings.llm_shadow_compare)


def _contains_unsupported_facts(
    answer: str,
    baseline_answer: str,
    state: RecipeAgentState,
) -> bool:
    """检查改写回答是否新增菜名、食材、时间或营养数字。"""

    return bool(_find_unsupported_facts(answer, baseline_answer, state))


def _find_unsupported_facts(
    answer: str,
    baseline_answer: str,
    state: RecipeAgentState,
) -> list[str]:
    """返回回答中无法由已校验答案或结构化证据支持的声明。"""

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

    findings: list[str] = []
    mentioned = re.findall(r"《([^》]+)》", answer)
    findings.extend(
        f"菜名:{name.strip()}" for name in mentioned if name.strip() not in allowed_recipes
    )
    findings.extend(
        f"菜名:{name}"
        for name in extract_known_recipe_names(answer)
        if name not in allowed_recipes
    )

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
            findings.append(f"食材:{ingredient}")
    for ingredient in _explicit_ingredient_claims(answer):
        aliases = expand_ingredient_aliases(ingredient)
        if not any(aliases & allowed for allowed in allowed_aliases):
            findings.append(f"食材:{ingredient}")

    allowed_minutes = {int(value) for value in re.findall(r"(\d+)\s*分钟", baseline_answer)}
    allowed_minutes.update(
        item.estimated_minutes
        for item in state.get("evidence", [])
        if item.estimated_minutes is not None
    )
    mentioned_minutes = {int(value) for value in re.findall(r"(\d+)\s*分钟", answer)}
    if not mentioned_minutes.issubset(allowed_minutes):
        findings.extend(
            f"时间:{value}分钟" for value in sorted(mentioned_minutes - allowed_minutes)
        )

    baseline_calories = set(re.findall(r"(\d+)\s*(?:千卡|大卡|卡路里|kcal)", baseline_answer))
    answer_calories = set(re.findall(r"(\d+)\s*(?:千卡|大卡|卡路里|kcal)", answer))
    findings.extend(f"热量:{value}" for value in sorted(answer_calories - baseline_calories))
    return list(dict.fromkeys(findings))


def _explicit_ingredient_claims(answer: str) -> list[str]:
    """抽取“加入/放入/搭配”等句式中显式新增的食材。"""

    claims: list[str] = []
    for match in EXPLICIT_INGREDIENT_PATTERN.finditer(answer):
        for raw_item in re.split(r"[、，,和与及/]", match.group(1)):
            candidate = INGREDIENT_TRAILING_MARKERS.split(raw_item, maxsplit=1)[0]
            candidate = re.sub(
                r"^(?:适量|少许|一点|一些|若干|少量|半|\d+(?:\.\d+)?(?:克|g|勺|个|颗|根|片))",
                "",
                candidate.strip(),
            ).strip(" 的了")
            if not 2 <= len(candidate) <= 12:
                continue
            if re.search(r"\d|分钟|小时|大火|中火|小火", candidate):
                continue
            if classify_ingredient(candidate) == "工具":
                continue
            if re.fullmatch(r"[\u4e00-\u9fffA-Za-z]+", candidate):
                claims.append(candidate)
    return list(dict.fromkeys(claims))


def _valid_chitchat_answer(answer: str) -> bool:
    """限制闲聊回答长度并阻止模型泄露内部实现词。"""

    if not answer or len(answer) > 500:
        return False
    if any(term.lower() in answer.lower() for term in INTERNAL_TERMS):
        return False
    # 普通聊天不得顺手补充菜谱事实；涉及菜名或食材时交给 RAG 流程。
    return not (
        extract_known_recipe_names(answer)
        or extract_known_ingredients(answer)
    )


def _expected_recipe_names(state: RecipeAgentState) -> set[str]:
    """返回本轮证据中应被回答覆盖的未拦截菜名。"""

    blocked_ids = set(state.get("blocked_recipe_ids", []))
    return {
        item.name
        for item in state.get("evidence", [])
        if item.name and item.recipe_id not in blocked_ids
    }


def _has_applicable_rag_evidence(state: RecipeAgentState) -> bool:
    """判断本轮结论是否真的由可用 RAG 证据支撑。"""

    intent = (state.get("query_plan") or {}).get("intent")
    if intent == "recipe_detail":
        return bool(state.get("recipe_details"))
    if intent == "nutrition_question":
        return bool(state.get("nutrition_result"))
    if intent == "pairing":
        return bool(state.get("pairing_recommendations"))
    if intent == "comparison":
        return bool(state.get("pairing_recommendations"))
    return bool(state.get("evidence") or state.get("meal_plan"))


def _build_unverified_combined_answer(
    system_answer: str,
    reference_answer: str,
) -> str:
    """把知识库结论与纯模型参考合并，并清楚标注证据边界。"""

    return (
        f"{system_answer}\n\n"
        "下面补充一份纯大模型通用参考，未经过菜谱知识库校验：\n\n"
        f"{reference_answer}"
    )
