"""查询语义解析器：LLM 负责理解问法，规则负责安全兜底。"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from langchain_openai import ChatOpenAI

from app.config import Settings
from app.domain.inventory import INGREDIENT_ALIASES
from app.observability import record_llm_response, trace_span

from .lexicon import extract_known_ingredients, known_recipe_names
from .query_plan import QueryPlan, build_query_plan, parse_query_plan

logger = logging.getLogger(__name__)

LIST_CONSTRAINT_KEYS = (
    "allergies",
    "avoid_ingredients",
    "inventory",
    "search_ingredients",
    "dietary_goals",
)
SCALAR_CONSTRAINT_KEYS = (
    "max_minutes",
    "max_difficulty",
    "category",
    "days",
    "meals_per_day",
    "household_size",
    "inventory_strict",
)
KNOWN_RECIPE_CATEGORIES = {
    "素菜",
    "荤菜",
    "汤类",
    "早餐",
    "主食",
    "甜品",
    "水产",
    "半成品",
    "西餐",
}


class QueryAnalyzer(Protocol):
    """把用户语言转换成强类型查询计划的协议。"""

    def analyze(
        self,
        goal: str,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """返回可交给 Agent state 的计划。"""


class RuleBasedQueryAnalyzer:
    """不依赖模型的确定性兜底解析器。"""

    def analyze(
        self,
        goal: str,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return build_query_plan(goal, history=history)


class LLMQueryAnalyzer:
    """使用模型理解自然语言，只保留可被原文证据支持的结构化字段。"""

    def __init__(self, model: Any, fallback: QueryAnalyzer | None = None) -> None:
        self.model = model
        self.fallback = fallback or RuleBasedQueryAnalyzer()

    def analyze(
        self,
        goal: str,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        rule_plan = self.fallback.analyze(goal, history=history)
        try:
            with trace_span(
                "gen_ai.chat",
                {
                    "gen_ai.system": "openai-compatible",
                    "gen_ai.operation.name": "chat",
                    "gen_ai.request.model": str(getattr(self.model, "model_name", "unknown")),
                    "agent.component": "query_analyzer",
                },
                kind="client",
            ) as span:
                response = self.model.invoke(self._build_prompt(goal, history or []))
                record_llm_response(span, response)
            content = response.content if hasattr(response, "content") else response
            if not isinstance(content, str):
                content = str(content)
            llm_plan = QueryPlan.model_validate_json(_extract_json_object(content))
            return merge_query_plans(
                rule_plan,
                llm_plan.model_dump(),
                goal=goal,
                history=history or [],
            )
        except Exception as exc:
            logger.warning("LLM 查询解析失败，使用规则解析：%s", exc)
            return rule_plan

    @staticmethod
    def _build_prompt(goal: str, history: list[dict[str, Any]]) -> str:
        schema = QueryPlan.model_json_schema()
        return f"""
你是食谱 Agent 的查询解析器。
请把用户表达转换成结构化查询计划，不要推荐菜谱，不要补充用户没有说过的食材。

【对话历史】
{json.dumps(history[-6:], ensure_ascii=False)}

【用户当前表达】
{goal}

【输出 JSON Schema】
{json.dumps(schema, ensure_ascii=False)}

要求：
1. 理解同义表达，例如“现有/手头有/家里还有/冰箱里剩”都表示用户已有某种食材。
2. 用户提到某种已知食材但问法不同，也要填入 constraints.search_ingredients。
3. 只有“只有/仅有/只剩/没别的/就这些”才把 inventory_strict 设为 true。
4. 所有字符串只能来自用户表达或对话历史，不能凭空创造食材。
5. 无法确定时留空，不要猜测。
6. 只返回一个合法 JSON 对象，不要 Markdown，不要解释。
""".strip()


def create_query_analyzer(settings: Settings) -> QueryAnalyzer:
    """根据配置创建语义解析器，没有密钥时自动使用规则解析。"""

    if not settings.use_llm_query_analyzer or not settings.llm_api_key:
        return RuleBasedQueryAnalyzer()
    model = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0,
    )
    return LLMQueryAnalyzer(model)


def merge_query_plans(
    rule_plan: dict[str, Any],
    llm_plan: dict[str, Any],
    goal: str,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """合并 LLM 理解和规则兜底，确定性字段优先保留。"""

    rule = parse_query_plan(rule_plan)
    llm = parse_query_plan(llm_plan)
    source_text = "\n".join([goal, *(_history_contents(history or []))])

    intent = llm.intent if rule.intent == "recommendation" else rule.intent
    if rule.intent in {"meal_plan", "shopping"}:
        intent = rule.intent
    if rule.constraints.unknown_food_terms:
        intent = rule.intent

    if rule.intent in {"meal_plan", "shopping"}:
        entities = rule.entities
        target_entities = rule.target_entities
    else:
        entities = _merge_known_entities(rule.entities, llm.entities, source_text)
        target_entities = _merge_known_entities(
            rule.target_entities,
            llm.target_entities,
            source_text,
        )
    constraints = rule.constraints.model_dump(exclude_none=True)
    llm_constraints = llm.constraints.model_dump(exclude_none=True)

    for key in LIST_CONSTRAINT_KEYS:
        rule_values = rule.constraints.model_dump().get(key) or []
        llm_values = llm_constraints.get(key) or []
        grounded = []
        for value in rule_values:
            text_value = str(value)
            if not text_value:
                continue
            if any(_same_ingredient(text_value, existing) for existing in grounded):
                continue
            grounded.append(text_value)
        for value in llm_values:
            text_value = str(value)
            if not text_value or not _is_grounded(text_value, source_text):
                continue
            if key in {"inventory", "search_ingredients"} and not _is_ingredient_entity(
                text_value,
                source_text,
            ):
                continue
            if key in {"allergies", "avoid_ingredients"} and not _looks_like_food_term(text_value):
                continue
            if any(_same_ingredient(text_value, existing) for existing in grounded):
                continue
            grounded.append(text_value)
        constraints[key] = grounded

    for key in SCALAR_CONSTRAINT_KEYS:
        if key == "inventory_strict":
            constraints[key] = bool(constraints.get(key) or llm_constraints.get(key))
        elif key == "category":
            category = str(llm_constraints.get(key) or "").strip()
            if category and category in KNOWN_RECIPE_CATEGORIES:
                constraints[key] = category
        elif llm_constraints.get(key) is not None:
            constraints[key] = llm_constraints[key]

    plan = QueryPlan.model_validate(
        {
            **rule.model_dump(),
            **llm.model_dump(),
            "intent": intent,
            "entities": entities,
            "target_entities": target_entities,
            "constraints": constraints,
            "context_candidates": rule.context_candidates,
            "requires_clarification": rule.requires_clarification,
        }
    )
    return plan.model_dump()


def _merge_known_entities(
    rule_values: list[str],
    llm_values: list[str],
    source_text: str,
) -> list[str]:
    values = [*rule_values, *llm_values]
    merged = []
    for value in values:
        if not value or (value not in rule_values and not _is_known_entity(value, source_text)):
            continue
        if any(_same_ingredient(value, existing) for existing in merged):
            continue
        merged.append(value)
    return merged


def _is_grounded(value: str, source_text: str) -> bool:
    if value in source_text:
        return True
    for canonical, aliases in INGREDIENT_ALIASES.items():
        if value in aliases or value == canonical:
            if any(alias in source_text for alias in aliases):
                return True
    known = extract_known_ingredients(source_text)
    return any(value in item or item in value for item in known)


def _is_known_entity(value: str, source_text: str) -> bool:
    if value in known_recipe_names():
        return value in source_text
    return _is_ingredient_entity(value, source_text)


def _is_ingredient_entity(value: str, source_text: str) -> bool:
    if any(_same_ingredient(value, item) for item in extract_known_ingredients(source_text)):
        return True
    return value in source_text and any(
        value == canonical or value in aliases for canonical, aliases in INGREDIENT_ALIASES.items()
    )


def _looks_like_food_term(value: str) -> bool:
    if not value or len(value) > 12:
        return False
    return not any(marker in value for marker in ("规则", "忽略", "菜单", "购物", "编一个"))


def _same_ingredient(left: str, right: str) -> bool:
    if left == right:
        return True
    for aliases in INGREDIENT_ALIASES.values():
        if left in aliases and right in aliases:
            return True
    return False


def _history_contents(history: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("content") or "") for item in history if isinstance(item, dict)]


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("查询解析器未返回 JSON 对象")
    return text[start : end + 1]
