"""Agent Planner 实现。"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from langchain_openai import ChatOpenAI

from app.observability import record_llm_response, trace_span
from app.state import PlannedAction, RecipeAgentState
from app.tools.registry import ToolRegistry

from .context import (
    available_ingredients,
    excluded_ingredients,
    get_constraints,
    get_primary_entity,
    get_query_plan,
    get_target_entities,
    inventory_is_strict,
    requested_recipe_name,
)
from .lexicon import known_recipe_names

logger = logging.getLogger(__name__)


def normalize_search_query(text: str) -> str:
    """移除问题中的操作指令词，保留可检索的菜品或食材名称。"""

    query = text.strip()
    for marker in ("怎么做", "如何做", "怎么制作", "的做法", "做法", "步骤"):
        query = query.replace(marker, "")
    return query.strip()


def is_meal_revision(goal: str) -> bool:
    """判断用户是否在修改已有菜单。"""

    return any(
        marker in goal for marker in ("换成", "改成", "替换", "调整菜单", "修改菜单", "重新安排")
    )


def is_shopping_revision(goal: str) -> bool:
    """判断用户是否在修改已有购物清单。"""

    return "购物" in goal and any(
        marker in goal for marker in ("去掉", "删除", "移除", "不要", "不买")
    )


def revision_search_query(goal: str) -> str:
    """从修改指令中提取适合检索的关键词。"""

    if any(word in goal for word in ("素菜", "素食", "纯素")):
        return ""
    for tag in ("低糖", "高蛋白", "低脂", "高纤维"):
        if tag in goal:
            return tag
    query = goal
    for marker in (
        "Shopping",
        "购物清单",
        "菜单",
        "第",
        "天",
        "把",
        "换成",
        "改成",
        "替换为",
        "替换",
        "重新安排",
        "调整",
        "修改",
    ):
        query = query.replace(marker, "")
    return query.strip() or "家常菜"


def revision_category(goal: str) -> str | None:
    """返回修改指令中的明确分类。"""

    if any(word in goal for word in ("素菜", "素食", "纯素")):
        return "素菜"
    return None


def _search_sources(
    search_ingredients: list[str],
    available: list[str],
    constraints: dict[str, Any],
) -> list[str]:
    """有明确实体或硬约束时同时走图和混合检索；开放推荐优先走向量混合。"""

    if (
        search_ingredients
        or available
        or constraints.get("category")
        or constraints.get("cuisine")
        or constraints.get("max_minutes") is not None
    ):
        return ["hybrid", "graph"]
    return ["hybrid"]


def is_pairing_question(goal: str) -> bool:
    """判断用户是否在询问主食材与蔬菜的搭配。"""

    return any(marker in goal for marker in ("配什么", "配哪些", "搭配什么", "和什么")) and any(
        target in goal for target in ("蔬菜", "配菜")
    )


def pairing_anchor(goal: str) -> str | None:
    """从搭配问题中提取主食材。"""

    for marker in ("配什么", "配哪些", "搭配什么", "和什么", "配"):
        if marker in goal:
            anchor = goal.split(marker, maxsplit=1)[0]
            return anchor.replace("推荐", "").replace("请问", "").strip(" ，,。？?") or None
    return None


class Planner(Protocol):
    """Planner 协议。"""

    def plan(self, state: RecipeAgentState) -> PlannedAction:
        """根据当前状态返回下一步动作。"""


class RuleBasedPlanner:
    """不依赖 LLM 的确定性规划器。"""

    def plan(self, state: RecipeAgentState) -> PlannedAction:
        """按规则推进检索、菜单规划和购物清单。"""

        goal = state["goal"]
        query_plan = get_query_plan(state)
        intent = query_plan.intent
        if not intent and is_pairing_question(goal):
            intent = "pairing"
        evidence = state.get("evidence", [])
        meal_plan = state.get("meal_plan")
        shopping_list = state.get("shopping_list", [])
        completed_tools = {call.tool_name for call in state.get("tool_calls", [])}
        constraints = get_constraints(state)
        excluded = excluded_ingredients(state)
        available = available_ingredients(state)
        search_ingredients = constraints.get("search_ingredients") or []
        strict_inventory = inventory_is_strict(state)

        if intent == "unsupported":
            return PlannedAction(
                action="final",
                intent="问题不属于食谱规划范围",
            )

        if query_plan.requires_clarification:
            return PlannedAction(
                action="final",
                intent="上一轮有多个可选菜谱，需要用户明确指定",
            )

        if intent in {"pairing", "comparison"}:
            if state.get("pairing_recommendations"):
                return PlannedAction(
                    action="final",
                    intent="已经完成食材搭配统计",
                )
            anchor = get_primary_entity(state) or pairing_anchor(goal)
            if not anchor:
                return PlannedAction(
                    action="final",
                    intent="搭配问题缺少主食材，需要用户补充",
                )
            return PlannedAction(
                action="tool",
                tool_name="recommend_pairings",
                arguments={
                    "anchor": anchor,
                    "target_type": "蔬菜",
                    "candidates": get_target_entities(state) if intent == "comparison" else [],
                    "limit": 8,
                },
                intent="从菜谱图谱中统计主食材与蔬菜的共现关系",
            )

        if intent == "nutrition_question":
            requested_name = requested_recipe_name(state)
            if not evidence:
                if not requested_name:
                    return PlannedAction(
                        action="final",
                        intent="营养问题缺少明确菜名",
                    )
                return PlannedAction(
                    action="tool",
                    tool_name="search_recipes",
                    arguments={
                        "query": requested_name,
                        "exclude_ingredients": excluded,
                        "sources": ["hybrid", "graph"],
                        "limit": 5,
                    },
                    intent="定位营养问题对应的菜谱",
                )
            matches = [
                item
                for item in evidence
                if not requested_name
                or item.name == requested_name
                or requested_name in item.name
                or item.name in requested_name
            ]
            if not matches:
                return PlannedAction(
                    action="final",
                    intent="知识库中没有找到对应菜谱",
                )
            if "get_nutrition" in completed_tools:
                return PlannedAction(
                    action="final",
                    intent="已经读取结构化营养数据",
                )
            return PlannedAction(
                action="tool",
                tool_name="get_nutrition",
                arguments={"recipe_id": matches[0].recipe_id},
                intent="读取菜谱的结构化热量和营养标签",
            )

        if intent == "yes_no":
            return PlannedAction(
                action="final",
                intent="当前图谱缺少足够的结构化信息回答该判断问题",
            )

        if intent == "substitution" and not meal_plan:
            return PlannedAction(
                action="final",
                intent="当前任务没有可以执行替换的已有菜单",
            )

        if is_shopping_revision(goal) and shopping_list:
            if "revise_shopping_list" in completed_tools:
                return PlannedAction(
                    action="final",
                    intent="购物清单已经按追问完成修改",
                )
            return PlannedAction(
                action="tool",
                tool_name="revise_shopping_list",
                arguments={
                    "instruction": goal,
                    "shopping_list": shopping_list,
                },
                intent="按照追问删除购物清单中的指定食材",
            )

        if is_meal_revision(goal) and meal_plan:
            if "revise_meal_plan" in completed_tools:
                return PlannedAction(
                    action="final",
                    intent="菜单已经按追问完成修改",
                )
            if not evidence:
                return PlannedAction(
                    action="tool",
                    tool_name="search_recipes",
                    arguments={
                        "query": revision_search_query(goal),
                        "max_minutes": constraints.get("max_minutes"),
                        "exclude_ingredients": excluded,
                        "required_ingredients": search_ingredients,
                        "category": revision_category(goal),
                        "available_ingredients": available,
                        "inventory_strict": strict_inventory,
                        "sources": _search_sources(search_ingredients, available, constraints),
                        "limit": 8,
                    },
                    intent="检索可用于替换的候选菜谱",
                )
            return PlannedAction(
                action="tool",
                tool_name="revise_meal_plan",
                arguments={
                    "instruction": goal,
                    "meal_plan": meal_plan,
                    "available_ingredients": available,
                    "recipes": [
                        {
                            "recipe_id": item.recipe_id,
                            "name": item.name,
                            "ingredients": item.ingredients,
                        }
                        for item in evidence
                    ],
                },
                intent="使用检索结果修改指定日期的菜单",
            )

        if not evidence:
            requested_name = requested_recipe_name(state)
            explicit_recipe = (
                requested_name
                if intent == "recommendation" and requested_name in known_recipe_names()
                else None
            )
            if intent == "recipe_detail" and not requested_name:
                return PlannedAction(
                    action="final",
                    intent="完整菜谱问题缺少明确菜名，需要用户补充",
                )
            if explicit_recipe:
                query = explicit_recipe
            elif intent == "recipe_detail" and requested_name:
                query = requested_name
            elif search_ingredients:
                query = " ".join(search_ingredients)
            elif available:
                query = " ".join(available)
            elif constraints.get("cuisine"):
                query = str(constraints["cuisine"])
            elif intent == "meal_plan":
                query = "家常菜"
            elif intent == "recommendation" and constraints.get("category"):
                query = ""
            else:
                query = normalize_search_query(goal)
            for phrase in ("低糖", "高蛋白", "低脂", "高纤维"):
                if phrase in query:
                    query = phrase
                    break

            return PlannedAction(
                action="tool",
                tool_name="search_recipes",
                arguments={
                    "query": query,
                    "category": constraints.get("category"),
                    "max_difficulty": constraints.get("max_difficulty"),
                    "max_minutes": constraints.get("max_minutes"),
                    "exclude_ingredients": [] if explicit_recipe else excluded,
                    "required_ingredients": search_ingredients,
                    "available_ingredients": available,
                    "inventory_strict": strict_inventory,
                    "sources": (
                        ["hybrid", "graph"]
                        if explicit_recipe
                        else _search_sources(search_ingredients, available, constraints)
                    ),
                    "limit": 5,
                },
                intent="检索满足用户约束的候选菜谱",
            )

        recipe_payload = [
            {
                "recipe_id": item.recipe_id,
                "name": item.name,
                "ingredients": item.ingredients,
            }
            for item in evidence
        ]
        if any(
            word in state["goal"] for word in ("怎么做", "做法", "步骤", "如何做")
        ) and not state.get("recipe_details"):
            requested_name = requested_recipe_name(state) if intent == "recipe_detail" else None
            if intent == "recipe_detail" and not requested_name:
                return PlannedAction(
                    action="final",
                    intent="问题中没有明确的菜名",
                )
            matches = [
                item
                for item in evidence
                if not requested_name
                or item.name == requested_name
                or requested_name in item.name
                or item.name in requested_name
            ]
            if not matches:
                return PlannedAction(
                    action="final",
                    intent="没有找到与问题菜名匹配的完整菜谱",
                )
            return PlannedAction(
                action="tool",
                tool_name="get_recipe",
                arguments={"recipe_id": matches[0].recipe_id},
                intent="读取目标菜谱的完整制作步骤",
            )

        if any(word in state["goal"] for word in ("菜单", "规划", "安排")) and not state.get(
            "meal_plan"
        ):
            return PlannedAction(
                action="tool",
                tool_name="create_meal_plan",
                arguments={
                    "days": int(state.get("constraints", {}).get("days") or 1),
                    "meals_per_day": int(state.get("constraints", {}).get("meals_per_day") or 1),
                    "recipes": recipe_payload,
                },
                intent="使用候选菜谱生成多日菜单",
            )

        if "购物" in state["goal"] and not state.get("shopping_list"):
            meal_plan = state.get("meal_plan") or {}
            required_ingredients = meal_plan.get("required_ingredients") or list(
                dict.fromkeys(ingredient for item in evidence for ingredient in item.ingredients)
            )
            return PlannedAction(
                action="tool",
                tool_name="create_shopping_list",
                arguments={
                    "required_ingredients": required_ingredients,
                    "available_ingredients": available,
                },
                intent="根据最终菜单和库存生成购物清单",
            )

        return PlannedAction(
            action="final",
            intent="当前任务需要的工具已经执行完成",
        )


class LLMRecipePlanner:
    """使用 ChatOpenAI 选择下一个工具或结束任务。"""

    def __init__(
        self,
        registry: ToolRegistry,
        model_name: str,
        api_key: str,
        base_url: str,
    ) -> None:
        self.registry = registry
        self.model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
        )
        self.fallback = RuleBasedPlanner()

    def plan(self, state: RecipeAgentState) -> PlannedAction:
        """调用模型规划下一步，失败时回退到规则规划。"""

        prompt = self._build_prompt(state)
        try:
            with trace_span(
                "gen_ai.chat",
                {
                    "gen_ai.system": "openai-compatible",
                    "gen_ai.operation.name": "chat",
                    "gen_ai.request.model": str(getattr(self.model, "model_name", "unknown")),
                    "agent.component": "planner",
                },
                kind="client",
            ) as span:
                response = self.model.invoke(prompt)
                record_llm_response(span, response)
            content = response.content
            if not isinstance(content, str):
                content = str(content)
            action = PlannedAction.model_validate_json(self._extract_json_object(content))
            return self._validate_action(action, state).model_copy(update={"planner": "llm"})
        except Exception as exc:
            logger.warning("LLM 规划失败，使用规则 Planner：%s", exc)
            return self.fallback.plan(state).model_copy(update={"planner": "llm_fallback"})

    def _build_prompt(self, state: RecipeAgentState) -> str:
        tools = self.registry.schemas()
        evidence = [item.model_dump() for item in state.get("evidence", [])]
        history = state.get("history", [])[-8:]
        meal_plan = state.get("meal_plan")
        recipe_details = [item.model_dump() for item in state.get("recipe_details", [])]
        return f"""
        你是食谱规划 Agent 的 Planner。你的任务是根据当前状态选择下一步。

        【对话历史】
        {json.dumps(history, ensure_ascii=False)}

        【用户目标】
        {state["goal"]}

        【用户约束】
        {json.dumps(state.get("constraints", {}), ensure_ascii=False)}

        【结构化查询计划】
        {json.dumps(state.get("query_plan", {}), ensure_ascii=False)}

        【可用工具】
        {json.dumps(tools, ensure_ascii=False)}

        【已经检索到的菜谱】
        {json.dumps(evidence, ensure_ascii=False)}

        【已经读取的完整菜谱】
        {json.dumps(recipe_details, ensure_ascii=False)}

        【当前菜单】
        {json.dumps(meal_plan, ensure_ascii=False)}

        【当前购物清单】
        {json.dumps(state.get("shopping_list", []), ensure_ascii=False)}

        【当前搭配推荐】
        {json.dumps(state.get("pairing_recommendations", []), ensure_ascii=False)}

        【已有工具调用】
        {len(state.get("tool_calls", []))} 次

        要求：
        1. 如果还需要检索，请选择可用工具并给出合法参数。
        2. 如果当前菜单存在，用户又在要求换菜或调整某一天，先按需检索，再调用 revise_meal_plan。
           如果用户要求素食，检索时使用 category=素菜，不要只把“素菜”当作菜名搜索。
        3. 如果用户要求从购物清单删除食材，调用 revise_shopping_list。
        4. create_meal_plan 只用于第一次生成菜单，不要用它覆盖已有菜单。
        5. 如果证据已经足够且无需修改，返回 action=final。
        6. 不要选择不存在的工具。
        7. 不要把用户过敏原从参数中删除。
        8. 如果用户问“某食材配什么蔬菜”，调用 recommend_pairings，不要反复搜索整份菜谱。

        只返回一个合法 JSON 对象，不要使用 Markdown 代码块，不要输出其他文字。
        格式：
        {{
          "action": "tool",
          "tool_name": "search_recipes",
          "arguments": {{}},
          "intent": "简短说明"
        }}
        """

    @staticmethod
    def _extract_json_object(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Planner 未返回 JSON：{text[:200]!r}")
        return text[start : end + 1]

    def _validate_action(
        self,
        action: PlannedAction,
        state: RecipeAgentState,
    ) -> PlannedAction:
        if action.action == "final":
            return action
        if not action.tool_name or action.tool_name not in self.registry.names():
            raise ValueError(f"Planner 选择了未知工具：{action.tool_name}")

        if action.tool_name == "create_meal_plan" and state.get("meal_plan"):
            if "购物" in state["goal"] and not state.get("shopping_list"):
                return PlannedAction(
                    action="tool",
                    tool_name="create_shopping_list",
                    arguments=self._normalize_shopping_arguments(
                        action.arguments,
                        state,
                    ),
                    intent="菜单已经生成，继续生成购物清单",
                )
            return PlannedAction(
                action="final",
                intent="菜单已经生成，无需重复规划",
            )

        if action.tool_name == "create_shopping_list" and state.get("shopping_list"):
            return PlannedAction(
                action="final",
                intent="购物清单已经生成，可以结束任务",
            )

        if action.tool_name == "get_recipe":
            if state.get("recipe_details"):
                return PlannedAction(
                    action="final",
                    intent="完整菜谱已经读取，无需重复调用",
                )
            evidence = state.get("evidence", [])
            recipe_id = str(action.arguments.get("recipe_id") or "")
            if not recipe_id:
                requested_name = requested_recipe_name(state)
                matches = [
                    item
                    for item in evidence
                    if not requested_name
                    or item.name == requested_name
                    or requested_name in item.name
                    or item.name in requested_name
                ]
                if len(matches) == 1:
                    recipe_id = matches[0].recipe_id
                elif len(evidence) == 1:
                    recipe_id = evidence[0].recipe_id
            if not recipe_id:
                raise ValueError("get_recipe 缺少 recipe_id，且当前没有候选菜谱")
            action.arguments = {"recipe_id": recipe_id}

        if action.tool_name == "revise_meal_plan":
            action.arguments = self._normalize_revision_meal_arguments(
                action.arguments,
                state,
            )

        if action.tool_name == "revise_shopping_list":
            action.arguments = self._normalize_revision_shopping_arguments(
                action.arguments,
                state,
            )

        if action.tool_name == "search_recipes":
            action.arguments = self._normalize_search_arguments(
                action.arguments,
                state,
            )
            if self._is_duplicate_tool_call(action, state):
                return PlannedAction(
                    action="final",
                    intent="相同参数的菜谱检索已经执行，避免重复调用",
                )
        elif action.tool_name == "create_meal_plan":
            action.arguments = self._normalize_meal_plan_arguments(
                action.arguments,
                state,
            )
        elif action.tool_name == "create_shopping_list":
            action.arguments = self._normalize_shopping_arguments(
                action.arguments,
                state,
            )
        return action

    def _normalize_search_arguments(
        self,
        arguments: dict,
        state: RecipeAgentState,
    ) -> dict:
        normalized = dict(arguments)
        constraints = get_constraints(state)
        query = normalize_search_query(str(normalized.get("query") or ""))
        if not query:
            query = " ".join(constraints.get("search_ingredients") or [])
        for tag in ("低糖", "高蛋白", "低脂", "高纤维"):
            if tag in query:
                query = tag
                break
        normalized["query"] = query
        normalized["max_minutes"] = constraints.get("max_minutes")
        normalized["max_difficulty"] = constraints.get("max_difficulty")
        normalized["category"] = normalized.get("category") or constraints.get("category")
        normalized["exclude_ingredients"] = excluded_ingredients(state)
        normalized["required_ingredients"] = constraints.get("search_ingredients") or []
        normalized["available_ingredients"] = available_ingredients(state)
        normalized["inventory_strict"] = inventory_is_strict(state)
        normalized["sources"] = normalized.get("sources") or _search_sources(
            normalized["required_ingredients"],
            normalized["available_ingredients"],
            constraints,
        )
        return normalized

    @staticmethod
    def _is_duplicate_tool_call(
        action: PlannedAction,
        state: RecipeAgentState,
    ) -> bool:
        if not state.get("evidence"):
            return False
        return any(
            call.tool_name == action.tool_name and call.arguments == action.arguments
            for call in state.get("tool_calls", [])
        )

    @staticmethod
    def _latest_recipes(state: RecipeAgentState) -> list[dict]:
        for result in reversed(state.get("tool_results", [])):
            recipes = result.data.get("recipes")
            if recipes:
                return list(recipes)
        return [item.model_dump() for item in state.get("evidence", [])]

    @classmethod
    def _normalize_revision_meal_arguments(
        cls,
        arguments: dict,
        state: RecipeAgentState,
    ) -> dict:
        normalized = dict(arguments)
        normalized["instruction"] = state["goal"]
        normalized["meal_plan"] = state.get("meal_plan") or normalized.get("meal_plan") or {}
        normalized["recipes"] = cls._latest_recipes(state)
        normalized["available_ingredients"] = available_ingredients(state)
        if not normalized["meal_plan"]:
            raise ValueError("revise_meal_plan 缺少当前菜单")
        if not normalized["recipes"]:
            raise ValueError("revise_meal_plan 缺少检索到的候选菜谱")
        return normalized

    @staticmethod
    def _normalize_revision_shopping_arguments(
        arguments: dict,
        state: RecipeAgentState,
    ) -> dict:
        normalized = dict(arguments)
        normalized["instruction"] = state["goal"]
        normalized["shopping_list"] = state.get("shopping_list") or []
        if not normalized["shopping_list"]:
            raise ValueError("revise_shopping_list 缺少当前购物清单")
        return normalized

    @staticmethod
    def _normalize_meal_plan_arguments(
        arguments: dict,
        state: RecipeAgentState,
    ) -> dict:
        normalized = dict(arguments)
        constraints = get_constraints(state)
        normalized["recipes"] = [
            {
                "recipe_id": item.recipe_id,
                "name": item.name,
                "ingredients": item.ingredients,
            }
            for item in state.get("evidence", [])
        ]
        normalized["days"] = int(constraints.get("days") or 1)
        normalized["meals_per_day"] = int(constraints.get("meals_per_day") or 1)
        return normalized

    @staticmethod
    def _normalize_shopping_arguments(
        arguments: dict,
        state: RecipeAgentState,
    ) -> dict:
        normalized = dict(arguments)
        meal_plan = state.get("meal_plan") or {}
        normalized["required_ingredients"] = meal_plan.get("required_ingredients") or list(
            dict.fromkeys(
                ingredient for item in state.get("evidence", []) for ingredient in item.ingredients
            )
        )
        normalized["available_ingredients"] = available_ingredients(state)
        return normalized
