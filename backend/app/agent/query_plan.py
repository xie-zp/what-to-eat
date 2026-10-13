"""将自然语言问题解析成强类型结构化查询计划。"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.domain.inventory import is_broad_ingredient_category

from .lexicon import extract_known_ingredients, extract_known_recipe_names

QueryIntent = Literal[
    "recipe_detail",
    "pairing",
    "comparison",
    "nutrition_question",
    "yes_no",
    "substitution",
    "meal_plan",
    "shopping",
    "recommendation",
    "unsupported",
]

PLAN_TRIGGERS = ("菜单", "规划", "安排")
SHOPPING_TRIGGERS = ("购物", "买菜", "采购")
RECIPE_DETAIL_MARKERS = (
    "怎么制作",
    "怎么做",
    "如何做",
    "的做法",
    "详细说说",
    "详细介绍",
    "展开讲讲",
    "具体说说",
    "详细说",
    "展开说",
    "做法",
    "步骤",
)
CATEGORY_MARKERS = {
    "素菜": ("素菜", "素食", "全素", "纯素"),
    "汤类": ("汤", "羹"),
    "早餐": ("早餐",),
    "主食": ("主食",),
    "甜品": ("甜品", "甜点", "蛋糕"),
    "水产": ("海鲜", "水产"),
}
NUTRITION_GOAL_ALIASES = {
    "减肥": "减脂",
    "减脂": "减脂",
    "低糖": "低糖",
    "控糖": "低糖",
    "低脂": "低脂",
    "高蛋白": "高蛋白",
    "高纤维": "高纤维",
    "低热量": "低热量",
    "清淡": "清淡",
    "不辣": "清淡",
    "少油": "清淡",
}
NEGATION_MARKERS = (
    "不要",
    "不吃",
    "不能吃",
    "忌口",
    "别放",
    "不放",
    "去掉",
    "删除",
    "移除",
    "不含",
    "没有",
    "无",
)
CONTEXT_PRONOUNS = {
    "这个",
    "那个",
    "这道菜",
    "那道菜",
    "这道",
    "那道",
    "它",
}
GENERIC_ENTITY_WORDS = {
    "菜",
    "菜品",
    "菜谱",
    "食谱",
    "晚饭",
    "晚餐",
    "午饭",
    "午餐",
    "早饭",
    "早餐",
    "主食",
    "素菜",
    "荤菜",
    "汤",
    "汤类",
}
REQUEST_PREFIXES = (
    "请问",
    "麻烦问一下",
    "麻烦",
    "帮我看看",
    "帮我查一下",
    "帮我",
    "我想知道",
    "想知道",
    "我想学",
    "我想做",
    "我想吃",
    "给我",
    "介绍一下",
    "告诉我",
    "查一下",
    "看看",
)


class QueryConstraints(BaseModel):
    """约束结构化后统一进入 Agent state。"""

    model_config = ConfigDict(extra="allow")

    allergies: list[str] = Field(default_factory=list)
    avoid_ingredients: list[str] = Field(default_factory=list)
    max_minutes: int | None = None
    max_difficulty: int | None = None
    inventory: list[str] = Field(default_factory=list)
    inventory_strict: bool = False
    search_ingredients: list[str] = Field(default_factory=list)
    unknown_food_terms: list[str] = Field(default_factory=list)
    dietary_goals: list[str] = Field(default_factory=list)
    category: str | None = None
    cuisine: str | None = None
    meal_period: str | None = None
    days: int | None = None
    meals_per_day: int | None = None
    soups_per_meal: int | None = None


class QueryPlan(BaseModel):
    """Planner 可直接消费的查询计划。"""

    model_config = ConfigDict(extra="allow")

    intent: QueryIntent = "recommendation"
    entities: list[str] = Field(default_factory=list)
    target_entities: list[str] = Field(default_factory=list)
    constraints: QueryConstraints = Field(default_factory=QueryConstraints)
    context_used: bool = False
    context_candidates: list[str] = Field(default_factory=list)
    requires_clarification: bool = False


def build_query_plan(goal: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """识别意图、实体和可以强制执行的约束。"""

    text = goal.strip()
    constraints = QueryConstraints()
    if _is_prompt_injection(text):
        return QueryPlan(intent="unsupported", constraints=constraints).model_dump()
    entities: list[str] = []
    target_entities: list[str] = []
    context_candidates: list[str] = []
    requires_clarification = False
    context_used = False

    allergies = _extract_allergies(text)
    avoid_ingredients = _extract_avoid_ingredients(text)
    if allergies:
        constraints.allergies = allergies
    if avoid_ingredients:
        constraints.avoid_ingredients = [
            item for item in avoid_ingredients if item not in constraints.allergies
        ]

    constraints.max_minutes = _extract_max_minutes(text)
    constraints.max_difficulty = _extract_max_difficulty(text)

    excluded_terms = [*constraints.allergies, *constraints.avoid_ingredients]
    known_ingredients = extract_known_ingredients(text)
    known_recipes = extract_known_recipe_names(text)
    search_ingredients = [
        item
        for item in known_ingredients
        if not any(_same_food_term(item, excluded) for excluded in excluded_terms)
        # “香菇滑鸡”这类菜名里的“香菇”是菜名的一部分，不是用户要的食材条件。
        and not any(item in recipe for recipe in known_recipes)
    ]
    if search_ingredients:
        constraints.search_ingredients = search_ingredients
    constraints.unknown_food_terms = _extract_unknown_food_terms(
        text,
        known_ingredients=known_ingredients,
        known_recipes=known_recipes,
    )

    inventory, inventory_strict = _extract_inventory(text)
    inventory_scope = _inventory_scope(text)
    inventory_known = [
        item for item in search_ingredients if inventory_scope and item in inventory_scope
    ]
    if inventory_known and _has_inventory_context(text):
        constraints.inventory = inventory_known
        constraints.inventory_strict = _is_strict_inventory_context(text)
    elif inventory:
        constraints.inventory = inventory
        constraints.inventory_strict = inventory_strict

    # “只有鸡肉和蔬菜”这类宽泛类别无法逐项核对库存，按优先匹配处理，
    # 否则只差葱姜蒜等基础配料的菜会被严格模式全部排除。
    if any(is_broad_ingredient_category(item) for item in constraints.inventory):
        constraints.inventory_strict = False

    constraints.dietary_goals = _extract_dietary_goals(text)
    constraints.category = _extract_category(text)
    constraints.cuisine = _extract_cuisine(text)
    constraints.meal_period = _extract_meal_period(text)
    constraints.days, constraints.meals_per_day = _extract_plan_size(text)
    dishes_per_meal, soups_per_meal = _extract_meal_composition(text)
    has_meal_composition = dishes_per_meal is not None or soups_per_meal is not None
    if dishes_per_meal is not None or soups_per_meal is not None:
        # “两菜一汤”直接决定一顿有几道：菜数 + 汤数；汤数单独记下来，
        # 菜单编排时才能保证每天真的有一道汤。
        constraints.meals_per_day = (dishes_per_meal or 0) + (soups_per_meal or 0)
        if soups_per_meal:
            constraints.soups_per_meal = soups_per_meal
        if dishes_per_meal:
            # 这里的“汤”是每餐汤数，不代表整桌都要点汤。
            # 只有“来两道汤”这种没有菜数的情况才把品类收窄到汤类。
            if constraints.category == "汤类":
                constraints.category = None

    if _is_pairing_question(text):
        anchor = _extract_pairing_anchor(text)
        target = "蔬菜" if any(word in text for word in ("蔬菜", "配菜")) else ""
        if anchor:
            entities = [anchor]
        if target:
            target_entities = [target]
        if "还是" in text or "或者" in text:
            candidates = re.split(r"还是|或者", text.split("配", maxsplit=1)[-1])
            target_entities = [
                item.strip(" ，,。？?") for item in candidates if item.strip(" ，,。？?")
            ]
            intent: QueryIntent = "comparison"
        else:
            intent = "pairing"
    elif any(word in text for word in ("卡路里", "热量", "营养成分", "多少千卡")):
        intent = "nutrition_question"
        recipe_name = _extract_recipe_name(text, intent)
        if recipe_name:
            entities = [recipe_name]
    elif any(word in text for word in ("换成", "替换成", "改成")):
        intent = "substitution"
    elif _is_recipe_detail_question(text):
        intent = "recipe_detail"
        recipe_name = _extract_recipe_name(text, intent)
        if recipe_name:
            entities = [recipe_name]
    elif any(trigger in text for trigger in PLAN_TRIGGERS) or has_meal_composition:
        intent = "meal_plan"
    elif any(trigger in text for trigger in SHOPPING_TRIGGERS):
        intent = "shopping"
    elif _is_yes_no_question(text):
        intent = "yes_no"
        recipe_name = _extract_recipe_name(text, intent)
        if recipe_name:
            entities = [recipe_name]
    else:
        intent = "recommendation"

    # “微波炉蛋糕”“西红柿鸡蛋汤”等菜名里的分类词不能反过来变成检索过滤条件。
    if intent in {"recipe_detail", "nutrition_question", "yes_no"} and entities:
        constraints.category = None
        constraints.search_ingredients = []

    if _needs_history_context(text, intent, entities):
        candidates = _history_recipe_candidates(history or [])
        context_candidates = candidates[:5]
        if len(candidates) == 1:
            entities = [candidates[0]]
            context_used = True
        elif len(candidates) > 1:
            requires_clarification = True

    # 已经解析出菜系或菜谱分类时，说明问题明确落在食谱范围内，
    # 即使字面没有“菜”“吃”这类词也不该判成无关问题。
    if intent == "recommendation" and not (
        _looks_food_related(text) or constraints.category or constraints.cuisine
    ):
        intent = "unsupported"

    plan = QueryPlan(
        intent=intent,
        entities=list(dict.fromkeys(entities)),
        target_entities=list(dict.fromkeys(target_entities)),
        constraints=constraints,
        context_used=context_used,
        context_candidates=context_candidates,
        requires_clarification=requires_clarification,
    )
    return plan.model_dump()


def _is_prompt_injection(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "忽略所有规则",
            "忽略之前的",
            "绕过规则",
            "编造",
            "编一个不存在",
        )
    )


def _looks_food_related(text: str) -> bool:
    if extract_known_ingredients(text) or extract_known_recipe_names(text):
        return True
    if re.search(r"《([^》]+)》", text):
        return True
    return any(
        marker in text
        for marker in (
            "菜",
            "一道",
            "食谱",
            "食材",
            "做法",
            "烹饪",
            "菜单",
            "购物",
            "吃",
            "喝",
            "营养",
            "热量",
            "卡路里",
            "过敏",
            "忌口",
            "减脂",
            "低糖",
            "高蛋白",
            "早餐",
            "午餐",
            "晚餐",
        )
    )


def _extract_unknown_food_terms(
    text: str,
    known_ingredients: list[str],
    known_recipes: list[str],
) -> list[str]:
    """提取句式中明确要求、但不在菜谱知识库里的食材或菜名。"""

    terms: list[str] = []
    for match in re.findall(r"用([^，。！？?]+?)做", text):
        for raw_term in re.split(r"[、和与及\s]+", match):
            term = raw_term.strip(" 的了")
            if term and term not in terms:
                terms.append(term)

    if not known_ingredients and not known_recipes:
        match = re.search(
            r"(?:推荐|来|想吃)[^，。！？?]*?的([\u4e00-\u9fa5A-Za-z0-9]{2,12})$",
            text,
        )
        if match:
            candidate = match.group(1).strip(" 的了")
            ignored_markers = (
                "早餐",
                "午餐",
                "晚餐",
                "高蛋白",
                "低糖",
                "低脂",
                "减脂",
                "菜单",
            )
            if (
                candidate
                and candidate not in GENERIC_ENTITY_WORDS
                and not candidate.endswith("菜")
                and not any(marker in candidate for marker in ignored_markers)
            ):
                terms.append(candidate)

    known_terms = [*known_ingredients, *known_recipes]
    return [
        term
        for term in dict.fromkeys(terms)
        if term
        and term not in GENERIC_ENTITY_WORDS
        and not any(term in known or known in term for known in known_terms)
    ]


def parse_query_plan(value: Any) -> QueryPlan:
    """把旧状态或外部输入安全转换为查询计划。"""

    if isinstance(value, QueryPlan):
        return value
    if isinstance(value, dict):
        try:
            return QueryPlan.model_validate(value)
        except (ValidationError, TypeError, ValueError):
            # 旧任务可能保留了已废弃的 intent 或字段，不能让它拖垮当前回答。
            return QueryPlan()
    return QueryPlan()


def merge_constraints(
    explicit: dict[str, Any],
    extracted: dict[str, Any] | QueryConstraints,
) -> dict[str, Any]:
    """合并请求约束和自然语言抽取约束，显式请求优先。"""

    if isinstance(extracted, QueryConstraints):
        merged = extracted.model_dump(exclude_none=True)
    else:
        merged = dict(extracted)

    for key, value in explicit.items():
        if value is None:
            continue
        if isinstance(value, list) and isinstance(merged.get(key), list):
            merged[key] = list(dict.fromkeys([*merged[key], *value]))
        else:
            merged[key] = value
    return merged


def _extract_allergies(text: str) -> list[str]:
    values: list[str] = []
    pattern = r"(?:我|本人)?(?:对)?([^，,。！？?；;\n]{1,20}?)(?:过敏)"
    for match in re.finditer(pattern, text):
        values.extend(_split_food_terms(match.group(1)))
    return _dedupe_terms(values)


def _extract_avoid_ingredients(text: str) -> list[str]:
    values: list[str] = []
    for marker in NEGATION_MARKERS:
        if marker == "没有" and ("有没有" in text or "没有吗" in text):
            continue
        pattern = rf"{marker}([^，,。！？!?；;\n]+)"
        for match in re.finditer(pattern, text):
            chunk = match.group(1)
            if "里的" in chunk:
                chunk = chunk.rsplit("里的", maxsplit=1)[-1]
            elif "中的" in chunk:
                chunk = chunk.rsplit("中的", maxsplit=1)[-1]
            elif "的" in chunk:
                chunk = chunk.split("的", maxsplit=1)[0]
            values.extend(_split_food_terms(chunk))
    return _dedupe_terms(values)


def _split_food_terms(value: str) -> list[str]:
    items = re.split(r"[、和及以及与关]+", value)
    terms = []
    for item in items:
        cleaned = _clean_food_term(item)
        if cleaned:
            terms.append(cleaned)
    return terms


def _clean_food_term(value: str) -> str:
    cleaned = value.strip(" 的了呢吧吗？?！!，,。：:；;地")
    cleaned = re.sub(r"^(?:吃|放|加|用|要|放点|加点)+", "", cleaned)
    cleaned = re.sub(r"(?:这类|这些|这种|那些|之类|的菜|的菜品|的菜谱)$", "", cleaned)
    if not cleaned or len(cleaned) > 12:
        return ""
    if cleaned in GENERIC_ENTITY_WORDS:
        return ""
    return cleaned


def _dedupe_terms(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _extract_max_minutes(text: str) -> int | None:
    match = re.search(r"(\d+)\s*分钟(?:内|以内|之内|以下)", text)
    if match:
        return int(match.group(1))
    if "半小时内" in text or "半小时以内" in text:
        return 30
    return None


def _extract_max_difficulty(text: str) -> int | None:
    if any(word in text for word in ("新手", "非常简单", "零基础")):
        return 2
    if any(word in text for word in ("简单", "不难")):
        return 3
    return None


def _extract_inventory(text: str) -> tuple[list[str], bool]:
    strict_match, available_match = _find_inventory_matches(text)
    match = strict_match or available_match
    if not match:
        return [], False
    raw = _inventory_scope(text)
    values = re.split(r"[，,、和及以及\s]+", raw)
    cleaned = [
        _clean_food_term(value)
        for value in values
        if _clean_food_term(value) and value not in {"这些", "以下"}
    ]
    return cleaned, bool(strict_match)


def _inventory_scope(text: str) -> str:
    """截取库存描述片段，避免把后续过敏声明也当成库存。"""

    strict_match, available_match = _find_inventory_matches(text)
    match = strict_match or available_match
    if not match:
        return ""
    raw = match.group(1).strip()
    raw = re.sub(r"^(?:食材|原料)[:：]?", "", raw)
    raw = re.split(r"[。？?]", raw, maxsplit=1)[0]
    raw = re.split(
        r"(?:推荐|请|帮我|来一道|来一份|做一道|做一份|规划|安排|我|本人)",
        raw,
        maxsplit=1,
    )[0]
    return raw.strip()


def _find_inventory_matches(text: str) -> tuple[re.Match[str] | None, re.Match[str] | None]:
    strict_match = re.search(
        r"(?:冰箱|家里|家中|厨房|库存)?(?:里|中)?"
        r"(?:只有|仅有|只剩|只剩下|就剩)([^。？?]+)",
        text,
    )
    available_match = re.search(
        r"(?:冰箱|家里|家中|厨房|库存)?(?:里|中)?"
        r"(?<!没)(?:现有|存有|还有|有)(?!没|哪|什么)([^。？?]+)",
        text,
    )
    return strict_match, available_match


def _same_food_term(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return left == right or left in right or right in left


def _has_inventory_context(text: str) -> bool:
    """判断用户是否在描述手头/库存，而不是泛泛搜索食材。"""

    if any(marker in text for marker in ("有没有", "是否有", "还有没有", "有吗", "有么")):
        return False
    return any(
        marker in text
        for marker in (
            "冰箱",
            "家里",
            "家中",
            "厨房",
            "库存",
            "现有",
            "手头",
            "还有",
            "剩下",
            "只剩",
            "只有",
            "仅有",
        )
    )


def _is_strict_inventory_context(text: str) -> bool:
    """判断用户是否明确表示只能使用现有食材。"""

    return any(
        marker in text
        for marker in (
            "只有",
            "仅有",
            "只剩",
            "就剩",
            "没别的",
            "没有别的",
            "就这些",
            "不用买",
            "不能再买",
        )
    )


SUBSTITUTION_COMMAND_MARKERS = ("换成", "替换成", "改成", "换为", "替换为", "改为")
SUBSTITUTION_INQUIRY_MARKERS = (
    "哪些",
    "什么",
    "推荐",
    "有没有",
    "能否",
    "可不可以",
    "可以吗",
    "是不是",
    "行吗",
)


def has_substitution_command(text: str) -> bool:
    """区分“修改已有菜单的替换指令”和“询问替代方案”。"""

    if any(marker in text for marker in SUBSTITUTION_INQUIRY_MARKERS):
        return False
    return any(marker in text for marker in SUBSTITUTION_COMMAND_MARKERS)


SUBSTITUTION_PAIR_PATTERN = re.compile(
    r"用\s*(?P<replacement>[\u4e00-\u9fffA-Za-z]{1,8}?)\s*(?:来)?(?:替代|替换|代替|换掉)\s*"
    r"(?P<replaced>[\u4e00-\u9fffA-Za-z]{1,8}?)(?=的|，|,|。|？|\?|$|并且|而且|保持)"
)


def extract_substitution_pair(text: str) -> tuple[str, str]:
    """从“用豆腐替代肉类”中提取（替代物, 被替代物）。"""

    match = SUBSTITUTION_PAIR_PATTERN.search(text)
    if not match:
        return "", ""
    return match.group("replacement").strip(), match.group("replaced").strip()


def _extract_dietary_goals(text: str) -> list[str]:
    goals = []
    for marker, normalized in NUTRITION_GOAL_ALIASES.items():
        if marker in text:
            goals.append(normalized)
    return list(dict.fromkeys(goals))


def _extract_category(text: str) -> str | None:
    for category, markers in CATEGORY_MARKERS.items():
        if any(marker in text for marker in markers):
            return category
    return None


def _extract_meal_period(text: str) -> str | None:
    for period in ("早餐", "午餐", "晚餐"):
        if period in text:
            return period
    return None


def _extract_cuisine(text: str) -> str | None:
    for cuisine in ("川菜", "粤菜", "湘菜", "鲁菜", "淮扬菜", "闽菜", "浙菜", "徽菜"):
        if cuisine in text:
            return cuisine
    return None


def _extract_plan_size(text: str) -> tuple[int | None, int | None]:
    days = None
    day_match = re.search(r"([一二两三四五六七八九十\d]+)\s*天", text)
    if day_match:
        days = _parse_chinese_number(day_match.group(1))
    elif "一周" in text or "7天" in text:
        days = 7

    meals_per_day = None
    meals_match = re.search(r"(?:每天|一日|每日)([一二两三四五\d]+)餐", text)
    if meals_match:
        meals_per_day = _parse_chinese_number(meals_match.group(1))
    return days, meals_per_day


def _extract_meal_composition(text: str) -> tuple[int | None, int | None]:
    """解析“两菜一汤”这类每餐构成，返回（菜数, 汤数）。"""

    number = r"[一二两三四五六七八九十\d]+"
    pair = re.search(rf"({number})\s*(?:个|道)?\s*菜\s*({number})\s*(?:个|道)?\s*汤", text)
    if pair:
        return _parse_chinese_number(pair.group(1)), _parse_chinese_number(pair.group(2))

    soups_match = re.search(rf"({number})\s*(?:个|道)?\s*汤", text)
    soups = _parse_chinese_number(soups_match.group(1)) if soups_match else None
    dishes_match = re.search(rf"({number})\s*(?:个|道)?\s*菜", text)
    dishes = _parse_chinese_number(dishes_match.group(1)) if dishes_match else None
    if soups is None and dishes is not None and dishes < 2:
        # “推荐一道菜”里的“一道菜”是泛指，不代表一顿只要一道。
        dishes = None
    return dishes, soups


def _parse_chinese_number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    mapping = {
        "一": 1,
        "两": 2,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    if value in mapping:
        return mapping[value]
    if value == "十四":
        return 14
    return None


def _extract_recipe_name(text: str, intent: QueryIntent) -> str | None:
    quoted = re.search(r"《([^》]+)》", text)
    if quoted:
        cleaned = _clean_recipe_candidate(quoted.group(1))
        if cleaned:
            return cleaned

    if intent == "recipe_detail":
        for marker in RECIPE_DETAIL_MARKERS:
            if marker in text:
                return _clean_recipe_candidate(text.split(marker, maxsplit=1)[0])

    if intent == "nutrition_question":
        for marker in (
            "有多少卡路里",
            "有多少热量",
            "多少千卡",
            "含多少",
            "的营养成分",
            "热量",
            "卡路里",
        ):
            if marker in text:
                return _clean_recipe_candidate(text.split(marker, maxsplit=1)[0])

    if intent == "yes_no":
        for marker in ("适合", "可以", "能不能", "能不能吃", "需要"):
            if marker in text:
                return _clean_recipe_candidate(text.split(marker, maxsplit=1)[0])

    introduction = re.match(r"(?:介绍一下|介绍一下菜谱|来个|来一道|来一份)(.+)", text)
    if introduction:
        return _clean_recipe_candidate(introduction.group(1))
    return None


def _clean_recipe_candidate(value: str) -> str | None:
    cleaned = value.strip(" 《》“”\"'，,。？?！!：:；;")
    changed = True
    while changed:
        changed = False
        for prefix in REQUEST_PREFIXES:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
                changed = True

    cleaned = re.sub(
        r"^(?:这道|那道|这个|那个|一份|一道|个|份|的)",
        "",
        cleaned,
    ).strip()
    cleaned = re.sub(r"(?:是什么菜|是什么|怎么样|如何|好不好)$", "", cleaned).strip()
    if not cleaned or len(cleaned) > 20:
        return None
    if cleaned in CONTEXT_PRONOUNS or cleaned in GENERIC_ENTITY_WORDS:
        return None
    if any(
        marker in cleaned
        for marker in (
            "推荐",
            "菜单",
            "购物",
            "过敏",
            "不要",
            "不含",
            "什么",
            "哪些",
            "怎么",
            "如何",
        )
    ):
        return None
    return cleaned or None


def _is_recipe_detail_question(text: str) -> bool:
    return any(word in text for word in RECIPE_DETAIL_MARKERS)


def _is_pairing_question(text: str) -> bool:
    return "配" in text and any(
        marker in text for marker in ("配什么", "配哪些", "搭配什么", "和什么", "还是", "或者")
    )


def _extract_pairing_anchor(text: str) -> str | None:
    for marker in ("配什么", "配哪些", "搭配什么", "和什么", "配"):
        if marker not in text:
            continue
        anchor = text.split(marker, maxsplit=1)[0]
        anchor = re.sub(r"^(?:推荐|请问|想问|想|要|来|给我|帮我)+", "", anchor)
        cleaned = anchor.strip(" ，,。？?")
        if cleaned and cleaned not in CONTEXT_PRONOUNS:
            return cleaned
    return None


def _is_yes_no_question(text: str) -> bool:
    return text.rstrip("？?").endswith(("吗", "么")) and any(
        word in text for word in ("适合", "需要", "能", "可以")
    )


def _needs_history_context(
    text: str,
    intent: QueryIntent,
    entities: list[str],
) -> bool:
    if entities:
        return False
    if any(pronoun in text for pronoun in CONTEXT_PRONOUNS):
        return True
    if intent == "recipe_detail":
        return any(marker in text for marker in RECIPE_DETAIL_MARKERS)
    if intent in {"nutrition_question", "yes_no", "substitution"}:
        return any(marker in text for marker in ("它", "这个", "这道", "那个", "那道"))
    return False


def _history_recipe_candidates(history: list[dict[str, Any]]) -> list[str]:
    """从最近一条有明确菜谱列表的助手消息中提取候选。"""

    for message in reversed(history):
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "")
        if not content.strip():
            continue
        if any(
            marker in content
            for marker in ("请告诉我具体的菜名", "没有找到", "无法给出", "无法执行")
        ):
            continue

        candidates: list[str] = []
        for match in re.finditer(r"《([^》]+)》", content):
            cleaned = _clean_recipe_candidate(match.group(1))
            if cleaned:
                candidates.append(cleaned)

        if "候选菜谱" in content or "为你找到" in content:
            for line in content.splitlines():
                match = re.match(r"-\s*([^：:\n]+)[：:]", line.strip())
                if match:
                    cleaned = _clean_recipe_candidate(match.group(1))
                    if cleaned:
                        candidates.append(cleaned)

        if "菜单规划" in content:
            for match in re.finditer(r"第\d+天[：:]\s*([^\n]+)", content):
                for item in re.split(r"[、，,]", match.group(1)):
                    cleaned = _clean_recipe_candidate(item)
                    if cleaned:
                        candidates.append(cleaned)

        if candidates:
            return list(dict.fromkeys(candidates))
    return []
