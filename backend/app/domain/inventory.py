"""库存约束的共享判断逻辑。"""

from __future__ import annotations

import re

from app.knowledge_graph.ingredient_classifier import classify_ingredient

PANTRY_INGREDIENTS = {
    "水",
    "清水",
    "开水",
    "热水",
    "温水",
    "凉白开水",
    "盐",
    "食用盐",
    "糖",
    "白糖",
    "白砂糖",
    "绵白糖",
    "冰糖",
    "砂糖",
    "生抽",
    "老抽",
    "酱油",
    "醋",
    "料酒",
    "蚝油",
    "食用油",
    "香油",
    "辣椒油",
    "胡椒",
    "花椒",
    "味精",
    "鸡精",
    "豆瓣酱",
    "番茄酱",
    "沙拉酱",
    "芝麻酱",
    "甜面酱",
    "黄豆酱",
    "蜂蜜",
    "八角",
    "桂皮",
    "香叶",
    "草果",
    "孜然",
    "咖喱",
    "五香粉",
    "十三香",
    "淀粉",
    "生粉",
    "酵母",
    "泡打粉",
    "小苏打",
    "香草",
    "迷迭香",
    "罗勒",
    "百里香",
    "陈皮",
    "白芷",
    "丁香",
}
NON_INGREDIENT_LABELS = {
    "原料",
    "原料:",
    "原料：",
    "工具",
    "厨具",
}
INGREDIENT_ALIASES = {
    "花菜": {"花菜", "花椰菜", "菜花"},
    "西兰花": {"西兰花", "西蓝花"},
    "西红柿": {"西红柿", "番茄"},
    "土豆": {"土豆", "马铃薯"},
    "鸡肉": {
        "鸡肉",
        "鸡胸肉",
        "鸡胸",
        "鸡腿",
        "鸡腿肉",
        "鸡翅",
        "鸡翅中",
        "鸡丁",
        "鸡块",
        "鸡丝",
        "大鸡腿",
    },
}


def recipe_matches_inventory(
    ingredients: list[str],
    available_ingredients: list[str],
) -> bool:
    """严格库存模式下只保留现有食材和基础调料可以完成的菜。"""

    available = [item.strip() for item in available_ingredients if item.strip()]
    if not available:
        return False
    for ingredient in ingredients:
        if not _ingredient_is_satisfied(ingredient, available):
            return False
    return True


def recipe_contains_required_ingredients(
    ingredients: list[str],
    required_ingredients: list[str],
) -> bool:
    """检查菜谱是否包含查询明确要求的全部食材。"""

    required = [item.strip() for item in required_ingredients if item.strip()]
    if not required:
        return True
    recipe_terms = [_alias_terms(item) for item in ingredients if item]
    return all(
        any(_alias_terms(term) & recipe_term for recipe_term in recipe_terms) for term in required
    )


def recipe_contains_any_ingredient(
    ingredients: list[str],
    candidates: list[str],
) -> bool:
    """检查菜谱是否至少包含一个候选食材。"""

    if not candidates:
        return True
    recipe_terms = [_alias_terms(item) for item in ingredients if item]
    return any(
        any(_alias_terms(candidate) & recipe_term for recipe_term in recipe_terms)
        for candidate in candidates
    )


def expand_ingredient_aliases(term: str) -> set[str]:
    """返回食材的规范名和常见别名，供 BM25/图谱查询扩展。"""

    return _alias_terms(term)


def is_pantry_ingredient(term: str) -> bool:
    """判断基础调料或水这类默认不进入采购清单的食材。"""

    return _is_pantry_item(term)


def _ingredient_is_satisfied(ingredient: str, available: list[str]) -> bool:
    """食材和库存之间使用受控匹配，避免长描述误命中单个调料字。"""

    raw = ingredient.strip()
    if not raw:
        return True
    if raw.startswith("可选"):
        return True
    if len(raw) > 24 and re.search(r"[：:，,。！!？?]", raw):
        return False

    alternatives = [item for item in re.split(r"[或/]", raw) if item.strip()]
    for alternative in alternatives:
        required_terms = [
            item.strip() for item in re.split(r"[，,、和及]", alternative) if item.strip()
        ]
        if required_terms and all(_term_is_available(term, available) for term in required_terms):
            return True
    return False


def _term_is_available(term: str, available: list[str]) -> bool:
    normalized = term.strip(" 　:：")
    normalized = re.sub(r"^(?:必备|必需|可选)[:：]", "", normalized).strip()
    if normalized in NON_INGREDIENT_LABELS:
        return True
    if classify_ingredient(normalized) == "工具":
        return True
    if _is_pantry_item(normalized):
        return True
    for item in available:
        if _alias_terms(normalized) & _alias_terms(item):
            return True
        if len(normalized) <= 8 and (item in normalized or normalized in item):
            return True
    return False


def _is_pantry_item(ingredient: str) -> bool:
    normalized = re.sub(r"^(?:必备|必需|可选)[:：]", "", ingredient).strip()
    if len(normalized) > 8:
        return False
    return any(pantry_item in normalized for pantry_item in PANTRY_INGREDIENTS)


def _alias_terms(term: str) -> set[str]:
    """展开常见食材别名，避免花菜和花椰菜被当成两种东西。"""

    expanded = {term}
    for canonical, aliases in INGREDIENT_ALIASES.items():
        if term == canonical or any(alias in term or term in alias for alias in aliases):
            expanded.update(aliases)
    return expanded
