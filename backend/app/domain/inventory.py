"""库存约束的共享判断逻辑。"""

from __future__ import annotations

import re

from app.knowledge_graph.ingredient_classifier import (
    AQUATIC_KEYWORDS,
    BEAN_KEYWORDS,
    EGG_DAIRY_KEYWORDS,
    FRUIT_KEYWORDS,
    MEAT_KEYWORDS,
    STAPLE_KEYWORDS,
    VEGETABLE_KEYWORDS,
    classify_ingredient,
)

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

# 用户常用“蔬菜”“肉类”这类宽泛说法描述库存，需要按食材大类匹配，
# 否则严格库存模式会把西兰花、青椒等蔬菜全部判成缺失。
BROAD_CATEGORY_TERMS = {
    "蔬菜": "蔬菜",
    "青菜": "蔬菜",
    "叶菜": "蔬菜",
    "菌菇": "蔬菜",
    "肉类": "肉类",
    "肉": "肉类",
    "水产": "水产",
    "海鲜": "水产",
    "蛋奶": "蛋奶",
    "豆制品": "豆制品",
    "主食": "主食",
    "水果": "水果",
}

CATEGORY_INGREDIENT_KEYWORDS = {
    "蔬菜": VEGETABLE_KEYWORDS,
    "肉类": MEAT_KEYWORDS,
    "水产": AQUATIC_KEYWORDS,
    "蛋奶": EGG_DAIRY_KEYWORDS,
    "豆制品": BEAN_KEYWORDS,
    "主食": STAPLE_KEYWORDS,
    "水果": FRUIT_KEYWORDS,
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
    return all(
        any(_ingredient_matches(item, term) for item in ingredients if item) for term in required
    )


def recipe_contains_any_ingredient(
    ingredients: list[str],
    candidates: list[str],
) -> bool:
    """检查菜谱是否至少包含一个候选食材。"""

    if not candidates:
        return True
    return any(
        _ingredient_matches(item, candidate)
        for item in ingredients
        if item
        for candidate in candidates
    )


def is_broad_ingredient_category(term: str) -> bool:
    """判断用户输入是否为蔬菜、肉类这类宽泛食材类别。"""

    return term.strip() in BROAD_CATEGORY_TERMS


def expand_ingredient_aliases(term: str) -> set[str]:
    """返回食材的规范名和常见别名，供 BM25/图谱查询扩展。

    宽泛类别词会展开成该类别下的代表食材，例如“蔬菜”会带上西兰花、
    青椒等具体名称，避免只用类别字面量召回。"""

    category = BROAD_CATEGORY_TERMS.get(term.strip())
    if category is not None:
        return {term.strip(), *CATEGORY_INGREDIENT_KEYWORDS.get(category, set())}
    return _alias_terms(term)


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
        if _ingredient_matches(normalized, item):
            return True
        if _alias_terms(normalized) & _alias_terms(item):
            return True
        if len(normalized) <= 8 and (item in normalized or normalized in item):
            return True
    return False


def _ingredient_matches(recipe_ingredient: str, candidate: str) -> bool:
    """判断菜谱食材是否满足候选词，支持别名和宽泛食材类别。"""

    category = BROAD_CATEGORY_TERMS.get(candidate.strip())
    if category is not None:
        return classify_ingredient(recipe_ingredient) == category
    return bool(_alias_terms(recipe_ingredient) & _alias_terms(candidate))


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
