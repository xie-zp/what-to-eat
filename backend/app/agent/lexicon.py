"""从真实菜谱数据构建查询实体词典。"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

NON_INGREDIENT_TERMS = {
    "冰箱",
    "工具",
    "原料",
    "厨具",
    "容器",
    "微波炉",
    "烤箱",
    "空气炸锅",
    "电饭煲",
    "炒锅",
    "平底锅",
    "蒸锅",
    "保鲜膜",
    "锡纸",
}


def extract_known_ingredients(text: str) -> list[str]:
    """用数据词典做实体链接，不依赖用户使用了哪一种提问句式。"""

    matches = [term for term in known_ingredients() if term in text]
    if not matches:
        return []

    # 只保留最长匹配，避免一个食材既命中全名又命中其子串。
    result = []
    for term in sorted(matches, key=len, reverse=True):
        if any(term in existing for existing in result):
            continue
        result.append(term)
    return sorted(result, key=lambda item: text.index(item))


def extract_known_recipe_names(text: str) -> list[str]:
    """从用户表达中识别真实菜名。"""

    matches = [name for name in known_recipe_names() if name in text]
    return sorted(matches, key=lambda item: text.index(item))


@lru_cache(maxsize=1)
def known_ingredients() -> tuple[str, ...]:
    """加载当前菜品库中出现过的食材名称和别名。"""

    path = Path(os.getenv("RECIPE_DATA_PATH", "./data/generated/recipes.json"))
    if not path.exists():
        return ()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()

    terms: set[str] = set()
    for recipe in data:
        for ingredient in recipe.get("ingredients") or []:
            if not isinstance(ingredient, dict):
                continue
            name = str(ingredient.get("name") or "").strip()
            if len(name) >= 2 and name not in NON_INGREDIENT_TERMS:
                terms.add(name)
            raw_text = str(ingredient.get("raw_text") or "")
            for alias in re.findall(r"别称[:：]\s*([^）)]+)", raw_text):
                cleaned = alias.strip(" ，,、")
                if len(cleaned) >= 2 and cleaned not in NON_INGREDIENT_TERMS:
                    terms.add(cleaned)

    return tuple(sorted(terms, key=lambda item: (-len(item), item)))


@lru_cache(maxsize=1)
def known_recipe_names() -> tuple[str, ...]:
    """加载真实菜谱名称。"""

    path = Path(os.getenv("RECIPE_DATA_PATH", "./data/generated/recipes.json"))
    if not path.exists():
        return ()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    return tuple(
        sorted(
            {
                str(recipe.get("name") or "").strip()
                for recipe in data
                if str(recipe.get("name") or "").strip()
            },
            key=lambda item: (-len(item), item),
        )
    )
