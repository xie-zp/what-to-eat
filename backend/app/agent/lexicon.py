"""从真实菜谱数据构建查询实体词典。"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

from app.domain.inventory import BROAD_CATEGORY_TERMS, INGREDIENT_ALIASES

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

    positions = {term: text.index(term) for term in known_ingredients() if term in text}
    for canonical, aliases in INGREDIENT_ALIASES.items():
        for alias in aliases:
            if alias in text:
                positions.setdefault(canonical, text.index(alias))
                break
    if not positions:
        return []

    # 只保留最长匹配，避免一个食材既命中全名又命中其子串。
    result = []
    for term in sorted(positions, key=len, reverse=True):
        if any(term in existing for existing in result):
            continue
        result.append(term)
    return sorted(result, key=positions.get)


def extract_known_recipe_names(text: str) -> list[str]:
    """从用户表达中识别真实菜名。"""

    matches = [name for name in known_recipe_names() if name in text]
    return sorted(matches, key=lambda item: text.index(item))


@lru_cache(maxsize=1)
def _recipe_records() -> tuple[dict, ...]:
    """加载菜谱记录，生成语料缺失时退化到内置内存菜谱库。"""

    records = _load_generated_recipes()
    return records or _memory_recipes()


def _load_generated_recipes() -> tuple[dict, ...]:
    """读取脚本生成的完整菜谱语料，未生成或损坏时返回空。"""

    path = Path(os.getenv("RECIPE_DATA_PATH", "./data/generated/recipes.json"))
    if not path.exists():
        return ()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    if not isinstance(data, list):
        return ()
    return tuple(recipe for recipe in data if isinstance(recipe, dict))


def _memory_recipes() -> tuple[dict, ...]:
    """把内置菜谱库转成与语料一致的结构。"""

    # 延迟导入，避免 agent 层和 tools 层在导入期互相牵连。
    from app.tools.catalog import RECIPES

    return tuple(
        {
            "name": recipe.name,
            "ingredients": [{"name": name} for name in recipe.ingredients],
        }
        for recipe in RECIPES
    )


def _base_terms() -> set[str]:
    """语料之外，产品本身就支持的基础词表。"""

    terms: set[str] = set(INGREDIENT_ALIASES)
    # “蔬菜”“肉类”这类宽泛说法是合法输入，不该取决于语料是否收录。
    terms.update(term for term in BROAD_CATEGORY_TERMS if len(term) >= 2)
    return terms


@lru_cache(maxsize=1)
def known_ingredients() -> tuple[str, ...]:
    """加载当前菜品库中出现过的食材名称和别名。"""

    terms = _base_terms()
    for recipe in _recipe_records():
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

    return tuple(
        sorted(
            {
                str(recipe.get("name") or "").strip()
                for recipe in _recipe_records()
                if str(recipe.get("name") or "").strip()
            },
            key=lambda item: (-len(item), item),
        )
    )
