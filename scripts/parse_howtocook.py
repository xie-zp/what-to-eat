"""将 HowToCook Markdown 菜谱解析为标准化 JSON。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from app.domain.ingredients import clean_ingredient_name as normalize_ingredient_name
from app.domain.recipe import ParsedIngredient, ParsedRecipe

HEADING_PATTERN = re.compile(r"(?m)^##\s+(.+?)\s*$")
LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.+?)\s*$")
DIFFICULTY_PATTERN = re.compile(r"★")
NUMBER_PATTERN = r"\d+(?:\.\d+)?"
TIME_UNIT_PATTERN = re.compile(
    rf"(?P<value>{NUMBER_PATTERN})\s*(?P<unit>分钟|分|小时|个小时|h|天)",
    re.IGNORECASE,
)
COMBINED_TIME_PATTERN = re.compile(
    rf"(?P<hours>{NUMBER_PATTERN})\s*(?:个)?小时\s*(?P<minutes>{NUMBER_PATTERN})\s*分钟",
    re.IGNORECASE,
)
CHINESE_NUMBER_PATTERN = re.compile(
    r"(?P<value>[零一二两三四五六七八九十半]+)\s*(?P<unit>分钟|分|小时|个小时|天)"
)
CHINESE_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

CATEGORY_NAMES = {
    "aquatic": "水产",
    "breakfast": "早餐",
    "condiment": "调料",
    "dessert": "甜品",
    "drink": "饮料",
    "meat_dish": "荤菜",
    "semi-finished": "半成品",
    "soup": "汤类",
    "staple": "主食",
    "vegetable_dish": "素菜",
}


def parse_recipe(path: Path, root: Path) -> ParsedRecipe:
    """解析单个 Markdown 文件。"""

    content = path.read_text(encoding="utf-8")
    relative_path = path.relative_to(root)
    recipe_id = "htc-" + hashlib.sha1(relative_path.as_posix().encode("utf-8")).hexdigest()[:12]

    sections = split_sections(content)
    intro = sections.pop("__intro__", "")
    name = extract_name(intro, path.stem)
    difficulty = len(DIFFICULTY_PATTERN.findall(intro))
    description = extract_description(intro)
    estimated_minutes = extract_estimated_minutes(description)
    estimated_calories = extract_estimated_calories(description)
    nutrition_tags = extract_nutrition_tags(description)

    ingredient_lines = section_bullets(
        sections,
        required_keywords=("必备原料", "原料和工具"),
    )
    quantity_notes = section_bullets(
        sections,
        required_keywords=("计算",),
    )
    steps = section_bullets(
        sections,
        required_keywords=("操作", "步骤"),
    )
    notes = section_bullets(
        sections,
        required_keywords=("附加内容", "附加说明"),
    )

    ingredients = []
    for line in ingredient_lines:
        ingredient_name = clean_ingredient_name(line)
        if not ingredient_name:
            continue
        ingredients.append(
            ParsedIngredient(
                name=ingredient_name,
                raw_text=line,
                amount_text=find_quantity_text(line, quantity_notes),
            )
        )

    category_key = relative_path.parts[0] if relative_path.parts else "unknown"
    return ParsedRecipe(
        recipe_id=recipe_id,
        name=name,
        category=CATEGORY_NAMES.get(category_key, category_key),
        difficulty=difficulty,
        estimated_minutes=estimated_minutes,
        description=description,
        estimated_calories=estimated_calories,
        nutrition_tags=nutrition_tags,
        ingredients=ingredients,
        quantity_notes=quantity_notes,
        steps=steps,
        notes=notes,
        source_path=relative_path.as_posix(),
    )


def split_sections(content: str) -> dict[str, str]:
    """把 Markdown 按二级标题拆成章节。"""

    matches = list(HEADING_PATTERN.finditer(content))
    if not matches:
        return {"__intro__": content}

    sections = {"__intro__": content[: matches[0].start()].strip()}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        sections[match.group(1).strip()] = content[start:end].strip()
    return sections


def extract_name(intro: str, fallback: str) -> str:
    """从一级标题中提取菜名。"""

    for line in intro.splitlines():
        if line.startswith("# "):
            return re.sub(r"的做法$", "", line[2:].strip()).strip()
    return fallback


def extract_description(intro: str) -> str:
    """提取菜谱简介，忽略一级标题和难度行。"""

    lines = []
    for line in intro.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("# ") or stripped.startswith("预估烹饪难度"):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def extract_estimated_minutes(text: str) -> int | None:
    """从菜谱简介中提取保守的总制作时间。范围值取上限。"""

    normalized = (
        text.replace("～", "-").replace("~", "-").replace("—", "-").replace("－", "-").lower()
    )
    candidates: list[int] = []

    for match in COMBINED_TIME_PATTERN.finditer(normalized):
        candidates.append(
            _to_minutes(float(match.group("hours")), "小时")
            + _to_minutes(float(match.group("minutes")), "分钟")
        )

    for match in TIME_UNIT_PATTERN.finditer(normalized):
        candidates.append(_to_minutes(float(match.group("value")), match.group("unit")))

    for match in CHINESE_NUMBER_PATTERN.finditer(normalized):
        value = _parse_chinese_number(match.group("value"))
        if value is not None:
            candidates.append(_to_minutes(value, match.group("unit")))

    positive = [value for value in candidates if value > 0]
    return max(positive) if positive else None


def extract_estimated_calories(text: str) -> int | None:
    """只提取简介中明确写出的单份热量。"""

    match = re.search(r"(\d+)\s*(?:千卡|大卡|kcal)", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def extract_nutrition_tags(text: str) -> list[str]:
    """提取简介中明确出现的营养标签，不做模型推断。"""

    tags = [
        tag
        for tag in ("减脂", "低脂", "低糖", "低热量", "高蛋白", "高纤维", "高热量")
        if tag in text
    ]
    return list(dict.fromkeys(tags))


def _to_minutes(value: float, unit: str) -> int:
    """把数值和单位转换为分钟。"""

    normalized_unit = unit.lower()
    if normalized_unit in {"天"}:
        return round(value * 24 * 60)
    if normalized_unit in {"小时", "个小时", "h"}:
        return round(value * 60)
    return round(value)


def _parse_chinese_number(text: str) -> float | None:
    """解析常见中文数字。"""

    if text == "半":
        return 0.5
    if "半" in text:
        return None
    if "十" in text:
        parts = text.split("十", maxsplit=1)
        tens = CHINESE_DIGITS.get(parts[0], 1) if parts[0] else 1
        ones = CHINESE_DIGITS.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return float(tens * 10 + ones)
    return float(CHINESE_DIGITS[text]) if text in CHINESE_DIGITS else None


def section_bullets(
    sections: dict[str, str],
    required_keywords: tuple[str, ...],
) -> list[str]:
    """从匹配章节中读取无序列表。"""

    for title, body in sections.items():
        if any(keyword in title for keyword in required_keywords):
            return [
                match.group(1).strip()
                for line in body.splitlines()
                if (match := LIST_ITEM_PATTERN.match(line))
            ]
    return []


def clean_ingredient_name(raw_text: str) -> str:
    """使用共享规则清洗食材名称，并过滤厨具和解析噪声。"""

    return normalize_ingredient_name(raw_text)


def find_quantity_text(ingredient_name: str, quantity_notes: list[str]) -> str | None:
    """尝试从计算章节中找到对应食材的用量说明。"""

    normalized = ingredient_name.replace("（", "(").replace("）", ")")
    for line in quantity_notes:
        if ingredient_name in line or normalized in line:
            return line
    return None


def parse_directory(root: Path, limit: int | None = None) -> list[ParsedRecipe]:
    """解析目录下的全部 Markdown 菜谱。"""

    paths = sorted(
        path for path in root.rglob("*.md") if "template" not in path.relative_to(root).parts
    )
    if limit is not None:
        paths = paths[:limit]
    return [parse_recipe(path, root) for path in paths]


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/HowToCook/dishes"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated/recipes.json"),
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    recipes = parse_directory(args.input, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            [recipe.model_dump() for recipe in recipes],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"已解析 {len(recipes)} 道菜谱：{args.output}")


if __name__ == "__main__":
    main()
