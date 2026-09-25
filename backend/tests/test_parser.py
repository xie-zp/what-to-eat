"""HowToCook Markdown 解析器测试。"""

from pathlib import Path

from scripts.parse_howtocook import (
    extract_estimated_calories,
    extract_estimated_minutes,
    extract_nutrition_tags,
    parse_directory,
    parse_recipe,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "howtocook"


def test_parse_recipe_extracts_core_sections() -> None:
    path = FIXTURE_ROOT / "aquatic" / "咖喱炒蟹.md"
    recipe = parse_recipe(path, FIXTURE_ROOT)

    assert recipe.name == "咖喱炒蟹"
    assert recipe.category == "水产"
    assert recipe.difficulty == 4
    assert [item.name for item in recipe.ingredients] == [
        "青蟹",
        "咖喱块",
        "洋葱",
        "椰浆",
        "鸡蛋",
    ]
    assert len(recipe.steps) == 3
    assert recipe.estimated_minutes is None
    assert recipe.source_path == "aquatic/咖喱炒蟹.md"


def test_parse_directory_supports_limit() -> None:
    recipes = parse_directory(FIXTURE_ROOT, limit=1)
    assert len(recipes) == 1
    assert recipes[0].name == "咖喱炒蟹"


def test_extract_estimated_minutes_uses_upper_bound() -> None:
    assert extract_estimated_minutes("一般需要 15-20 分钟即可完成。") == 20
    assert extract_estimated_minutes("预计制作时间 1～1.5h。") == 90
    assert extract_estimated_minutes("第一次做的话总耗时 8 个小时 15 分钟。") == 495
    assert extract_estimated_minutes("预计制作时长 三小时。") == 180
    assert extract_estimated_minutes("制作方法简单，只是有些耗时。") is None


def test_extract_explicit_nutrition_only() -> None:
    text = "一份 298 千卡，低脂高蛋白，适合减脂。"

    assert extract_estimated_calories(text) == 298
    assert extract_nutrition_tags(text) == ["减脂", "低脂", "高蛋白"]
    assert extract_estimated_calories("没有标注热量") is None
    assert extract_nutrition_tags("味道很好") == []
