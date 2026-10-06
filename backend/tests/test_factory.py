"""工具注册表的数据源模式测试。"""

from __future__ import annotations

import json

from app.config import Settings
from app.domain.recipe import ParsedIngredient, ParsedRecipe
from app.tools import factory


def test_file_search_registry_uses_full_recipe_file(monkeypatch, tmp_path) -> None:
    recipe_path = tmp_path / "recipes.json"
    recipe = ParsedRecipe(
        recipe_id="file-001",
        name="西红柿炒鸡蛋",
        category="素菜",
        difficulty=1,
        estimated_minutes=15,
        description="家常菜",
        ingredients=[
            ParsedIngredient(name="西红柿", raw_text="西红柿 2 个"),
            ParsedIngredient(name="鸡蛋", raw_text="鸡蛋 2 个"),
        ],
        steps=["炒熟"],
        source_path="test/西红柿炒鸡蛋.md",
    )
    other = ParsedRecipe(
        recipe_id="file-002",
        name="宫保鸡丁",
        category="荤菜",
        difficulty=2,
        estimated_minutes=30,
        description="川菜",
        ingredients=[
            ParsedIngredient(name="鸡胸肉", raw_text="鸡胸肉 200 克"),
            ParsedIngredient(name="花生", raw_text="花生 50 克"),
        ],
        steps=["炒熟"],
        source_path="test/宫保鸡丁.md",
    )
    recipe_path.write_text(
        json.dumps(
            [recipe.model_dump(mode="json"), other.model_dump(mode="json")],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        factory,
        "get_settings",
        lambda: Settings(
            use_file_search=True,
            recipe_data_path=str(recipe_path),
        ),
    )

    registry = factory.create_registry()
    search_tool = registry.get("search_recipes")

    assert {document.name for document in search_tool.searcher.documents} == {
        "西红柿炒鸡蛋",
        "宫保鸡丁",
    }
    assert "get_recipe" in registry.names()
    assert "get_nutrition" in registry.names()
