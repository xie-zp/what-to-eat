"""菜谱检索文档测试。"""

from pathlib import Path

from app.rag.documents import build_recipe_document

from scripts.parse_howtocook import parse_directory

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "howtocook"


def test_build_recipe_document_contains_searchable_fields() -> None:
    recipe = parse_directory(FIXTURE_ROOT)[0]
    document = build_recipe_document(recipe)

    assert document.recipe_id == recipe.recipe_id
    assert "菜名：咖喱炒蟹" in document.text
    assert "分类：水产" in document.text
    assert "食材：青蟹、咖喱块" in document.text
    assert "步骤：" in document.text
