"""菜谱检索文档测试。"""

from pathlib import Path

from app.rag.documents import build_recipe_document
from app.rag.vector_store import safe_truncate

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


def test_build_recipe_document_splits_into_linked_chunks() -> None:
    recipe = parse_directory(FIXTURE_ROOT)[0]
    document = build_recipe_document(recipe)

    chunk_types = [chunk.chunk_type for chunk in document.chunks]
    assert chunk_types == ["overview", "ingredients", "steps"]
    assert all(chunk.parent_id == document.recipe_id for chunk in document.chunks)

    chunk_ids = [chunk.chunk_id for chunk in document.chunks]
    assert len(set(chunk_ids)) == len(chunk_ids)
    assert all(chunk_id.startswith(document.recipe_id) for chunk_id in chunk_ids)

    steps_chunk = next(chunk for chunk in document.chunks if chunk.chunk_type == "steps")
    assert "步骤：" in steps_chunk.text


def test_safe_truncate_limits_field_length() -> None:
    assert safe_truncate("abcdef", 3) == "abc"
    assert safe_truncate(None, 5) == ""
    assert safe_truncate(123, 5) == "123"
