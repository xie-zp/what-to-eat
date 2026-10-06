"""上游菜谱解析后的领域模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ParsedIngredient(BaseModel):
    """从菜谱原料章节解析出的食材。"""

    name: str
    raw_text: str
    amount_text: str | None = None


class ParsedRecipe(BaseModel):
    """从 HowToCook Markdown 解析出的标准化菜谱。"""

    recipe_id: str
    name: str
    category: str
    difficulty: int
    estimated_minutes: int | None = None
    description: str
    estimated_calories: int | None = None
    nutrition_tags: list[str] = Field(default_factory=list)
    cuisine_type: str = ""
    ingredients: list[ParsedIngredient] = Field(default_factory=list)
    quantity_notes: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    source_path: str
