"""食材清洗和厨具过滤测试。"""

from app.domain.ingredients import (
    canonical_shopping_name,
    clean_ingredient_name,
    clean_ingredient_names,
)
from app.state import RecipeEvidence
from app.tools import MealPlanTool, ShoppingListTool


def test_clean_ingredient_names_filters_equipment_and_noise() -> None:
    assert clean_ingredient_name("空气炸锅") == ""
    assert clean_ingredient_name("锡纸或保鲜膜") == ""
    assert clean_ingredient_name("白砂糖：必须水油炒") == "白砂糖"
    assert clean_ingredient_name("里脊肉 200g") == "里脊肉"
    assert canonical_shopping_name("蒜片") == "蒜"
    assert canonical_shopping_name("生姜片") == "姜"
    assert canonical_shopping_name("食用盐") == "盐"
    assert canonical_shopping_name("鸡蛋清") == "鸡蛋"
    assert canonical_shopping_name("土豆淀粉") == "淀粉"
    assert clean_ingredient_names(["花菜", "筷子", "盐", "花菜"]) == ["花菜", "盐"]
    assert clean_ingredient_name("秒表") == ""
    assert clean_ingredient_name("菜刀一个") == ""


def test_recipe_evidence_filters_equipment() -> None:
    recipe = RecipeEvidence(
        recipe_id="r1",
        name="测试菜",
        ingredients=["花菜", "盘子", "盐"],
    )

    assert recipe.ingredients == ["花菜", "盐"]


def test_meal_plan_and_shopping_list_ignore_equipment() -> None:
    plan = MealPlanTool().run(
        {
            "days": 1,
            "meals_per_day": 1,
            "recipes": [
                {
                    "recipe_id": "r1",
                    "name": "测试菜",
                    "ingredients": ["花菜 200g", "炒锅", "盐 5g"],
                }
            ],
        }
    )
    shopping = ShoppingListTool().run(
        {
            "required_ingredients": ["花菜", "铲子", "盐"],
            "available_ingredients": [],
        }
    )

    assert plan.ok is True
    assert plan.data["meal_plan"]["required_ingredients"] == ["花菜", "盐"]
    assert [item["name"] for item in shopping.data["shopping_list"]] == ["花菜", "盐"]
