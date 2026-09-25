# Agent 可靠性评测

- 场景数量：18
- 通过：18
- 失败：0
- 通过率：100.0%

| 场景 | 结果 | 意图 | 工具 | 失败原因 |
|---|---:|---|---|---|
| inventory-soft | 通过 | recommendation | search_recipes |  |
| inventory-word-order | 通过 | recommendation | search_recipes |  |
| inventory-strict | 通过 | recommendation | search_recipes、search_recipes |  |
| multi-ingredient | 通过 | recommendation | search_recipes |  |
| allergy | 通过 | recommendation | search_recipes |  |
| allergy-dislike | 通过 | recommendation | search_recipes |  |
| pairing | 通过 | pairing | recommend_pairings |  |
| comparison | 通过 | comparison | recommend_pairings |  |
| unknown-recipe | 通过 | recipe_detail | search_recipes |  |
| nutrition | 通过 | nutrition_question | search_recipes、get_nutrition |  |
| nutrition-explicit | 通过 | nutrition_question | search_recipes、get_nutrition |  |
| conflict | 通过 | recommendation | search_recipes、search_recipes |  |
| menu | 通过 | meal_plan | search_recipes、create_meal_plan、create_shopping_list |  |
| irrelevant | 通过 | unsupported |  |  |
| prompt-injection | 通过 | unsupported |  |  |
| follow-up-how-to | 通过 | recipe_detail | search_recipes、get_recipe |  |
| new-topic-does-not-inherit-task-constraints | 通过 | recommendation | search_recipes |  |
| polite-recipe-detail | 通过 | recipe_detail | search_recipes、get_recipe |  |
