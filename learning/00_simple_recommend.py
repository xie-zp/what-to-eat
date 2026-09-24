"""最基础版本：一个函数完成菜谱推荐。"""

RECIPES = [
    {
        "name": "青椒鸡胸肉",
        "minutes": 20,
        "ingredients": ["鸡胸肉", "青椒", "大蒜"],
        "tags": ["低糖", "高蛋白"],
    },
    {
        "name": "西红柿炒鸡蛋",
        "minutes": 15,
        "ingredients": ["西红柿", "鸡蛋"],
        "tags": ["低糖"],
    },
    {
        "name": "宫保鸡丁",
        "minutes": 30,
        "ingredients": ["鸡胸肉", "花生", "黄瓜"],
        "tags": ["高蛋白"],
    },
]


def recommend(goal: str, max_minutes: int, allergies: list[str]) -> str:
    """根据目标、时间和过敏原推荐菜谱。"""

    matched = []

    for recipe in RECIPES:
        if recipe["minutes"] > max_minutes:
            continue

        if any(allergy in recipe["ingredients"] for allergy in allergies):
            continue

        if goal not in recipe["tags"]:
            continue

        matched.append(recipe["name"])

    if not matched:
        return "没有找到符合条件的菜谱。"

    return "推荐：" + "、".join(matched)


if __name__ == "__main__":
    result = recommend(
        goal="低糖",
        max_minutes=30,
        allergies=["花生"],
    )
    print(result)
