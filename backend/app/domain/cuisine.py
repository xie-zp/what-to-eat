"""根据菜名、描述和食材推断中国菜系。"""

from __future__ import annotations

CUISINE_KEYWORDS = {
    "川菜": (
        "川菜",
        "麻婆",
        "宫保",
        "鱼香",
        "回锅肉",
        "水煮肉片",
        "水煮牛肉",
        "水煮鱼",
        "口水鸡",
        "干煸",
        "蚂蚁上树",
        "夫妻肺片",
        "辣子鸡",
    ),
    "鲁菜": (
        "鲁菜",
        "九转大肠",
        "葱烧海参",
        "糖醋鲤鱼",
        "油焖大虾",
        "芫爆",
    ),
    "粤菜": (
        "粤菜",
        "广式",
        "白灼",
        "豉汁",
        "柱候",
        "叉烧",
        "烧鹅",
        "煲仔",
        "老火",
        "艇仔粥",
        "肠粉",
        "虾饺",
        "盐焗鸡",
        "姜葱捞鸡",
        "皮蛋瘦肉粥",
    ),
    "淮扬菜": (
        "淮扬菜",
        "扬州炒饭",
        "狮子头",
        "煮干丝",
        "文思豆腐",
        "蟹粉",
    ),
    "闽菜": (
        "闽菜",
        "佛跳墙",
        "荔枝肉",
        "醉排骨",
        "肉燕",
        "沙茶",
        "海蛎煎",
        "鱼丸",
    ),
    "浙菜": (
        "浙菜",
        "西湖醋鱼",
        "东坡肉",
        "龙井虾仁",
        "宋嫂鱼羹",
    ),
    "湘菜": (
        "湘菜",
        "剁椒鱼头",
        "辣椒炒肉",
        "农家一碗香",
        "小炒肉",
    ),
    "徽菜": (
        "徽菜",
        "徽派",
        "臭鳜鱼",
        "毛豆腐",
    ),
}
AQUATIC_MARKERS = ("鱼", "虾", "蟹", "蚝", "鳝", "鲈", "鳜", "贝", "鱿鱼")


def classify_recipe_cuisine(
    name: str,
    description: str = "",
    ingredients: list[str] | None = None,
) -> str:
    """返回菜谱最明确的菜系，无法可靠判断时返回空字符串。

    菜谱描述里常出现“口味近似川菜”“去川菜馆吃过”这类提及，
    因此只根据菜名和食材判断，避免把提及当成归属。
    """

    text = " ".join([name, *(ingredients or [])])
    for cuisine, keywords in CUISINE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return cuisine
    if "清蒸" in text and any(marker in text for marker in AQUATIC_MARKERS):
        return "粤菜"
    return ""
