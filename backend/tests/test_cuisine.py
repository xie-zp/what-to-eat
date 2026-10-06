"""菜系分类测试。"""

from app.domain.cuisine import classify_recipe_cuisine


def test_classifies_representative_chinese_cuisines() -> None:
    examples = {
        "麻婆豆腐": "川菜",
        "葱烧海参": "鲁菜",
        "白灼虾": "粤菜",
        "清蒸鲈鱼": "粤菜",
        "扬州炒饭": "淮扬菜",
        "佛跳墙": "闽菜",
        "西湖醋鱼": "浙菜",
        "辣椒炒肉": "湘菜",
        "臭鳜鱼": "徽菜",
    }

    for name, expected in examples.items():
        assert classify_recipe_cuisine(name) == expected
