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


def test_description_mention_does_not_decide_cuisine() -> None:
    # 描述里的“川菜馆”“近似川菜”只是提及，不能当作菜系归属。
    assert (
        classify_recipe_cuisine(
            "蒜蓉空心菜",
            "去学校附近的川菜馆吃过蒜蓉空心菜，之后就一直很喜欢吃。",
            ["空心菜", "蒜", "盐"],
        )
        == ""
    )
    assert (
        classify_recipe_cuisine(
            "黔式腊肠娃娃菜",
            "口味近似于川菜、湘菜，却是西南菜系中鲜见的不辣菜式。",
            ["腊肠", "娃娃菜"],
        )
        == ""
    )
    # 菜名本身是明确信号时仍然要识别出来。
    assert classify_recipe_cuisine("蚂蚁上树", "一道经典的川菜", ["粉丝", "肉末"]) == "川菜"
