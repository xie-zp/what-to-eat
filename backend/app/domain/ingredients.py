"""食材名称清洗和厨具过滤。"""

from __future__ import annotations

import re

EQUIPMENT_MARKERS = {
    "工具",
    "原料",
    "容器",
    "保鲜膜",
    "锡纸",
    "烘焙纸",
    "厨房纸",
    "竹签",
    "牙签",
    "棉线",
    "手套",
    "温度计",
    "电子秤",
    "量杯",
    "打蛋器",
    "榨汁机",
    "豆浆机",
    "微波炉",
    "烤箱",
    "空气炸锅",
    "电饭煲",
    "电炖锅",
    "高压锅",
    "平底锅",
    "炒锅",
    "蒸锅",
    "砂锅",
    "煎锅",
    "汤锅",
    "奶锅",
    "不粘锅",
    "烤盘",
    "模具",
    "案板",
    "砧板",
    "菜刀",
    "刀具",
    "水果刀",
    "剪刀",
    "锅铲",
    "锅盖",
    "蒸屉",
    "笊篱",
    "漏勺",
    "夹子",
    "钳子",
    "秒表",
    "计时器",
    "定时器",
    "抹布",
    "筷子",
    "铲子",
    "勺子",
    "盘子",
    "碗",
    "杯子",
    "高球杯",
    "雪克杯",
    "面粉筛",
}
INVALID_EXACT = {
    "",
    "原料:",
    "原料：",
    "工具:",
    "工具：",
}
SHOPPING_CANONICALS = {
    "蒜": {"蒜", "大蒜", "蒜瓣", "蒜片", "蒜末", "蒜头"},
    "姜": {"姜", "生姜", "姜片", "姜末", "生姜片"},
    "葱": {"葱", "小葱", "大葱", "葱花", "葱段", "葱白"},
    "食用油": {"食用油", "植物油", "菜籽油"},
    "酱油": {"酱油", "生抽", "生抽酱油"},
    "白糖": {"白糖", "白砂糖", "砂糖"},
    "盐": {"盐", "食用盐", "食盐"},
    "鸡蛋": {"鸡蛋", "蛋清", "鸡蛋清"},
    "淀粉": {"淀粉", "土豆淀粉", "生粉"},
}


def clean_ingredient_name(raw_text: str) -> str:
    """移除数量、别名、厨具和解析噪声，返回规范食材名。"""

    name = str(raw_text or "").strip()
    name = re.sub(r"（[^）]*）|\([^)]*\)", "", name).strip()
    name = re.sub(r"^(?:必备|必需|可选)[:：]", "", name).strip()
    if name in INVALID_EXACT:
        return ""
    if any(marker in name for marker in EQUIPMENT_MARKERS):
        return ""

    if ":" in name or "：" in name:
        head = re.split(r"[:：]", name, maxsplit=1)[0].strip()
        if head and head not in INVALID_EXACT:
            name = head
    name = re.sub(
        r"\s*\d+(?:\.\d+)?\s*"
        r"(?:g|kg|ml|l|克|千克|毫升|升|个|颗|只|片|瓣|根|把|勺|汤匙|茶匙|杯|碗)\s*$",
        "",
        name,
        flags=re.IGNORECASE,
    ).strip()
    name = re.sub(
        r"\s*[一二两三四五六七八九十]+"
        r"(?:个|颗|只|片|瓣|根|把|勺|汤匙|茶匙|杯|碗)\s*$",
        "",
        name,
    ).strip()
    if len(name) > 30:
        return ""
    return name.strip(" ，,。；;")


def clean_ingredient_names(values: list[str]) -> list[str]:
    """清洗并去重食材列表。"""

    return list(
        dict.fromkeys(cleaned for value in values if (cleaned := clean_ingredient_name(value)))
    )


def canonical_shopping_name(name: str) -> str:
    """把同一食材的常见切配名合并为购物清单规范名。"""

    cleaned = clean_ingredient_name(name)
    if not cleaned:
        return ""
    for canonical, aliases in SHOPPING_CANONICALS.items():
        if cleaned in aliases:
            return canonical
    return cleaned
