"""第一阶段 Agent 使用的内存菜谱库。"""

from __future__ import annotations

from app.state import RecipeEvidence

RECIPES: list[RecipeEvidence] = [
    RecipeEvidence(
        recipe_id="r001",
        name="青椒鸡胸肉",
        ingredients=["鸡胸肉", "青椒", "大蒜", "食用油", "生抽"],
        steps=[
            "鸡胸肉切片并加入生抽腌制。",
            "青椒切块，大蒜切末。",
            "热锅下油，炒熟鸡胸肉后加入青椒翻炒。",
        ],
        nutrition_tags=["高蛋白", "低脂", "低糖"],
        category="荤菜",
        difficulty=2,
        estimated_calories=320,
        prep_minutes=8,
        cook_minutes=12,
        source="memory:r001",
        score=0.95,
    ),
    RecipeEvidence(
        recipe_id="r002",
        name="西红柿炒鸡蛋",
        ingredients=["西红柿", "鸡蛋", "食用油", "盐"],
        steps=[
            "西红柿切块，鸡蛋打散。",
            "热锅下油，将鸡蛋炒至凝固后盛出。",
            "炒软西红柿，加入鸡蛋和盐翻匀。",
        ],
        nutrition_tags=["高蛋白", "低糖", "家常"],
        category="素菜",
        difficulty=2,
        prep_minutes=5,
        cook_minutes=10,
        source="memory:r002",
        score=0.91,
    ),
    RecipeEvidence(
        recipe_id="r003",
        name="宫保鸡丁",
        ingredients=["鸡胸肉", "花生", "干辣椒", "黄瓜", "白糖"],
        steps=[
            "鸡胸肉切丁腌制，黄瓜切丁。",
            "炒香干辣椒和花生。",
            "加入鸡丁和调味汁翻炒。",
        ],
        nutrition_tags=["高蛋白", "高热量"],
        category="荤菜",
        difficulty=3,
        prep_minutes=15,
        cook_minutes=15,
        source="memory:r003",
        score=0.88,
    ),
    RecipeEvidence(
        recipe_id="r004",
        name="蒜蓉空心菜",
        ingredients=["空心菜", "大蒜", "食用油", "盐"],
        steps=[
            "空心菜洗净切段，大蒜切末。",
            "热锅下油爆香蒜末。",
            "加入空心菜快速翻炒并调味。",
        ],
        nutrition_tags=["低脂", "低糖", "高纤维"],
        category="素菜",
        difficulty=1,
        prep_minutes=4,
        cook_minutes=4,
        source="memory:r004",
        score=0.86,
    ),
    RecipeEvidence(
        recipe_id="r005",
        name="咖喱炒蟹",
        ingredients=["青蟹", "咖喱块", "洋葱", "椰浆", "鸡蛋"],
        steps=[
            "青蟹处理干净，切口蘸生粉。",
            "煎封蟹肉和蟹盖。",
            "炒香洋葱和咖喱块，加入蟹和椰浆焖煮。",
        ],
        nutrition_tags=["高蛋白", "高热量"],
        category="水产",
        difficulty=4,
        prep_minutes=10,
        cook_minutes=8,
        source="memory:r005",
        score=0.82,
    ),
]
