import os
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Config:
    """
    Описание одного конкретного набора настроек (yaml-файл).
    """
    name: str
    file_path: str


@dataclass
class Recipe:
    """
    Рецепт запуска (скрипт + список конфигураций).

    :param name: Название рецепта (например, "train_full")
    :param file_path: Путь до исполняемого скрипта (здесь "train_transformers.py")
    :param configs: Список конфигураций (yaml-файлов)
    :param supports_distributed: Флаг, указывающий, поддерживает ли рецепт распределённый запуск
    """
    name: str
    file_path: str
    configs: List[Config]
    supports_distributed: bool


def parse_config_filename(filename: str) -> Optional[Config]:
    """
    Парсит имя файла вида: <модель>_(<размер>_)?<тип тренировки>(_<остальное>)?.yaml
    и возвращает объект Config с «красивым» именем.
    """

    if not filename.endswith(".yaml"):
        return None

    base = os.path.splitext(filename)[0]  # "ruGPT35_13B_lora_cot"
    parts = base.split("_")  # ["ruGPT35", "13B", "lora", "cot"]

    # Первая часть — это модель
    model = parts[0]  # "ruGPT35"
    idx = 1

    # Проверяем, выглядит ли следующая часть как размер (7B, 13B, 127m, и т.д.)
    size = ""
    if len(parts) > 1 and re.match(r"^\d+(\.\d+)?[Bm]$", parts[1], re.IGNORECASE):
        size = parts[1]  # "13B"
        idx += 1

    # Тип тренировки (например, full или lora)
    training_type = parts[idx]
    idx += 1

    # Всё, что осталось, считаем «датасетом»
    dataset = "_".join(parts[idx:]) if idx < len(parts) else ""

    # Формируем удобочитаемое имя конфига
    config_name = f"{training_type}/{model}/"
    if size:
        if not config_name.endswith("/"): config_name += f"_"
        config_name += f"{size}"
    if dataset:
        if not config_name.endswith("/"): config_name += f"_"
        config_name += f"{dataset}"

    file_path = os.path.join("configs", filename)
    return Config(name=config_name, file_path=file_path)


def get_all_recipes(config_dir: str = "configs") -> List[Recipe]:
    lora_configs: List[Config] = []
    full_configs: List[Config] = []

    # Сортируем список файлов, чтобы конфиги шли в алфавитном порядке
    sorted_filenames = sorted(os.listdir(config_dir))

    for filename in sorted_filenames:
        if not filename.endswith(".yaml"):
            continue

        config_obj = parse_config_filename(filename)
        if config_obj is None:
            continue

        # Если в названии файла есть "lora", отправляем в lora_configs
        # Иначе — в full_configs
        if "lora" in filename.lower():
            lora_configs.append(config_obj)
        else:
            full_configs.append(config_obj)

    recipe_full = Recipe(
        name="full_finetune",
        file_path="train_transformers.py",
        configs=full_configs,
        supports_distributed=False,
    )
    recipe_lora = Recipe(
        name="lora_finetune",
        file_path="train_transformers.py",
        configs=lora_configs,
        supports_distributed=False,
    )

    return [recipe_full, recipe_lora]


if __name__ == "__main__":
    recipes = get_all_recipes("../configs")
    for recipe in recipes:
        print(f"\nRecipe: {recipe.name}")
        print(f"  Script: {recipe.file_path}")
        print(f"  Distributed: {recipe.supports_distributed}")
        print("  Configs:")
        for conf in recipe.configs:
            print(f"    - {conf.name}  ->  {conf.file_path}")
    print()
