import os
from dataclasses import dataclass
from typing import List
from pathlib import Path

import impruver

ROOT = Path(impruver.__file__).parent.parent


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
    Объект описывающий рецепт запуска

    :param name: Название рецепта (например, "full_finetune")
    :param file_path: Путь до исполняемого скрипта (здесь "finetune_transformers.py")
    :param configs: Список объектов Config, содержащих конфигурации
    :param supports_distributed: Флаг, указывающий, поддерживает ли рецепт распределённый запуск
    """
    name: str
    file_path: str
    configs: List[Config]
    supports_distributed: bool


def get_all_recipes(config_dir: str = ROOT / "recipes") -> List[Recipe]:
    all_configs: List[Config] = []
    for root, test, filenames in os.walk(config_dir):
        sorted_filenames = sorted(filenames)
        for filename in sorted_filenames:
            if not filename.endswith(".yaml"):
                continue
            file_path = os.path.join(root, filename)
            folder_name = os.path.basename(root)
            file_base = os.path.splitext(filename)[0]
            config_obj = Config(name=folder_name + '/' + file_base, file_path=file_path)
            if config_obj is None:
                continue
            all_configs.append(config_obj)
    # Сортируем все конфигурации в алфавитном порядке по атрибуту file_path
    all_configs.sort(key=lambda x: x.file_path)

    recipe_compose_dataset = Recipe(
        name="compose_dataset",
        file_path="compose_dataset.py",
        configs=[],
        supports_distributed=False,
    )
    recipe_chat = Recipe(
        name="chat",
        file_path="chat_transformers.py",
        configs=[],
        supports_distributed=False,
    )
    recipe_finetune = Recipe(
        name="finetune",
        file_path="finetune_transformers.py",
        configs=all_configs,
        supports_distributed=False,
    )

    return [recipe_finetune, recipe_compose_dataset, recipe_chat]


if __name__ == "__main__":
    import recipes

    recipes_files = get_all_recipes(os.path.join(recipes.__path__[0], "/configs"))
    for recipe in recipes_files:
        print(f"\nRecipe: {recipe.name}")
        print(f"  Script: {recipe.file_path}")
        print(f"  Distributed: {recipe.supports_distributed}")
        print("  Configs:")
        for conf in recipe.configs:
            print(f"    - {conf.name}  ->  {conf.file_path}")
    print()
