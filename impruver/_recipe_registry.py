import os
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


def get_all_recipes(config_dir: str = "configs") -> List[Recipe]:
    lora_configs: List[Config] = []
    full_configs: List[Config] = []

    for root, _, filenames in os.walk(config_dir):
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
            if "lora" in filename.lower():
                lora_configs.append(config_obj)
            else:
                full_configs.append(config_obj)

    recipe_lora_finetune = Recipe(
        name="lora_finetune",
        file_path="finetune_transformers.py",
        configs=lora_configs,
        supports_distributed=False,
    )
    recipe_full_finetune = Recipe(
        name="full_finetune",
        file_path="finetune_transformers.py",
        configs=full_configs,
        supports_distributed=False,
    )

    return [recipe_lora_finetune, recipe_full_finetune]


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
