import os
import recipes
import argparse
import textwrap

from impruver._cli.subcommand import Subcommand
from impruver._recipe_registry import get_all_recipes


class List(Subcommand):
    """Holds all the logic for the `tune ls` subcommand."""

    NULL_VALUE = "<>"

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self._parser = subparsers.add_parser(
            "ls",
            prog="impruver ls",
            help="Отобразить список всех рецептов и конфигов из стандартной поставки",
            description="Список всех рецептов и конфигов",
            epilog=textwrap.dedent(
                """\
            Например:
                $ impruver ls
                RECIPE                                   CONFIG                                  
                lora_finetune                            rugpt3large/760M_lora_fc
                                                         ruGPT-3.5/13B_lora_saiga2
                full_finetune                            rugpt3large/760M_full_saiga2           
                                                         rugpt3small/125M_full_toxicator
                                                         rugpt3medium/457M_full_fc
                                                         nanoGPT/30M_full_alpaca
                ...

            Чтобы запустить рецепт, используйте:
                $ impruver run lora_finetune --config ruGPT-3.5/13B_lora_saiga2
            """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._parser.set_defaults(func=self._ls_cmd)

    def _ls_cmd(self, args: argparse.Namespace) -> None:
        """List all available recipes and configs."""
        # Print table header
        header = f"{'RECIPE':<40} {'CONFIG':<40}"
        print(header)

        # Print recipe/config pairs
        recipies_path = os.path.join(recipes.__path__[0])
        # print(recipies_path)
        # exit()
        for recipe in get_all_recipes(recipies_path):
            # If there are no configs for a recipe, print a blank config
            recipe_str = recipe.name
            if len(recipe.configs) == 0:
                row = f"{recipe_str:<40} {self.NULL_VALUE:<40}"
                print(row)
            for i, config in enumerate(recipe.configs):
                # If there are multiple configs for a single recipe, omit the recipe name
                # on latter configs
                if i > 0:
                    recipe_str = ""
                row = f"{recipe_str:<40} {config.name:<40}"
                print(row)
