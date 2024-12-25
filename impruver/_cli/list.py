import argparse
import textwrap

from impruver._cli.subcommand import Subcommand
from impruver._recipe_registry import get_all_recipes


class List(Subcommand):
    """Holds all the logic for the `tune ls` subcommand."""

    NULL_VALUE = "<любой>"

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self._parser = subparsers.add_parser(
            "ls",
            prog="impruver ls",
            help="List all built-in recipes and configs",
            description="List all built-in recipes and configs",
            epilog=textwrap.dedent(
                """\
            examples:
                $ impruver ls
                RECIPE               CONFIG
                finetune             ruGPT-3.5/13B_lora_saiga2
                                     rugpt3large/760M_full_fc
                                     rugpt3large/760M_full_saiga2
                                     zephyr/7B_lora_python-ru
                compose_dataset      <любой>
                chat                 <любой>
                ...

            To run one of these recipes:
                $ impruver run finetune --config ruGPT-3.5/13B_lora_saiga2
            """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._parser.set_defaults(func=self._ls_cmd)

    def _ls_cmd(self, args: argparse.Namespace) -> None:
        """List all available recipes and configs."""
        # Print table header
        header = f"{'RECIPE':<20} {'CONFIG':<40}"
        print(header)

        # Print recipe/config pairs
        for recipe in get_all_recipes():
            # If there are no configs for a recipe, print a blank config
            recipe_str = recipe.name
            if len(recipe.configs) == 0:
                row = f"{recipe_str:<20} {self.NULL_VALUE:<40}"
                print(row)
            for i, config in enumerate(recipe.configs):
                # If there are multiple configs for a single recipe, omit the recipe name
                # on latter configs
                if i > 0:
                    recipe_str = ""
                row = f"{recipe_str:<20} {config.name:<40}"
                print(row)
