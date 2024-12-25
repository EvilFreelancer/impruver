import argparse
import shutil
import textwrap
from pathlib import Path

import impruver
from impruver._cli.subcommand import Subcommand
from impruver._recipe_registry import get_all_recipes

ROOT = Path(impruver.__file__).parent.parent


class Copy(Subcommand):
    """Содержит всю логику для подкоманды `tune cp`."""

    def __init__(self, subparsers):
        super().__init__()
        self._parser = subparsers.add_parser(
            "cp",
            prog="impruver cp",
            usage="impruver cp <recipe|config> <destination> [OPTIONS]",
            help="Copy a built-in recipe or config to a local path.",
            description="Copy a built-in recipe or config to a local path.",
            epilog=textwrap.dedent(
                """\
            examples:
                $ impruver cp finetune .
                Copied file to ./finetune_transformers.py

                $ impruver cp ruGPT-3.5/13B_lora_saiga2 ./new_dir/my_custom_lora.yaml --make-parents
                Copied file to ./new_dir/my_custom_lora.yaml

            Need to see all possible recipes/configs to copy? Try running `impruver ls`.
            """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self._parser.set_defaults(func=self._cp_cmd)

    def _add_arguments(self) -> None:
        self._parser.add_argument(
            "file", type=str,
            help="Recipe/config to copy. For a list of all possible options, run `impruver ls`",
        )
        self._parser.add_argument(
            "destination", type=Path,
            help="Location to copy the file to",
        )
        self._parser.add_argument(
            "-n", "--no-clobber", action="store_true", default=False,
            help="Do not overwrite destination if it already exists",

        )
        self._parser.add_argument(
            "--make-parents", action="store_true", default=False,
            help="Create parent directories for destination if they do not exist. "
                 "If not set to True, will error if parent directories do not exist",
        )

    def _cp_cmd(self, args: argparse.Namespace):
        destination: Path = args.destination
        src = None

        # Iterate through all recipes and configs
        for recipe in get_all_recipes():
            if recipe.name == args.file:
                src = ROOT / "recipes" / recipe.file_path
                proper_suffix = ".py"
                break
            for config in recipe.configs:
                if config.name == args.file:
                    src = ROOT / "recipes" / "configs" / config.file_path
                    proper_suffix = ".yaml"
                    break

        # Fail if no file exists
        if src is None:
            self._parser.error(
                f"Invalid file name: {args.file}. Try `impruver ls` to see all available files to copy."
            )

        # Attach proper suffix if needed
        if destination.name != "" and destination.suffix != proper_suffix:
            destination = destination.with_suffix(proper_suffix)

        # Copy file
        try:
            if args.no_clobber and destination.exists():
                print(
                    f"File already exists at {destination.absolute()}, not overwriting."
                )
            else:
                if args.make_parents:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                output = shutil.copy(src, destination)
                print(f"Copied file to {output}")
        except FileNotFoundError:
            self._parser.error(
                f"Cannot create regular file: '{destination}'. No such file or directory. "
                "If the specified destination's parent directory does not exist and you would "
                "like to create it on-the-fly, use the --make-parents flag."
            )
