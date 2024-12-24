import argparse

# from impruver._cli.cp import Copy
# from impruver._cli.download import Download
from impruver._cli.ls import List
# from impruver._cli.run import Run


class ImpruverCLIParser:
    """Holds all information related to running the CLI"""

    def __init__(self):
        # Initialize the top-level parser
        self._parser = argparse.ArgumentParser(
            prog="tune",
            description="Welcome to the impruver CLI!",
            add_help=True,
        )
        # Default command is to print help
        self._parser.set_defaults(func=lambda args: self._parser.print_help())

        # Add subcommands
        subparsers = self._parser.add_subparsers(title="subcommands")
        # Download.create(subparsers)
        List.create(subparsers)
        # Copy.create(subparsers)
        # Run.create(subparsers)

    def parse_args(self) -> argparse.Namespace:
        """Parse CLI arguments"""
        return self._parser.parse_args()

    def run(self, args: argparse.Namespace) -> None:
        """Execute CLI"""
        args.func(args)


def main():
    parser = ImpruverCLIParser()
    args = parser.parse_args()
    parser.run(args)


if __name__ == "__main__":
    main()
