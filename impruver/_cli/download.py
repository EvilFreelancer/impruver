import argparse
import json
import os
import textwrap
import traceback
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from impruver._cli.subcommand import Subcommand


class Download(Subcommand):
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self._parser = subparsers.add_parser(
            "download",
            prog="impruver download",
            usage="impruver download <repo-id> [OPTIONS]",
            help="Download a model or a dataset from the Hugging Face Hub.",
            description="Download a model or a dataset from the Hugging Face Hub.",
            epilog=textwrap.dedent(
                """
                examples:
                    # Download a model from the HuggingFace Hub with a HuggingFace API token
                    $ tune download ai-forever/ruGPT-3.5-13B --hf-token <HF_TOKEN>

                    # Download a dataset from the HuggingFace Hub and save it to ./my_datasets/my_dataset
                    $ tune download IlyaGusev/ru_turbo_alpaca --output-dir ./my_datasets/my_dataset --repo-type=dataset
                """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self._parser.set_defaults(func=self._download_cmd)

    def _add_arguments(self) -> None:
        self._parser.add_argument(
            "repo_id", type=str,
            help="Name of the repository on Hugging Face Hub (ID of model or dataset).",
        )
        self._parser.add_argument(
            "--repo-type", type=str, required=False, default="model", choices=["dataset", "model"],
            help="Repository type, can be 'dataset' or 'model'. Defaults to 'model'.",
        )
        self._parser.add_argument(
            "--output-dir", type=Path, required=False, default=None,
            help="Directory in which to save the model. Defaults to `./<repo_id>`.",
        )
        self._parser.add_argument(
            "--hf-token", type=str, required=False, default=os.getenv("HF_TOKEN", None),
            help="Hugging Face API token. Required for private repositories.",
        )
        self._parser.add_argument(
            "--ignore-patterns", type=str, required=False,
            help="If provided, files matching any of the patterns are not downloaded. Example: '*.safetensors'.",
        )

    def _download_cmd(self, args: argparse.Namespace) -> None:

        # Default output_dir is `./<model_name>`
        output_dir = args.output_dir
        if output_dir is None:
            repo_name = args.repo_id.split("/")[-1]
            output_dir = Path(".") / repo_name

        try:
            true_output_dir = snapshot_download(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                local_dir=output_dir,
                ignore_patterns=args.ignore_patterns,
                token=args.hf_token,
            )
        except GatedRepoError:
            if args.hf_token:
                self._parser.error(
                    "It looks like you are trying to access a gated repository. Please ensure you "
                    "have access to the repository."
                )
            else:
                self._parser.error(
                    "It looks like you are trying to access a gated repository. Please ensure you "
                    "have access to the repository and have provided the proper Hugging Face API token "
                    "using the option `--hf-token` or by running `huggingface-cli login`."
                    "You can find your token by visiting https://huggingface.co/settings/tokens"
                )
        except RepositoryNotFoundError:
            self._parser.error(
                f"Repository '{args.repo_id}' not found on the HuggingFace Hub."
            )
        except Exception as e:
            tb = traceback.format_exc()
            self._parser.error(
                f"Failed to download {args.repo_id} with error: '{e}' and traceback: {tb}"
            )

        # Save repo metadata
        metadata_path = output_dir / "repo_metadata.json"
        with open(metadata_path, "w") as metadata_file:
            json.dump({"repo_id": args.repo_id}, metadata_file, indent=4)

        print(
            "Successfully downloaded model repo and wrote to the following locations:",
            *list(Path(true_output_dir).iterdir()),
            sep="\n",
        )
