import argparse
import logging
import fire

from .utils import get_logger

_log: logging.Logger = get_logger()


def main():
    parser = argparse.ArgumentParser(description='Convert a .csv file to a .json file')
    parser.add_argument('input_file', help='input file')
    parser.add_argument('output_file', help='output file')
    args = parser.parse_args()

    _log.info(f'Input arguments: {args}')


if __name__ == "__main__":
    fire.Fire(main)
