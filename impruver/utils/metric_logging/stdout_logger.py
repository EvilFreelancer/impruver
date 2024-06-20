import sys
from typing import Mapping, Union
from numpy import ndarray
from torch import Tensor

from ._metric_logger_interface import MetricLoggerInterface

Scalar = Union[Tensor, ndarray, int, float]


class StdoutLogger(MetricLoggerInterface):
    """Logger to standard output."""

    def log(self, name: str, data: Scalar, step: int) -> None:
        print(f"Step {step} | {name}:{data}")

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        print(f"Step {step} | ", end="")
        for name, data in payload.items():
            print(f"{name}:{data} ", end="")
        print("\n", end="")

    def __del__(self) -> None:
        sys.stdout.flush()

    def close(self) -> None:
        sys.stdout.flush()
