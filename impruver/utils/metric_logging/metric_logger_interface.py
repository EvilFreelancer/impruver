from typing import Mapping, Union
from typing_extensions import Protocol
from numpy import ndarray
from torch import Tensor

Scalar = Union[Tensor, ndarray, int, float]


class MetricLoggerInterface(Protocol):
    def log(self, name: str, data: Scalar, step: int) -> None:
        pass

    def log_config(self, config) -> None:
        pass

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        pass

    def close(self) -> None:
        pass
