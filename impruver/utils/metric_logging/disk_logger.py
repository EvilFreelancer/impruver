import time
from typing import Mapping, Optional, Union
from pathlib import Path
from numpy import ndarray
from torch import Tensor

from ._metric_logger_interface import MetricLoggerInterface

Scalar = Union[Tensor, ndarray, int, float]


class DiskLogger(MetricLoggerInterface):
    """Disk logger that writes to a file"""

    def __init__(self, log_dir: str, filename: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if not filename:
            unix_timestamp = int(time.time())
            filename = f"log_{unix_timestamp}.txt"
        self._file_name = self.log_dir / filename
        self._file = open(self._file_name, "a")
        print(f"Writing logs to {self._file_name}")

    def path_to_log_file(self) -> Path:
        return self._file_name

    def log(self, name: str, data: Scalar, step: int) -> None:
        self._file.write(f"Step {step} | {name}:{data}\n")

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        self._file.write(f"Step {step} | ")
        for name, data in payload.items():
            self._file.write(f"{name}:{data} ")
        self._file.write("\n")

    def __del__(self) -> None:
        self._file.close()

    def close(self) -> None:
        self._file.close()
