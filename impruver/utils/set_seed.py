import logging
import os
import random
from typing import Optional, Union
import numpy as np
import torch

from .get_logger import get_logger

_log: logging.Logger = get_logger()


def set_seed(
        seed: Optional[int] = None,
        debug_mode: Optional[Union[str, int]] = None
) -> int:
    if seed is None:
        seed = random.randint(1, 2 ** 32 - 1)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if debug_mode is not None:
        _log.debug(f"Setting deterministic debug mode to {debug_mode}")
        torch.set_deterministic_debug_mode(debug_mode)
        deterministic_debug_mode = torch.get_deterministic_debug_mode()
        if deterministic_debug_mode == 0:
            _log.debug("Disabling cuDNN deterministic mode")
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            _log.debug("Enabling cuDNN deterministic mode")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return seed
