from typing import Dict, Optional

import torch

PRECISION_STR_TO_DTYPE: Dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}


def verify_bf16_support() -> bool:
    return (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            and torch.distributed.is_nccl_available()
            and torch.cuda.nccl.version() >= (2, 10)
    )


def get_dtype(dtype: Optional[str] = None, device: Optional[torch.device] = None) -> torch.dtype:
    # None defaults to float32
    if dtype is None:
        return torch.float32

    # Convert to torch.dtype
    torch_dtype = PRECISION_STR_TO_DTYPE.get(dtype, dtype)

    # dtype must be one of the supported precisions
    if torch_dtype not in PRECISION_STR_TO_DTYPE.values():
        raise ValueError(
            f"Dtype {torch_dtype} must be one of {', '.join(list(PRECISION_STR_TO_DTYPE.keys()))} for finetuning."
        )

    if torch_dtype == torch.bfloat16 and device != torch.device("cpu") and not verify_bf16_support():
        raise RuntimeError(
            "bf16 precision was requested but not available on this hardware. Please use fp32 precision instead."
        )

    return torch_dtype
