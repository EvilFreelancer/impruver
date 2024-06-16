import torch


def get_device(device: str | int) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    elif device == "cuda":
        return torch.device("cuda")
    else:
        return torch.device(f"cuda:{device}")
