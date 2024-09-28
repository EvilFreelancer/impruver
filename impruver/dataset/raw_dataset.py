import torch
from torch import LongTensor, Tensor

from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Callable, Mapping, List, Tuple, Any, Dict, Optional

from impruver.data import (
    Message,
    Tokenizer,
    validate_messages,
    apply_chat_template,
    last_message_by_assistant
)


class RawDataset:
    def __init__(
            self,
            *,
            tokenizer: Tokenizer,
            source: str,
            convert_function: Callable,
            max_tokens_count: Optional[int] = None,
            strategy_function: Optional[Callable] = None,
            chat_template: Optional[str] = None,
            **load_dataset_kwargs: Dict[str, Any],
    ):
        self._tokenizer = tokenizer
        self._tokenize = tokenizer
        self._convert_function = convert_function
        self._chat_template = chat_template
        self._strategy_function = strategy_function
        self._data = load_dataset(source, **load_dataset_kwargs)
        self._max_tokens_count = max_tokens_count

        # If max amount of tokens is not set
        if max_tokens_count is None:
            # THen try to get it from tokenizer settings
            self._max_tokens_count = tokenizer.model_max_length

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> None | dict[str, LongTensor | Tensor]:
        # Convert incoming sample to messages list, output should be in format like:
        # [{"role": "<system|user|assistant>", "content": "content"},...]
        messages = self._convert_function(sample)

        # Validata messages
        validate_messages(messages)

        # Apply strategy to message if it's required
        if self._strategy_function is not None:
            messages = self._strategy_function(
                messages=messages,
                tokenizer=self._tokenizer,
                max_tokens_count=self._max_tokens_count,
                chat_template=self._chat_template
            )

        return messages


def raw_dataset(
        *,
        tokenizer: Tokenizer,
        source: str,
        convert_function_name: str = "default",
        chat_template: Optional[str] = None,
        strategy_function_name: str = "default",
        max_tokens_count: Optional[int] = None,
        **load_dataset_kwargs: Dict[str, Any],
) -> RawDataset:
    if convert_function_name == "default":
        convert_function = lambda x: [
            Message.from_dict({'role': 'assistant' if m['role'] == 'bot' else m['role'], 'content': m['content']})
            for m in x["messages"]
        ]
    else:
        raise ValueError(f"convert_function {convert_function_name} is not supported")

    if strategy_function_name == "default":
        strategy_function = last_message_by_assistant
    else:
        raise ValueError(f"strategy_function {strategy_function_name} is not supported")

    return RawDataset(
        tokenizer=tokenizer,
        source=source,
        convert_function=convert_function,
        strategy_function=strategy_function,
        chat_template=chat_template,
        max_tokens_count=max_tokens_count,
        **load_dataset_kwargs
    )
