import torch
from torch import LongTensor, Tensor

from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Callable, Mapping, List, Tuple, Any, Dict, Optional

from impruver.data import Message, Tokenizer, validate_messages, apply_chat_template


class ChatDataset(Dataset):

    def __init__(
            self,
            *,
            tokenizer: Tokenizer,
            source: str,
            convert_function: Callable,
            max_tokens_count: Optional[int] = None,
            format_function: Optional[Callable] = None,
            strategy_function: Optional[Callable] = None,
            labels_pad_token_id: int = -100,
            **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._convert_function = convert_function
        self._format_function = format_function
        self._strategy_function = strategy_function
        self._data = load_dataset(source, **load_dataset_kwargs)
        self._labels_pad_token_id = labels_pad_token_id
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

    def _prepare_sample(self, sample: Mapping[str, Any]) -> dict[str, LongTensor | Tensor]:
        # Convert incoming sample to messages list, output should be in format like:
        # [{"role": "<system|user|assistant>", "content": "content"},...]
        messages = self._convert_function(sample)

        # Validata messages
        validate_messages(messages)

        # Apply a format of messages
        formated_messages = None
        tokenized_messages = None
        if self._format_function is not None:
            formated_messages = self._format_function(messages)
        if self._strategy_function is not None:
            messages, formated_messages, tokenized_messages = self._strategy_function(messages)

        # Check for edge case situations
        if formated_messages is None:
            formated_messages = apply_chat_template(messages, tokenize=False, tokenizer=self._tokenizer)
        if tokenized_messages is None:
            tokenized_messages = self._tokenizer.encode(formated_messages, return_tensors="pt")

        # Return tokenized chat
        input_ids = torch.LongTensor(tokenized_messages)
        labels = torch.LongTensor([self._labels_pad_token_id for _ in range(len(tokenized_messages))])
        attention_mask = input_ids.new_ones(input_ids.size())
        assert (input_ids.size(0) == labels.size(0) == attention_mask.size(0) <= self._max_tokens_count)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
