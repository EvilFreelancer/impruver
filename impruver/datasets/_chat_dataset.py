from datasets import Dataset, load_dataset
from typing import Callable, Mapping, List, Tuple, Any, Dict

from impruver.data import Message, Tokenizer, validate_messages


class ChatDataset(Dataset):

    def __init__(
            self,
            *,
            tokenizer: Tokenizer,
            source: str,
            max_tokens_count: int,
            convert_function: Callable,
            format_function: Callable,
            strategy_function: Callable,
            **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._max_tokens_count = max_tokens_count
        self._convert_function = convert_function
        self._format_function = format_function
        self._strategy_function = strategy_function
        self._data = load_dataset(source, **load_dataset_kwargs)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _get_tokens(self, messages):
        """Convert messages to tokens"""
        tokens = self._tokenizer.apply_chat_template(
            messages,
            add_special_tokens=False,
            tokenize=True,
            add_generation_prompt=False,
        )
        if tokens[0] == self._tokenizer.bos_token_id:
            tokens = tokens[1:]
        return tokens

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
        # Convert sample to messages list
        messages = self._convert_function(sample)

        # Validata messages, they should be in OpenAI format
        validate_messages(messages)

        # Apply a format of messages
        if self._format_function is not None:
            messages = self._format_function(messages)

        if self._strategy_function is not None:
            messages, formatted_messages, tokenized_messages = self._strategy_function(messages)

        # Return tokenized chat
        return self._tokenizer.tokenize_messages(
            messages, _max_tokens_count=self._max_tokens_count
        )
