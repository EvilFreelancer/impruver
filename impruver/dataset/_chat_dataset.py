import torch
from torch import LongTensor, Tensor

from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Callable, Mapping, List, Tuple, Any, Dict, Optional

from impruver.data import Message, Tokenizer, validate_messages, apply_chat_template, last_message_by_assistant


class ChatDataset(Dataset):

    def __init__(
            self,
            *,
            tokenizer: Tokenizer,
            source: str,
            convert_function: Callable,
            max_tokens_count: Optional[int] = None,
            strategy_function: Optional[Callable] = None,
            chat_template: Optional[str] = None,
            labels_pad_token_id: int = -100,
            **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._convert_function = convert_function
        self._chat_template = chat_template
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

    def _prepare_sample(self, sample: Mapping[str, Any]) -> None | dict[str, LongTensor | Tensor]:
        # Convert incoming sample to messages list, output should be in format like:
        # [{"role": "<system|user|assistant>", "content": "content"},...]
        messages = self._convert_function(sample)

        # Validata messages
        validate_messages(messages)

        # Defaults
        formated_messages = None
        tokenized_messages = None

        # Apply strategy if it's present
        if self._strategy_function is not None:
            messages, formated_messages, tokenized_messages = self._strategy_function(
                tokenizer=self._tokenizer,
                messages=messages,
                max_tokens_count=self._max_tokens_count,
                chat_template=self._chat_template
            )

        # If there is no strategies, then format messages
        if formated_messages is None:
            # Apply chat format from tokenizers template
            if hasattr(self._tokenizer, 'apply_chat_template'):
                # On modern tokenizers we may use chat_template
                formated_messages = self._tokenizer.apply_chat_template(
                    messages,
                    chat_template=self._chat_template,  # Use default_chat_template if None
                    tokenize=False
                )
            else:
                # On old tokenizers we will use a custom apply_chat_template
                formated_messages = apply_chat_template(
                    messages,
                    chat_template=self._chat_template,  # Use DEFAULT_CHAT_TEMPLATE if None
                    tokenize=False,
                    tokenizer=self._tokenizer
                )

        # If there is no strategies, then tokenize messages
        if tokenized_messages is None:
            tokenized_messages = self._tokenizer.encode(formated_messages, return_tensors="pt")

        # Calculate tokens count
        input_ids = torch.LongTensor(tokenized_messages[0])
        if input_ids.size(0) > self._max_tokens_count:
            return None

        # Return tokenized chat
        # input_ids = torch.LongTensor(tokenized_messages)
        labels = torch.LongTensor([self._labels_pad_token_id for _ in range(len(messages))])
        attention_mask = input_ids.new_ones(input_ids.size())

        assert (input_ids.size(0) == attention_mask.size(0))
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def chat_dataset(
        *,
        tokenizer: Tokenizer,
        source: str,
        convert_function_name: str = "default",
        chat_format_name: str = "default",
        strategy_function_name: str = "default",
        max_tokens_count: Optional[int] = None,
        **load_dataset_kwargs: Dict[str, Any],
) -> ChatDataset:
    if convert_function_name == "default":
        convert_function = lambda x: [
            Message.from_dict({
                'role': 'assistant' if m['role'] == 'bot' else m['role'],
                'content': m['content']
            }) for m in x["messages"]]

    else:
        raise ValueError(f"convert_function {convert_function_name} is not supported")

    if chat_format_name == "default":
        chat_format = None
    else:
        raise ValueError(f"chat_format {chat_format_name} is not supported")

    if strategy_function_name == "default":
        strategy_function = last_message_by_assistant
    else:
        raise ValueError(f"strategy_function {strategy_function_name} is not supported")

    return ChatDataset(
        tokenizer=tokenizer,
        source=source,
        convert_function=convert_function,
        # strategy_function=strategy_function,
        chat_format=chat_format,
        max_tokens_count=max_tokens_count,
        **load_dataset_kwargs
    )
