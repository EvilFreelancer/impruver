import random
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Callable, Optional, Any
from tqdm import tqdm

from impruver.data import apply_chat_template


class ChatDataset(Dataset):
    def __init__(
            self,
            original_records: List[Dict],
            tokenizer,
            max_tokens_count: int,
            sample_rate: float = 1.0,
            only_target_loss: bool = True,
            add_global_bos: bool = True,
            add_global_eos: bool = True,
            labels_pad_token_id: int = -100,
            convert_function: Optional[Callable[[Dict], List[Dict]]] = None,
            strategy_function: Optional[Callable[[List[Dict]], List[Dict]]] = None,
            chat_template: Optional[str] = None,
    ):
        self.original_records = original_records
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.max_tokens_count = max_tokens_count
        self.only_target_loss = only_target_loss
        self.labels_pad_token_id = labels_pad_token_id
        self.add_global_bos = add_global_bos
        self.add_global_eos = add_global_eos
        self.convert_function = convert_function
        self.strategy_function = strategy_function
        self.chat_template = chat_template
        self.is_printed = False

        self.records = []
        for record in tqdm(original_records):
            if random.random() > self.sample_rate:
                continue
            tensors = self.convert_record(record)
            if tensors is None:
                continue
            self.records.append(tensors)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def get_tokens(self, messages):
        if hasattr(self.tokenizer, 'apply_chat_template'):
            tokens = self.tokenizer.apply_chat_template(
                messages,
                chat_template=self.chat_template,
                add_special_tokens=False,
                tokenize=True,
                add_generation_prompt=False,
            )
        else:
            # Assume apply_chat_template is a function you have defined
            tokens = apply_chat_template(
                messages,
                chat_template=self.chat_template,
                tokenize=True,
                add_generation_prompt=False,
                tokenizer=self.tokenizer,
            )
        if tokens[0] == self.tokenizer.bos_token_id:
            tokens = tokens[1:]
        return tokens

    def convert_record(self, record):
        if self.convert_function:
            messages = self.convert_function(record)
        else:
            messages = record["messages"]

        if self.strategy_function:
            messages = self.strategy_function(
                messages=messages,
                tokenizer=self.tokenizer,
                max_tokens_count=self.max_tokens_count,
                chat_template=self.chat_template,
            )

        # Tokenize the entire conversation
        input_ids = self.get_tokens(messages)
        if len(input_ids) > self.max_tokens_count - 2:
            input_ids = input_ids[: self.max_tokens_count - 2]

        labels = [self.labels_pad_token_id] * len(input_ids)

        # Find the last assistant message
        assistant_indices = [
            idx for idx, msg in enumerate(messages) if msg["role"] in ("assistant", "bot", "gpt")
        ]
        if assistant_indices:
            last_assistant_idx = assistant_indices[-1]

            # Tokenize messages up to and including the last assistant message
            tokens_up_to_last_assistant = self.get_tokens(messages[: last_assistant_idx + 1])

            # Tokenize the last assistant message
            last_assistant_tokens = self.get_tokens([messages[last_assistant_idx]])

            # Calculate start and end indices of the last assistant's message in tokens
            start_idx = len(tokens_up_to_last_assistant) - len(last_assistant_tokens)
            end_idx = len(tokens_up_to_last_assistant)

            # Adjust indices if truncated
            if end_idx > len(labels):
                end_idx = len(labels)
            if start_idx < 0:
                start_idx = 0

            # Set labels for the last assistant's message
            labels[start_idx:end_idx] = input_ids[start_idx:end_idx]

        # Add global BOS and EOS tokens if specified
        if self.add_global_bos and input_ids[0] != self.tokenizer.bos_token_id:
            input_ids = [self.tokenizer.bos_token_id] + input_ids
            labels = [self.labels_pad_token_id] + labels

        if self.add_global_eos and input_ids[-1] != self.tokenizer.eos_token_id:
            input_ids.append(self.tokenizer.eos_token_id)
            labels.append(self.tokenizer.eos_token_id)

        # Convert to tensors
        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        attention_mask = torch.ones_like(input_ids)

        # Ensure lengths are consistent
        assert (
                input_ids.size(0)
                == labels.size(0)
                == attention_mask.size(0)
                <= self.max_tokens_count
        )

        # Print sample once for verification
        if not self.is_printed:
            print("Input IDs:", input_ids)
            print("Labels:", labels)
            print("Fll prompt:", self.tokenizer.decode(input_ids.tolist(), skip_special_tokens=False))
            self.is_printed = True

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
