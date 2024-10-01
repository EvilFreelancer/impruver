import json
from typing import List, Dict, Callable, Optional
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, logging

from impruver.data import apply_chat_template

_log = logging.get_logger()
logging.set_verbosity_info()


class ChatDataset(Dataset):
    def __init__(
            self,
            original_records: List[Dict],
            tokenizer,
            max_tokens_count: int,
            add_global_bos: bool = True,
            add_global_eos: bool = True,
            labels_pad_token_id: int = -100,
            converter: Optional[Callable[[Dict], List[Dict]]] = None,
            strategy_function: Optional[Callable[[List[Dict]], List[Dict]]] = None,
            chat_template: Optional[str] = None,
            only_target_loss: bool = False,
    ):
        self.original_records = original_records
        self.tokenizer = tokenizer
        self.max_tokens_count = max_tokens_count
        self.labels_pad_token_id = labels_pad_token_id
        self.add_global_bos = add_global_bos
        self.add_global_eos = add_global_eos
        self.converter = converter
        self.strategy_function = strategy_function
        self.chat_template = chat_template
        self.only_target_loss = only_target_loss
        self.is_printed = True

        self.records = []
        for record in tqdm(original_records):
            tensors = self.convert_record(record)
            if tensors is None:
                continue
            self.records.append(tensors)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def get_tokens(self, messages):
        if (
                hasattr(self.tokenizer, 'apply_chat_template')
                and hasattr(self.tokenizer, 'chat_template')
                and self.tokenizer.chat_template is not None
        ):
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
                add_special_tokens=False,
                tokenize=True,
                add_generation_prompt=False,
                tokenizer=self.tokenizer,
            )

        if int(tokens[0][0]) == self.tokenizer.bos_token_id:
            tokens = tokens[0][1:]

        return tokens

    def convert_record(self, record):
        if self.converter:
            messages = self.converter(record)
        else:
            messages = record["messages"]
            if isinstance(record["messages"], str):
                messages = json.loads(record["messages"])

        # Ig sample is empty, then we don't need it
        if messages is None:
            return None

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
            # If length of conversation is larger than allowed then skip this sample
            _log.info(f'Input is "{len(input_ids)}" tokens, max allowed is "{self.max_tokens_count}" tokens, skip...')
            return None

        # Create list of labels with same size as input_ids list
        labels = [self.labels_pad_token_id] * len(input_ids)

        # Find the last message in conversation
        last_indices = [idx for idx, msg in enumerate(messages)]
        if last_indices:
            last_idx = last_indices[-1]

            # Tokenize messages including the last message
            tokens_up_to_last = self.get_tokens(messages[: last_idx + 1])

            # Tokenize the last message
            last_tokens = self.get_tokens([messages[last_idx]])

            # Calculate start and end indices of the last assistant's message in tokens
            start_idx = len(tokens_up_to_last) - len(last_tokens)
            end_idx = len(tokens_up_to_last)

            # Adjust indices if truncated
            if end_idx > len(labels):
                end_idx = len(labels)
            if start_idx < 0:
                start_idx = 0

            if self.only_target_loss:
                current_idx = 0
                for message in messages:
                    message_tokens = self.get_tokens([message])
                    message_len = len(message_tokens)
                    if message["role"] in ("assistant", "bot", "gpt"):
                        labels[current_idx:current_idx + message_len] = input_ids[current_idx:current_idx + message_len]
                    current_idx += message_len
            else:
                # If only_target_loss=False, then labels is a copy of input_ids
                labels = input_ids.tolist()

            # Set labels for the last message
            labels[start_idx:end_idx] = input_ids[start_idx:end_idx]

        # Add global BOS and EOS tokens if specified
        if self.add_global_bos and input_ids[0] != self.tokenizer.bos_token_id:
            input_ids = torch.cat((torch.tensor([self.tokenizer.bos_token_id]), input_ids), dim=-1)
            labels = [self.labels_pad_token_id] + labels

        if self.add_global_eos and input_ids[-1] != self.tokenizer.eos_token_id:
            input_ids = torch.cat((input_ids, torch.tensor([self.tokenizer.eos_token_id])), dim=-1)
            labels += [self.tokenizer.eos_token_id]

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
            print("Full prompt:", self.tokenizer.decode(input_ids.tolist(), skip_special_tokens=False))
            self.is_printed = True

        return {
            "input_ids": input_ids.tolist(),
            "labels": labels.tolist(),
            "attention_mask": attention_mask.tolist(),
        }


if __name__ == '__main__':
    records = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing great!"},
                {"role": "user", "content": "Can you help me with something?"},
                {"role": "assistant", "content": "Sure, what do you need help with?"}
            ]
        }
    ]
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    dataset = ChatDataset(
        original_records=records,
        tokenizer=tokenizer,
        max_tokens_count=1024,
        only_target_loss=True,
    )

    print(dataset[0])
