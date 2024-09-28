import torch
from torch.utils.data import Dataset
from typing import Callable, Dict, Any, Optional, List
from datasets import load_dataset

class RawDataset(Dataset):
    def __init__(
        self,
        *,
        tokenizer,
        source: str,
        max_tokens_count: Optional[int] = None,
        convert_function: Optional[Callable[[Dict[str, Any]], str]] = None,
        strategy_function: Optional[Callable[[str, Any], str]] = None,
        **load_dataset_kwargs: Any,
    ):
        self.tokenizer = tokenizer
        self.source = source
        self.convert_function = convert_function
        self.strategy_function = strategy_function
        self.data = load_dataset(self.source, **load_dataset_kwargs)
        self.max_tokens_count = max_tokens_count or tokenizer.model_max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.data[index]
        text = self.convert_function(sample) if self.convert_function else sample['text']

        # Apply the strategy function if provided
        if self.strategy_function is not None:
            text = self.strategy_function(
                text=text,
                tokenizer=self.tokenizer,
                max_tokens_count=self.max_tokens_count,
            )

        # Tokenize the text
        tokenized = self.tokenizer(
            text,
            max_length=self.max_tokens_count,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

        input_ids = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
