from time import sleep

import yaml
import mmh3
import fire
import json
from datasets import load_dataset
from transformers import AutoTokenizer

from impruver.dataset import ChatDataset


def convert_function(conversation: dict) -> list:
    messages = []
    for item in conversation['conversations']:
        content = item['value']
        role = 'assistant'
        if item['from'] == 'human':
            role = 'user'
        if item['from'] == 'gpt':
            role = 'assistant'
        messages.append({"role": role, "content": content})
    return messages


def compose_dataset(config_path: str, train_path: str, val_path: str):
    # Load the config file
    with open(config_path, "r") as r:
        config = yaml.safe_load(r)

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['name'])

    # Default max_tokens_count from tokenizer
    max_tokens_count = tokenizer.model_max_length

    # If max_tokens_count was replaced in config, then overwrite it
    if 'max_tokens_count' in config['tokenizer']:
        max_tokens_count = config['tokenizer']['max_tokens_count']

    # Prevent automatic truncation by setting model_max_length to a large value
    tokenizer.model_max_length = int(1e30)  # Disable automatic truncation

    # For each dataset in the config...
    for dataset in config['datasets']:

        # Load the actual dataset from Hugging Face's datasets library.
        hf_dataset = load_dataset(dataset['name'], split=dataset['split'])

        # Create an instance of ChatDataset
        chat_dataset = ChatDataset(
            original_records=list(hf_dataset),
            tokenizer=tokenizer,
            convert_function=convert_function,
            max_tokens_count=dataset.get("max_tokens_count", max_tokens_count),
        )

        # TODO: multithread

        train_records = []
        val_records = []
        for record in chat_dataset:
            string = str(record)
            hash = mmh3.hash(string, signed=False)
            if hash % 100 < 97:
                train_records.append(record)
            else:
                val_records.append(record)

        with open(train_path, "w") as w:
            for record in train_records:
                w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")

        with open(val_path, "w") as w:
            for record in val_records:
                w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    fire.Fire(compose_dataset)
