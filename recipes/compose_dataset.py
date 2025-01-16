import os
import yaml
import json
import random
import fire

from datasets import load_dataset
from transformers import logging
from datasketch import MinHash, MinHashLSH, LeanMinHash
from tqdm import tqdm

from impruver.dataset import ChatDataset
from impruver.utils import set_seed, dynamic_import
from impruver.data import DEFAULT_CHAT_TEMPLATE


def calc_fingerprint(tokens, num_perm=128):
    """
    Calculate the MinHash fingerprint of a given list of token IDs
    """
    m = MinHash(num_perm=num_perm)
    for token_id in tokens:
        # Convert the token_id to bytes and update the MinHash
        m.update(token_id.to_bytes(8, byteorder='little', signed=False))
    return LeanMinHash(m)


def load_datasets(config, tokenizer, max_tokens_count):
    """
    Load datasets specified in the config and return a list of records.
    """
    all_records = []
    for dataset in config['datasets']:
        print(dataset['name'])

        # If split was specified in the config, then use it
        split = dataset.get('split', 'train')

        # If subset (called "name" in load_dataset) was specified in the config, then use it, otherwise use None
        subset = dataset.get('subset', None)

        # Some datasets have a converter function
        converter = None
        if 'converter' in dataset:
            converter = dynamic_import(dataset['converter'])

        # Load the actual dataset from HuggingFace's datasets library.
        hf_dataset = load_dataset(dataset['name'], name=subset, split=split, trust_remote_code=True)

        # Create an instance of ChatDataset
        chat_dataset = ChatDataset(
            original_records=list(hf_dataset),
            tokenizer=tokenizer,
            converter=converter,
            mapping=dataset.get("mapping", None),
            add_global_bos=dataset.get("add_global_bos", True),
            add_global_eos=dataset.get("add_global_eos", True),
            only_target_loss=dataset.get("only_target_loss", True),
            max_tokens_count=dataset.get("max_tokens_count", max_tokens_count),
        )

        # Add records to the list
        all_records.extend(chat_dataset)

        # Optionally clear the chat_dataset to free memory
        del chat_dataset

    return all_records


def deduplicate_records(records, num_perm: int = 128, threshold: float = 0.8):
    idx_to_minhash = {}

    # Step 1: Calculate MinHash for each record
    print("Fingerprinting...")
    for idx, record in tqdm(enumerate(records), total=len(records)):
        tokens = record['input_ids']

        # Calculate MinHash fingerprint directly from tokens
        minhash = calc_fingerprint(tokens, num_perm=num_perm)
        idx_to_minhash[idx] = minhash

    # Step 2: Deduplicate records using MinHashLSH
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    filtered_records = []

    print("Deduplicating...")
    for idx, minhash in tqdm(idx_to_minhash.items()):
        is_dup = False
        for other_idx in lsh.query(minhash):
            other_minhash = idx_to_minhash[other_idx]
            if minhash.jaccard(other_minhash) > threshold:
                is_dup = True
                break  # Stop checking if a duplicate is found
        if not is_dup:
            lsh.insert(idx, minhash)
            filtered_records.append(records[idx])

    return filtered_records


def split_and_save_records(records: list, train_path: str, val_path: str):
    """
    Split records into train and validation sets and save them to files.
    """
    random.shuffle(records)
    border = int(0.95 * len(records))
    train_records = records[:border]
    val_records = records[border:]

    # Write the records to files
    with open(train_path, "w", encoding='utf-8') as w:
        for record in train_records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")

    with open(val_path, "w", encoding='utf-8') as w:
        for record in val_records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


def compose_dataset(
    config: str,
    train_path: str = None,
    val_path: str = None,
    seed: int = 42,
):
    """
    Compose a dataset from multiple datasets specified in the config file.

    Args:
        config (str): Path to the config file
        train_path (str): Path to the train set
        val_path (str): Path to the validation set
        seed (int): Random(?) seed
    """

    set_seed(seed)
    logging.set_verbosity_info()

    #
    # Load configuration
    #

    if os.path.exists(config):
        config_path = config
    else:
        import recipes
        recipes_path = os.path.join(recipes.__path__[0])
        config_path = recipes_path + '/configs/' + config + '.yaml'

    # Read config
    with open(config_path, "r") as r:
        config = yaml.safe_load(r)

    # Get paths to train and validation sets
    if train_path is None:
        train_path = config['train_path']
    if val_path is None:
        val_path = config['val_path']

    # Class to work with Tokenizer
    tokenizer_class = "transformers.AutoTokenizer"
    if "class" in config["tokenizer"]:
        tokenizer_class = config["tokenizer"]["class"]

    # Read repo_id of tokenizer or path to configs on disk#
    tokenizer_name = None
    if "name" in config["tokenizer"]:
        tokenizer_name = config["tokenizer"]["name"]
    elif "name" in config["model"]:
        tokenizer_name = config["model"]["name"]

    #
    # Tokenizer preparation
    #

    # Init tokenizer object
    tokenizer_obj = dynamic_import(tokenizer_class)
    tokenizer = tokenizer_obj.from_pretrained(tokenizer_name, trust_remote_code=True)

    # Add special tokens to tokenizer
    if 'special_tokens' in config['tokenizer']:
        special_tokens = {}
        if 'pad_token_id' in config['tokenizer']['special_tokens']:
            special_tokens['pad_token'] = config['tokenizer']['special_tokens']['pad_token']
        if 'eos_token_id' in config['tokenizer']['special_tokens']:
            special_tokens['eos_token'] = config['tokenizer']['special_tokens']['eos_token']
        if 'bos_token_id' in config['tokenizer']['special_tokens']:
            special_tokens['bos_token'] = config['tokenizer']['special_tokens']['bos_token']
        if 'mask_token_id' in config['tokenizer']['special_tokens']:
            special_tokens['mask_token'] = config['tokenizer']['special_tokens']['mask_token']
        if 'unk_token_id' in config['tokenizer']['special_tokens']:
            special_tokens['unk_token'] = config['tokenizer']['special_tokens']['unk_token']
        tokenizer.add_special_tokens(special_tokens)

    # Check if `chat_template` attribute exists in tokenizer
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        # If it doesn't exist, use default
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    # Use provided chat_template if set in config
    if 'chat_template' in config['tokenizer']:
        tokenizer.chat_template = config['tokenizer']['chat_template']

    #
    # Tokenizer chat length
    #

    # Default max_tokens_count from tokenizer
    max_tokens_count = tokenizer.model_max_length

    # If max_tokens_count was replaced in config, then overwrite it
    if 'max_tokens_count' in config['tokenizer']:
        max_tokens_count = config['tokenizer']['max_tokens_count']

    # Prevent automatic truncation by setting model_max_length to a large value
    tokenizer.model_max_length = int(1e30)  # Disable automatic truncation

    #
    # Make work
    #

    # Step 1: Load datasets
    records = load_datasets(config, tokenizer, max_tokens_count)

    # Step 2: Deduplicate records
    deduplicated_records = deduplicate_records(records)

    # Step 3: Split and save records
    split_and_save_records(deduplicated_records, train_path, val_path)


if __name__ == "__main__":
    fire.Fire(compose_dataset)
