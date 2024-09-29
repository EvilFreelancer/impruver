import os
import random
import fire
import yaml

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments, logging
from peft import get_peft_model, LoraConfig

from impruver.utils import set_seed, read_jsonl


def train(
        config_file: str,
        train_file: str,
        val_file: str,
        output_dir: str,
        report_to: str = None,  # "wandb",
        seed: int = 42,
):
    set_seed(seed)
    logging.set_verbosity_info()
    os.environ["WANDB_DISABLED"] = "true"

    #
    # Load configuration
    #

    # Read config from disk
    with open(config_file, "r") as r:
        config = yaml.safe_load(r)

    # Get settings of trainer object
    trainer_config = config.get("trainer")

    # Get settings of Peft/LoRA adapter
    lora_config = config.get("lora")

    # Read repo_id of model or path to model weights on disk
    model_name = config.get("model.name")

    # Read repo_id of tokenizer or path to configs on disk#
    tokenizer_name = config.get("tokenizer.name", model_name)

    # Settings related to bitsandbytes and useful only with LoRA adapter training
    load_in_8bit = bool(config.get("load_in_8bit", False))
    load_in_4bit = bool(config.get("load_in_4bit", False))

    #
    # Tokenizer preparation
    #

    # Init tokenizer object
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # If special tokens is set, then replace defaults
    if config.get("tokenizer.pad_token", None):
        tokenizer.pad_token = config["pad_token"]
    if config.get("tokenizer.eos_token", None):
        tokenizer.pad_token = config["eos_token"]
    if config.get("tokenizer.bos_token", None):
        tokenizer.pad_token = config["bos_token"]

    # Save tokenizer object with all configs to an output folder
    tokenizer.save_pretrained(output_dir)

    #
    # Dataset
    #

    # Read train and evaluation datasets form JSONL
    train_dataset = read_jsonl(train_file)
    val_dataset = read_jsonl(val_file)

    # Randomize order of items in train dataset
    random.shuffle(train_dataset)

    # Init data collator for adding pad tokens to `input_ids` and `labels` lists
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    #
    # Model preparation
    #

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        device_map="auto",
        torch_dtype=torch.float16,  # todo: configurable
        attn_implementation="flash_attention_2",
    )

    # If we need to train a LoRA adapter
    if lora_config:
        lora_config = LoraConfig(**lora_config)
        model = get_peft_model(model, lora_config)

    #
    # Trainer
    #

    # Prepare trainer configuration
    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to=report_to,
        **trainer_config
    )

    # Init trainer object and pass all important parameters to it
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # If reporting to W&B is enabled
    if trainer_config.get("report_to", None) == "wandb":
        import wandb
        os.environ["WANDB_DISABLED"] = "false"
        wandb.init(project="rulm_self_instruct", name=config_file)

    #
    # Training loop
    #

    trainer.train()

    #
    # Saving results
    #

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
