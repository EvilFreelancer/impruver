import os
import random
import fire
import yaml

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments, logging, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

from impruver.utils import set_seed, read_jsonl, get_dtype


def train(
        config_file: str,
        train_file: str,
        val_file: str,
        output_dir: str,
        report_to: str = None,
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
    model_name = config["model"]["name"]

    # Read repo_id of tokenizer or path to configs on disk#
    tokenizer_name = model_name
    if "name" in config["tokenizer"]:
        tokenizer_name = config["tokenizer"]["name"]

    # Settings related to bitsandbytes and useful only with LoRA adapter training
    load_in_4bit = False
    if "load_in_4bit" in config["model"]:
        load_in_4bit = bool(config["model"]["load_in_4bit"])
    load_in_8bit = False
    if "load_in_8bit" in config["model"]:
        load_in_8bit = bool(config["model"]["load_in_8bit"])

    #
    # Tokenizer preparation
    #

    # Init tokenizer object
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

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

    # Data Type is bfloat16 by default
    dtype = torch.bfloat16
    if "dtype" in config["model"]:
        dtype = get_dtype(config["model"]["dtype"])

    # Quantization related settings, can be used only in combination with LoRA adapter
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )

    # Attention implementation
    attn_implementation = None
    if "attn_implementation" in config["model"]:
        attn_implementation = config["model"]["attn_implementation"]

    # Init model object
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
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

    model.save_pretrained(output_dir, safe_serialization=False)


if __name__ == "__main__":
    fire.Fire(train)
