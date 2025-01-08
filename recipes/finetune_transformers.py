import os
import random
import fire
import yaml
import json

import wandb
import torch
from transformers import Trainer, TrainingArguments, logging, BitsAndBytesConfig, DataCollatorForTokenClassification
from peft import get_peft_model, LoraConfig

from impruver.utils import set_seed, read_jsonl, get_dtype, dynamic_import
from impruver.data import DEFAULT_CHAT_TEMPLATE


def finetune(
    config: str,
    train_path: str = None,
    val_path: str = None,
    output_dir: str = None,
    report_to: str = "none",
    seed: int = 42,
):
    """
    Finetune a model using classes from `transformers` package on a given configuration.

    Args:
        config (str): Path to the configuration file
        train_path (str): Path to the training set
        val_path (str): Path to the validation set
        output_dir (str): Path where the model will be saved
        report_to (str): Where to report the results
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

    # Path where model will be saved
    if output_dir is None:
        output_dir = config['output_dir']

    # Get settings of trainer object
    trainer_config = config.get("trainer", {})

    # Get settings of Peft/LoRA adapter
    lora_config = config.get("lora", None)

    # Class to work with Tokenizer
    model_class = "transformers.AutoModelForCausalLM"
    if "class" in config["model"]:
        model_class = config["model"]["class"]

    # Class to work with Tokenizer
    tokenizer_class = "transformers.AutoTokenizer"
    if "class" in config["tokenizer"]:
        tokenizer_class = config["tokenizer"]["class"]

    # Class to work with Tokenizer
    trainer_class = "transformers.Trainer"
    if "class" in config["trainer"]:
        trainer_class = config["trainer"]["class"]
        del config["trainer"]["class"]

    # Read repo_id of tokenizer or path to configs on disk#
    tokenizer_name = None
    if "name" in config["tokenizer"]:
        tokenizer_name = config["tokenizer"]["name"]
    elif "name" in config["model"]:
        tokenizer_name = config["model"]["name"]

    # Settings related to bitsandbytes and useful only with LoRA adapter training
    load_in_4bit = False
    if "load_in_4bit" in config["model"]:
        load_in_4bit = bool(config["model"]["load_in_4bit"])
    load_in_8bit = False
    if "load_in_8bit" in config["model"]:
        load_in_8bit = bool(config["model"]["load_in_8bit"])

    # Get ddp settings from config if available
    ddp_config = config.get("ddp", {})

    #
    # Tokenizer preparation
    #

    # Init tokenizer object
    tokenizer_obj = dynamic_import(tokenizer_class)
    tokenizer = tokenizer_obj.from_pretrained(tokenizer_name)

    # Check if `chat_template` attribute exists in tokenizer
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        # If it doesn't exist, use default
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    # Use provided chat_template if set in config
    if 'chat_template' in config['tokenizer']:
        tokenizer.chat_template = config['tokenizer']['chat_template']

    # Save tokenizer object with all configs to an output folder
    tokenizer.save_pretrained(output_dir)

    #
    # Dataset
    #

    # Read train and evaluation datasets form JSONL
    train_dataset = read_jsonl(train_path)
    val_dataset = read_jsonl(val_path)

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
    # If ddp enabled and bnb enabled too then use PartialState
    device_map = None
    if load_in_4bit or load_in_8bit:
        from accelerate import PartialState
        device_map = {"": PartialState().process_index}

    # Attention implementation
    attn_implementation = None
    if "attn_implementation" in config["model"]:
        attn_implementation = config["model"]["attn_implementation"]

    # Load model class
    model_obj = dynamic_import(model_class)

    # If model name is set then pre-train
    if 'name' in config["model"]:
        model = model_obj.from_pretrained(
            config["model"]["name"],
            quantization_config=quantization_config,
            # For DDP without bnb should be None, with bnb need to use PartialState index
            device_map=device_map if ddp_config else "auto",
            torch_dtype=dtype,
            attn_implementation=attn_implementation
        )
    else:
        # Init from scratch
        model_config_class = config["model"]['config_class']
        model_config_class_obj = dynamic_import(model_config_class)
        model_config = model_config_class_obj(**config["model"]["config"])
        model = model_obj._from_config(model_config)

    # If we need to train a LoRA adapter
    if lora_config:
        lora_config = LoraConfig(**lora_config)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # First we need to check if we have a default generation config in model
        if hasattr(model, "generation_config") and model.generation_config is not None:
            generation_config = model.generation_config.to_diff_dict()
        else:
            generation_config = {"max_new_tokens": 200, "repetition_penalty": 1.2, "do_sample": True}
        # Next let's save a default generation config
        generation_config_path = os.path.join(output_dir, "generation_config.json")
        with open(generation_config_path, "w", encoding="utf-8") as f:
            json.dump(generation_config, f, ensure_ascii=False, default=lambda o: o.__dict__, indent=2)

    #
    # Trainer
    #

    # Merge trainer_config and ddp_config
    training_args_dict = trainer_config.copy()
    training_args_dict.update(ddp_config)

    # Fixing "evaL_loss" issue
    training_args_dict.update({"label_names": ["labels"]})

    # If reporting to W&B is enabled
    if report_to == "wandb":
        os.environ["WANDB_MODE"] = "online"
        wandb.init(project="impruver", name=str(config_path), config=config)

    # Prepare trainer configuration
    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to=report_to,
        run_name=str(config_path),
        **training_args_dict
    )

    # Load trainer class object
    trainer_obj = dynamic_import(trainer_class)

    # Init trainer object and pass all important parameters to it
    trainer = trainer_obj(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    #
    # Training loop
    #

    trainer.train()

    #
    # Saving results
    #

    model.save_pretrained(output_dir, safe_serialization=False)


if __name__ == "__main__":
    fire.Fire(finetune)
