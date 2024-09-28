import random
import json
import fire
import wandb
import yaml
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForTokenClassification,
)
from transformers import (
    Trainer,
    TrainingArguments,
    logging,
)
from peft import get_peft_model, LoraConfig

from impruver.dataset import ChatDataset
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
    with open(config_file, "r") as r:
        config = yaml.safe_load(r)

    trainer_config = config.get("trainer")
    lora_config = config.get("lora")
    training_args = TrainingArguments(
        output_dir=output_dir, report_to=report_to, **trainer_config
    )

    model_name = config["model"]["name"]
    tokenizer_name = config.get("tokenizer.name", model_name)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.save_pretrained(output_dir)

    train_records = read_jsonl(train_file)
    val_records = read_jsonl(val_file)
    random.shuffle(train_records)

    train_dataset = train_records
    val_dataset = val_records

    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    print("INPUT_IDS")
    print(data_collator([train_dataset[0], train_dataset[1]])["input_ids"][0])
    print("MASK")
    print(data_collator([train_dataset[0], train_dataset[1]])["attention_mask"][0])
    print("LABELS")
    print(data_collator([train_dataset[0], train_dataset[1]])["labels"][0])

    load_in_8bit = bool(config.get("load_in_8bit", False))
    load_in_4bit = bool(config.get("load_in_4bit", False))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    if lora_config:
        lora_config = LoraConfig(**lora_config)
        model = get_peft_model(model, lora_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # if trainer_config.get("report_to", "wandb") == "wandb":
    #     wandb.init(project="rulm_self_instruct", name=config_file)

    trainer.train()
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
