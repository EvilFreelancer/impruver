import numpy as np
import evaluate
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

logger = logging.getLogger(__name__)

# Initialize tokenizer and model
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
model = AutoModelForCausalLM.from_pretrained(
    "ai-forever/rugpt3small_based_on_gpt2",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map=device,
)

# Load dataset
dataset = load_dataset("yelp_review_full")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


# Add labels to the datasets
def add_labels(batch):
    batch["labels"] = batch["input_ids"].copy()
    return batch


small_train_dataset = small_train_dataset.map(add_labels, batched=True)
small_eval_dataset = small_eval_dataset.map(add_labels, batched=True)

training_args = TrainingArguments(
    output_dir="test_trainer",
    eval_strategy="no",  # no, epoch, steps
    eval_steps=100,  # if eval_strategy is "steps"
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
)

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training

model_to_save.save_pretrained(save_directory=training_args.output_dir, safe_serialization=False)
tokenizer.save_pretrained(save_directory=training_args.output_dir)
