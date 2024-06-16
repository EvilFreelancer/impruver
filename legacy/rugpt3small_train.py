import numpy as np
import evaluate
import logging

import torch
from torch.utils.data import DataLoader
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
    device_map=device,
)

# Load dataset
dataset = load_dataset("yelp_review_full")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


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
    eval_strategy="steps",
    eval_steps=100,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    fp16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    eval_accumulation_steps=8,
    max_grad_norm=1.0,
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
