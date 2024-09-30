import yaml
import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import logging

from impruver.utils import set_seed
from impruver.data import apply_chat_template, DEFAULT_CHAT_TEMPLATE

DEFAULT_SYSTEM_PROMPT = "Ты — Saiga 2, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."


class ChatHistory:
    def __init__(self, history_limit: int = None):
        self.history_limit = history_limit
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def add_message(self, role, message):
        self.messages.append({"role": role, "content": message})
        self.trim_history()

    def add_user_message(self, message):
        self.add_message("user", message)

    def add_assistant_message(self, message):
        self.add_message("assistant", message)

    def trim_history(self):
        appendix = 0
        if self.system_prompt is not None:
            appendix = 1
        if self.history_limit is not None and len(self.messages) > self.history_limit + appendix:
            overflow = len(self.messages) - (self.history_limit + appendix)
            self.messages = [self.messages[0]] + self.messages[overflow + appendix:]

    def get_messages(self) -> list:
        return self.messages


def get_prompt(tokenizer, messages):
    if (
            hasattr(tokenizer, 'apply_chat_template')
            and hasattr(tokenizer, 'chat_template')
            and tokenizer.chat_template is not None
    ):
        prompt = tokenizer.apply_chat_template(
            messages,
            chat_template=DEFAULT_CHAT_TEMPLATE,
            add_special_tokens=False,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        # Assume apply_chat_template is a function you have defined
        prompt = apply_chat_template(
            messages,
            chat_template=DEFAULT_CHAT_TEMPLATE,
            tokenize=False,
            add_generation_prompt=False,
            # tokenizer=tokenizer,
        )

    return prompt


def generate(model, tokenizer, prompt, generation_config):
    data = tokenizer(prompt, return_tensors="pt")
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data,
        generation_config=generation_config
    )[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()


def infer(
        config_file: str,
        output_dir: str = None,
        seed: int = 42,
):
    set_seed(seed)
    logging.set_verbosity_info()

    #
    # Load configuration
    #

    # Read config from disk
    with open(config_file, "r") as r:
        config = yaml.safe_load(r)

    # Assume that model and tokenizer are in the same directory
    tokenizer_name = output_dir
    model_name = output_dir
    if output_dir is None:
        tokenizer_name = config['output_dir']
        model_name = config['output_dir']

    #
    # Load mode and tokenizer
    #

    # Init pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    generation_config = GenerationConfig.from_pretrained(model_name)
    generation_config.max_new_tokens = 1024
    generation_config.repetition_penalty = 1.2
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    #
    # Chat loop
    #

    # Start chat loop
    chat_history = ChatHistory(history_limit=10)
    while True:
        user_message = input("User: ")

        # Reset chat command
        if user_message.strip() == "/reset":
            chat_history = ChatHistory(history_limit=10)
            print("History reset completed!")
            continue

        # Skip empty messages from user
        if user_message.strip() == "":
            continue

        # Add user message to chat history
        chat_history.add_user_message(user_message)

        # Get list of messages
        prompt = get_prompt(tokenizer, chat_history.get_messages())
        print("==============================")
        print(prompt)
        print("==============================")
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            generation_config=generation_config
        )
        chat_history.add_assistant_message(output)
        print("Bot:", output)
        print()
        print("==============================")
        print()


if __name__ == "__main__":
    fire.Fire(infer)
