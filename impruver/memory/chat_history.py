from typing import List, Dict

from impruver.data.tokenizer import Tokenizer


class ChatHistory:
    def __init__(self, history_limit: int = None, system_prompt: str = None):
        self.history_limit: int | None = history_limit
        self.system_prompt: str | None = system_prompt
        self.messages: List[Dict] = []
        if self.system_prompt is not None:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def add_message(self, role: str, message: str):
        self.messages.append({"role": role, "content": message})
        self.trim_history()

    def add_user_message(self, message: str):
        self.add_message("user", message)

    def add_assistant_message(self, message: str):
        self.add_message("assistant", message)

    def add_function_call(self, message: str):
        self.add_message("function_call", message)

    def add_function_response(self, message: str):
        self.add_message("function_response", message)

    def trim_history(self):
        appendix = 0
        if self.system_prompt is not None:
            appendix = 1
        if self.history_limit is not None and len(self.messages) > self.history_limit + appendix:
            overflow = len(self.messages) - (self.history_limit + appendix)
            self.messages = [self.messages[0]] + self.messages[overflow + appendix:]

    def get_messages(self) -> list:
        return self.messages


def get_prompt(tokenizer: Tokenizer, messages: List[Dict], add_generation_prompt: bool = False):
    return tokenizer.apply_chat_template(
        messages,
        add_special_tokens=False,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def generate(model, tokenizer, prompt, generation_config):
    data = tokenizer(prompt, return_tensors="pt")
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(**data, generation_config=generation_config)[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()
