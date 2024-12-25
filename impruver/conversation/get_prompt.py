from typing import List, Dict

from impruver.data.tokenizer import Tokenizer


def get_prompt(tokenizer: Tokenizer, messages: List[Dict], add_generation_prompt: bool = False):
    return tokenizer.apply_chat_template(
        messages,
        add_special_tokens=False,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
