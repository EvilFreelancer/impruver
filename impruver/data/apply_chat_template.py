from typing import Optional, List, Union, Dict
from jinja2 import Template

from .tokenizer import Tokenizer

DEFAULT_RAW_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
        "{% set content = '' + message['content'] | trim + '' %}"
        '{{ bos_token + content + eos_token }}'
    "{% endfor %}"
)

DEFAULT_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
        "{% set content = '' + message['role'] + '\n' + message['content'] | trim + '' %}"
        "{{ bos_token + content + eos_token}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
        "{% set content = 'assistant' + '\n' %}"
        "{{ bos_token + content }}"
    "{% endif %}"
)


def apply_chat_template(
        conversation: List[Dict],
        chat_template: Optional[str] = None,
        add_special_tokens: bool = False,
        add_generation_prompt: Optional[bool] = False,
        tokenize: bool = True,
        tokenizer: Tokenizer = None
) -> Union[List[int], Dict, str]:
    """
    Apply a chat template to a conversation in GPT2 sample format.

    Args:
        conversation (List[dict]): The conversation to apply the template to.
        chat_template (Optional[str]): The Jinja2 template to apply to the conversation. Defaults to None.
        add_special_tokens (bool): Whether to add begin and end of text tokens. Defaults to False.
        add_generation_prompt (Optional[bool]): Whether to add generation prompt. Defaults to False.
        tokenize (bool): Whether to tokenize the rendered text. Defaults to True.
        tokenizer (Optional[Tokenizer]): The tokenizer to use. Defaults to None.

    Returns:
        Union[List[int], Dict, str]: The rendered text.
    """

    # Use provided chat template or default template
    template_str = chat_template if chat_template else DEFAULT_CHAT_TEMPLATE
    template = Template(template_str)

    # Prepare the context for the template
    context = {
        'messages': [{'role': msg['role'], 'content': msg['content']} for msg in conversation],
        'add_generation_prompt': add_generation_prompt,
        'bos_token': tokenizer.bos_token if tokenizer else '',
        'eos_token': tokenizer.eos_token if tokenizer else ''
    }

    # Render the template
    rendered = template.render(context).strip()

    # Add begin and end of text tokens
    if add_special_tokens:
        rendered = "<|startoftext|>" + rendered + "<|endoftext|>"

    if tokenize and tokenizer:
        # Tokenize the rendered text
        return tokenizer.encode(rendered, return_tensors="pt")
    else:
        # Return the rendered text as is
        return rendered
