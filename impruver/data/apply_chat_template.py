from typing import Optional, List, Union, Dict
from jinja2 import Template
from impruver.data._message import Message

DEFAULT_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
            "{% set content = '' + message['role'] + '\n' + message['content'] | trim + '' %}"
            "{% if loop.index0 == 0 %}"
                "{% set content = bos_token + content %}"
            "{% endif %}"
            '{{ content }}\n\n'
        "{% endfor %}"
    "{% if add_generation_prompt %}"
        "{{ 'assistant\n' }}"
    "{% endif %}"
)


def apply_chat_template(
        conversation: List[dict],
        chat_template: Optional[str] = None,
        add_special_tokens: bool = False,
        add_generation_prompt: Optional[bool] = False,
        tokenize: bool = True,
        tokenizer=None
) -> Union[List[int], Dict, str]:
    # Use provided chat template or default template
    template_str = chat_template if chat_template else DEFAULT_CHAT_TEMPLATE
    template = Template(template_str)

    # Prepare the context for the template
    context = {
        'messages': [{'role': msg['role'], 'content': msg['content']} for msg in conversation],
        'add_generation_prompt': add_generation_prompt,
        'bos_token': tokenizer.bos_token if tokenizer else ''
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
