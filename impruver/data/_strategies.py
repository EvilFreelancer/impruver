import logging
from typing import List, Optional
from impruver.data._message import Message
from impruver.data._tokenizer import Tokenizer
from impruver.data.apply_chat_template import apply_chat_template
from impruver.utils.get_logger import get_logger

_log: logging.Logger = get_logger("DEBUG")


def last_message_by_assistant(
        tokenizer: Tokenizer,
        messages: List[Message],
        max_tokens_count: int,
        chat_template: Optional[str] = None,
) -> (List[Message], List[str]):
    _log.debug(f"Max tokens count: {max_tokens_count}")
    while messages:

        # Apply chat format from template
        if hasattr(tokenizer, 'apply_chat_template'):
            # On modern tokenizers we may use chat_template
            formated_messages = tokenizer.apply_chat_template(
                messages,
                chat_template=chat_template,  # Use default_chat_template if None
                tokenize=False
            )
        else:
            # On old tokenizers we will use a custom apply_chat_template
            formated_messages = apply_chat_template(
                messages,
                chat_template=chat_template,  # Use DEFAULT_CHAT_TEMPLATE if None
                tokenize=False,
                tokenizer=tokenizer
            )

        # Tokenize all messages and count total tokens
        tokenized_messages = tokenizer.encode(formated_messages, return_tensors="pt")

        # Calculate sum of total tokens of all formated messages
        total_tokens = sum(len(tokens) for tokens in tokenized_messages)
        _log.debug(f"Total tokens: {total_tokens}")

        # If total tokens pass the limit
        if total_tokens <= max_tokens_count:
            # If last message is from the assistant
            if messages[-1].role == 'assistant':
                # Return the messages, formated and tokenized they versions
                return messages, formated_messages, tokenized_messages
            else:
                # Remove the last message, then try again
                messages.pop()
        else:
            # Remove the last message if total tokens exceed the limit
            messages.pop()

    raise ValueError(
        "Unable to fit messages within the maximum token length "
        "with the last message being from the assistant."
    )
