import logging
from typing import List, Optional
from impruver.data.message import Message
from impruver.data.tokenizer import Tokenizer
from impruver.data.apply_chat_template import apply_chat_template
from impruver.utils.get_logger import get_logger

_log: logging.Logger = get_logger()


def last_message_by_assistant(
        tokenizer: Tokenizer,
        messages: List[Message],
        max_tokens_count: int,
        chat_template: Optional[str] = None,
) -> List[Message]:
    """
    Return the last message by assistant.

    If there is no message by assistant, return the first message.
    If there is more than one message by assistant, return the last one.

    Args:
        tokenizer (Tokenizer): Tokenizer object
        messages (List[Message]): List of Message objects
        max_tokens_count (int): Max tokens count
        chat_template (Optional[str]): The Jinja2 template to apply to the conversation. Defaults to None.

    Returns:
        List[Message]: List of Message objects
    """

    _log.debug(f"Max tokens count: {max_tokens_count}")
    tmp_messages = messages.copy()
    while tmp_messages:

        # Apply chat format from template
        if hasattr(tokenizer, 'apply_chat_template'):
            # On modern tokenizers we may use chat_template
            formated_messages = tokenizer.apply_chat_template(
                tmp_messages,
                chat_template=chat_template,  # Use default_chat_template if None
                tokenize=False
            )
        else:
            # On old tokenizers we will use a custom apply_chat_template
            formated_messages = apply_chat_template(
                tmp_messages,
                chat_template=chat_template,  # Use DEFAULT_CHAT_TEMPLATE if None
                tokenize=False,
                tokenizer=tokenizer
            )

        # Tokenize all messages
        tokenized_messages = tokenizer.encode(formated_messages, return_tensors="pt")

        # Calculate sum of total tokens of all formated messages
        total_tokens = sum(len(tokens) for tokens in tokenized_messages)
        _log.debug(f"Total tokens: {total_tokens}")

        # If total tokens pass the limit
        if total_tokens <= max_tokens_count:
            # If last message is from the assistant
            if tmp_messages[-1].role == 'assistant':
                # Return the messages, formated and tokenized they versions
                return tmp_messages
            else:
                # Remove the last message, then try again
                tmp_messages.pop()
        else:
            # Remove the last message if total tokens exceed the limit
            tmp_messages.pop()

    _log.error(
        "Unable to fit messages within the maximum token length "
        "with the last message being from the assistant."
    )

    return messages
