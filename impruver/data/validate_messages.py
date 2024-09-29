from typing import List

from .message import Message


def validate_messages(messages: List[Message]) -> None:
    """
    Given a list of messages, ensure that messages form a valid
    back-and-forth conversation. An error will be raised if:

    - There is a system message that's not the first message
    - There are two consecutive user messages
    - An assistant message comes before the first user message
    - The message is empty
    - Messages are shorter than length of 2 (min. one user-assistant turn)


    Args:
        messages (List[Message]): the messages to validate.

    Raises:
        ValueError: If the messages are invalid.
    """
    if len(messages) < 2:
        raise ValueError(
            f"Messages must be at least length 2, but got {len(messages)} messages"
        )

    last_turn = "assistant"
    for i, message in enumerate(messages):
        if message.role == "assistant" and last_turn != "user" and last_turn != "system":
            raise ValueError(
                f"Assistant message before expected user message at index {i} in messages"
            )
        if message.role == "user" and last_turn == "user":
            raise ValueError(
                f"Two consecutive user messages at index {i} and {i - 1} in messages"
            )
        if message.role == "system" and i > 0:
            raise ValueError(
                f"System message at index {i} in messages, but system messages must come first"
            )
        last_turn = message.role
