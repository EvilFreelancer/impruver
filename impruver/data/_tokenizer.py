from typing import List, Protocol, Set

from ._message import Message


class Tokenizer(Protocol):
    """Abstract tokenizer"""

    bos_token_id: int
    eos_token_id: int
    pad_token_id: int

    # Tokens indicating that generation should stop
    stop_tokens: Set[int]

    def encode(self, text: str | List[str], **kwargs) -> List[int] | List[List[int]]:
        """
        Given a string, return the a list of token ids.
        """

    def decode(self, token_ids: List[int], add_bos: bool, add_eos: bool, **kwargs) -> str:
        """
        Given a list of token ids, return the decoded text.
        """

    def apply_chat_template(self, token_ids: List[Message], **kwargs):
        """
        Given a list of messages, return a list of tokens for the concatenated
        and formatted messages.
        """
        pass
