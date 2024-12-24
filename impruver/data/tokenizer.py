from typing import List, Protocol, Set, Dict


class Tokenizer(Protocol):
    """Abstract tokenizer"""

    bos_token_id: int
    eos_token_id: int
    pad_token_id: int

    bos_token: str
    eos_token: str
    pad_token: str

    # Tokens indicating that generation should stop
    stop_tokens: Set[int]

    # Max length of model' context
    model_max_length: int

    def encode(self, text: str | List[str], **kwargs) -> List[int] | List[List[int]]:
        """
        Given a string, return a list of token ids
        """

    def decode(self, token_ids: List[int], add_bos: bool, add_eos: bool, **kwargs) -> str:
        """
        Given a list of token ids, return the decoded text
        """

    def apply_chat_template(self, token_ids: List[Dict], **kwargs):
        """
        Given a list of messages, return a list of tokens for the concatenated
        and formatted message
        """
        pass
