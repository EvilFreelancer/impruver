from dataclasses import dataclass
from typing import Literal, Dict

Role = Literal[
    "system",
    "user",
    "think",
    "functioncall",
    "functionresponse",
    "assistant"
]


@dataclass
class Message:
    """
    Represents a siungle message in a conversation.
    A message can be a text message, a function call, or a function response.
    """

    role: Role
    content: str

    @classmethod
    def from_dict(cls, d: dict)-> "Message":
        """
        Construct a Message object from a dictionary.

        Args:
            d (dict): A dictionary containing the role and content of the message.

        Returns:
            Message: A Message object.
        """
        return cls(role=d["role"], content=d["content"])

    def to_dict(self) -> Dict[str, str]:
        """
        Convert the Message object to a dictionary.

        Returns:
            dict: A dictionary containing the role and content of the message.
        """
        return {"role": self.role, "content": self.content}
