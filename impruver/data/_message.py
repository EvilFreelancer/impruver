from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            role=d["role"],
            content=d["content"]
        )

    def to_dict(self):
        return {"role": self.role, "content": self.content}
