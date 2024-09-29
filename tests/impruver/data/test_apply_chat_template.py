import torch
import unittest
from transformers import AutoTokenizer

from impruver.data.message import Message
from impruver.data.apply_chat_template import apply_chat_template


class TestApplyChatTemplate(unittest.TestCase):
    def setUp(self):
        model = "openai-community/gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello, how are you?"),
            Message(role="assistant", content="I'm doing great!"),
            Message(role="user", content="Can you help me with something?"),
            Message(role="assistant", content="Sure, what do you need help with?"),
        ]
        self.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] }}: {{ message['content'] }}\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "assistant:"
            "{% endif %}"
        )

    def test_apply_chat_template_default(self):
        # Test using a default template
        formatted_messages = apply_chat_template(
            conversation=self.messages,
            chat_template=None,
            add_generation_prompt=False,
            tokenize=False,
            tokenizer=self.tokenizer
        )
        expected_output = (
            "<|endoftext|>system\nYou are a helpful assistant.\n\n"
            "user\nHello, how are you?\n\n"
            "assistant\nI'm doing great!\n\n"
            "user\nCan you help me with something?\n\n"
            "assistant\nSure, what do you need help with?\n\n"
        )
        self.assertEqual(formatted_messages, expected_output.strip())

    def test_apply_chat_template_custom(self):
        # Test using a custom template
        formatted_messages = apply_chat_template(
            conversation=self.messages,
            chat_template=self.chat_template,
            add_generation_prompt=False,
            tokenize=False,
            tokenizer=self.tokenizer
        )
        expected_output = (
            "system: You are a helpful assistant.\n"
            "user: Hello, how are you?\n"
            "assistant: I'm doing great!\n"
            "user: Can you help me with something?\n"
            "assistant: Sure, what do you need help with?\n"
        )
        self.assertEqual(formatted_messages, expected_output.strip())

    def test_apply_chat_template_tokenized(self):
        # Test with tokenization
        tokenized_messages = apply_chat_template(
            conversation=self.messages,
            chat_template=None,
            add_generation_prompt=True,
            tokenize=True,
            tokenizer=self.tokenizer
        )
        # Since tokenization outputs tensor, just verify the type here
        self.assertTrue(isinstance(tokenized_messages, torch.Tensor))


if __name__ == "__main__":
    unittest.main()
