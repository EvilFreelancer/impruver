import unittest
import torch
import logging
from transformers import AutoTokenizer

from impruver.utils import get_logger
from impruver.data._strategies import last_message_by_assistant
from impruver.data._message import Message
from impruver.dataset._chat_dataset import ChatDataset

_log: logging.Logger = get_logger()


class TestChatDataset(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant 111"},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing great!"},
            {"role": "user", "content": "Can you help me with something?"},
            {"role": "assistant", "content": "Sure, what do you need help with?"},
        ]
        self.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] }}: {{ message['content'] }}\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "assistant:"
            "{% endif %}"
        )
        self.convert_function = lambda x: [Message.from_dict(m) for m in x["messages"]]
        self.dataset = ChatDataset(
            tokenizer=self.tokenizer,
            source="json",
            split="train",
            convert_function=self.convert_function,
            max_tokens_count=1024,
            data_files="tests/impruver/dataset/_test.jsonl",
        )

    def test_len(self):
        self.assertEqual(len(self.dataset), len(self.dataset._data))

    def test_getitem(self):
        sample = self.dataset[0]
        self.assertIn("input_ids", sample)
        self.assertIn("labels", sample)
        self.assertIn("attention_mask", sample)
        self.assertIsInstance(sample["input_ids"], torch.LongTensor)
        self.assertIsInstance(sample["labels"], torch.LongTensor)
        self.assertIsInstance(sample["attention_mask"], torch.Tensor)

    def test_strategy_function(self):
        dataset = ChatDataset(
            tokenizer=self.tokenizer,
            source="json",
            split="train",
            convert_function=self.convert_function,
            chat_template=self.chat_template,
            strategy_function=last_message_by_assistant,
            max_tokens_count=1024,
            data_files="tests/impruver/dataset/_test.jsonl",
        )
        sample = dataset[0]
        messages = self.convert_function({"messages": self.messages})

        _, _, tokenized_messages = last_message_by_assistant(
            tokenizer=self.tokenizer,
            messages=messages,
            max_tokens_count=1024,
            chat_template=self.chat_template
        )

        self.assertEqual(sample["input_ids"].tolist(), tokenized_messages[0].tolist())

    # TODO: ValueError: Messages must be at least length 2, but got 0 messages
    # def test_prepare_sample_edge_case(self):
    #     sample = self.dataset._prepare_sample({"messages": []})
    #     self.assertEqual(sample["input_ids"].size(0), 0)
    #     self.assertEqual(sample["labels"].size(0), 0)
    #     self.assertEqual(sample["attention_mask"].size(0), 0)


if __name__ == "__main__":
    unittest.main()
