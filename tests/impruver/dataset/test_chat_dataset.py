import unittest
import torch
import logging
from transformers import AutoTokenizer

from impruver.utils import get_logger
from impruver.data._strategies import last_message_by_assistant
from impruver.data._message import Message
from impruver.dataset.chat_dataset import ChatDataset

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
        # messages_filtered = [
        #   Message(role='system', content='You are a helpful assistant 111'),
        #   Message(role='user', content='Hello, how are you?'),
        #   Message(role='assistant', content="I'm doing great!")
        # ]  # 3 of 5 messages in set was filtered by to strategy

        messages_filtered_ids = [
            50256, 10057, 25, 921, 389, 257, 7613, 8796, 13374, 198,
            7220, 25, 18435, 11, 703, 389, 345, 30, 198, 562,
            10167, 25, 314, 1101, 1804, 1049, 0, 198, 50256
        ]

        dataset = ChatDataset(
            tokenizer=self.tokenizer,
            source="json",
            split="train",
            convert_function=self.convert_function,
            chat_template=self.chat_template,
            strategy_function=last_message_by_assistant,
            max_tokens_count=30,
            data_files="tests/impruver/dataset/_test.jsonl",
        )
        sample = dataset[0]
        self.assertEqual(sample["input_ids"].tolist(), messages_filtered_ids)


if __name__ == "__main__":
    unittest.main()
