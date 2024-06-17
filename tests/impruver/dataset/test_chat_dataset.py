import unittest
import torch
from transformers import AutoTokenizer

from impruver.data._strategies import last_message_by_assistant
from impruver.data._message import Message
from impruver.dataset._chat_dataset import ChatDataset


class TestChatDataset(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing great!"},
            {"role": "user", "content": "Can you help me with something?"},
            {"role": "assistant", "content": "Sure, what do you need help with?"},
        ]

        self.convert_function = lambda x: [Message.from_dict(m) for m in x["messages"]]

        print(self.convert_function)
        exit()

        self.dataset = ChatDataset(
            tokenizer=self.tokenizer,
            source="json",
            convert_function=self.convert_function,
            max_tokens_count=1024,
            data_files="tests/impruver/datasets/_test.json",
        )

#     def test_len(self):
#         self.assertEqual(len(self.dataset), len(self.dataset._data))
#
#     def test_getitem(self):
#         sample = self.dataset[0]
#         self.assertIn("input_ids", sample)
#         self.assertIn("labels", sample)
#         self.assertIn("attention_mask", sample)
#         self.assertIsInstance(sample["input_ids"], torch.LongTensor)
#         self.assertIsInstance(sample["labels"], torch.LongTensor)
#         self.assertIsInstance(sample["attention_mask"], torch.Tensor)
#
#     def test_format_function(self):
#         def format_function(messages):
#             for message in messages:
#                 message.content = message.content.upper()
#             return messages
#
#         dataset = ChatDataset(
#             tokenizer=self.tokenizer,
#             source="json",
#             convert_function=self.convert_function,
#             format_function=format_function,
#             max_tokens_count=1024,
#             data_files="impruver/datasets/test.json",
#         )
#         sample = dataset[0]
#         for msg in sample["input_ids"]:
#             self.assertTrue(all(c.isupper() for c in self.tokenizer.decode(msg).split() if c.isalpha()))
#
#     def test_strategy_function(self):
#         dataset = ChatDataset(
#             tokenizer=self.tokenizer,
#             source="json",
#             convert_function=self.convert_function,
#             strategy_function=last_message_by_assistant,
#             max_tokens_count=1024,
#             data_files="impruver/datasets/test.json",
#         )
#         sample = dataset[0]
#         messages = self.convert_function({"messages": self.messages})
#         _, _, tokenized_messages = last_message_by_assistant(messages, [m.content for m in messages],
#                                                              max_tokens_count=1024)
#         self.assertEqual(sample["input_ids"].tolist(), tokenized_messages.tolist())
#
#     def test_prepare_sample_edge_case(self):
#         sample = self.dataset._prepare_sample({"messages": []})
#         self.assertEqual(sample["input_ids"].size(0), 0)
#         self.assertEqual(sample["labels"].size(0), 0)
#         self.assertEqual(sample["attention_mask"].size(0), 0)
#
#
# if __name__ == "__main__":
#     unittest.main()
