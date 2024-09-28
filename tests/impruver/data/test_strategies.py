import unittest
from transformers import AutoTokenizer
from impruver.data._message import Message
from impruver.data._strategies import last_message_by_assistant


class TestStrategies(unittest.TestCase):
    def setUp(self):
        model = "openai-community/gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.messages = [
            Message.from_dict({"role": "system", "content": "You are a helpful assistant 111"}),
            Message.from_dict({"role": "user", "content": "Hello, how are you?"}),
            Message.from_dict({"role": "assistant", "content": "I'm doing great!"}),
            Message.from_dict({"role": "user", "content": "Can you help me with something?"}),
            Message.from_dict({"role": "assistant", "content": "Sure, what do you need help with?"}),
        ]
        self.max_tokens_count = 30

    #
    def test_last_message_by_assistant(self):
        results_messages = last_message_by_assistant(
            tokenizer=self.tokenizer,
            messages=self.messages,
            chat_template=None,
            max_tokens_count=self.max_tokens_count,
        )

        # 3 of 5 messages should pass a filter in 30 tokens
        self.assertEqual(len(results_messages), 3)

        # Verify the last message is from the assistant
        self.assertEqual(results_messages[-1].role, 'assistant')

        # Print the results for manual inspection
        print("===== Original Messages =====")
        for msg in self.messages:
            print(f"{msg.role}: {msg.content}")

        print("===== Filtered Messages =====")
        for msg in results_messages:
            print(f"{msg.role}: {msg.content}")
