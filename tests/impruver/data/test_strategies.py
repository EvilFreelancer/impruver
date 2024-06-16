import unittest
from transformers import AutoTokenizer
from impruver.data._message import Message
from impruver.data._strategies import last_message_by_assistant


class TestStrategies(unittest.TestCase):
    def setUp(self):
        model = "openai-community/gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello, how are you?"),
            Message(role="assistant", content="I'm doing great!"),
            Message(role="user", content="Hello, how are you?"),
            Message(role="assistant", content="I'm doing great!"),
        ]
        self.max_tokens_count = 40

    #
    def test_last_message_by_assistant(self):
        results_messages, formated_messages, tokenized_messages = last_message_by_assistant(
            tokenizer=self.tokenizer,
            messages=self.messages,
            chat_template=None,
            max_tokens_count=self.max_tokens_count,
        )

        # Verify the last message is from the assistant
        self.assertEqual(results_messages[-1].role, 'assistant')

        # Verify the total tokens do not exceed the max_tokens_count
        total_tokens = sum(len(tokens) for tokens in tokenized_messages)
        self.assertLessEqual(total_tokens, self.max_tokens_count)

        # Print the results for manual inspection
        print("Filtered Messages:")
        for msg in results_messages:
            print(f"{msg.role}: {msg.content}")

        print("Tokenized Messages:")
        print(tokenized_messages)
