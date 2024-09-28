import unittest
import torch
from transformers import AutoTokenizer

from impruver.dataset import ChatDataset
from impruver.data import apply_chat_template


# Now, we write the test class using unittest.TestCase
class TestChatDataset(unittest.TestCase):
    def setUp(self):
        # Sample data
        self.sample_records = [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing great!"},
                    {"role": "user", "content": "Can you help me with something?"},
                    {"role": "assistant", "content": "Sure, what do you need help with?"}
                ]
            }
        ]

        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')

        # Initialize the dataset
        self.dataset = ChatDataset(
            original_records=self.sample_records,
            tokenizer=self.tokenizer,
            max_tokens_count=100,
            only_target_loss=True,
            add_global_bos=True,
            add_global_eos=True,
            labels_pad_token_id=-100,
            convert_function=None,
            strategy_function=None,
            chat_template=None,
        )

    def test_dataset_length(self):
        self.assertEqual(len(self.dataset), 1)

    def test_sample_output(self):
        sample = self.dataset[0]
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        attention_mask = sample["attention_mask"]

        # Check that input_ids, labels, and attention_mask are tensors
        self.assertIsInstance(input_ids, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertIsInstance(attention_mask, torch.Tensor)

        # Check that the lengths of input_ids, labels, and attention_mask are equal
        self.assertEqual(len(input_ids), len(labels))
        self.assertEqual(len(input_ids), len(attention_mask))

        # Decode input_ids to get the input text
        input_text = self.tokenizer.decode(input_ids.tolist(), skip_special_tokens=True)

        # Expected input text
        expected_input_text = apply_chat_template(
            self.sample_records[0]["messages"],
            add_special_tokens=False,
            tokenize=False,
        )

        # Check that the input text matches the expected text
        self.assertEqual(input_text.strip(), expected_input_text.strip())


# Run the tests
if __name__ == '__main__':
    unittest.main()
