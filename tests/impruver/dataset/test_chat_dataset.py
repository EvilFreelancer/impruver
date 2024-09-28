import unittest
import torch
from typing import List, Dict, Any
from tiktoken import get_encoding
from dataclasses import dataclass
from impruver.dataset import ChatDataset
from impruver.data import apply_chat_template


@dataclass
class Message:
    role: str
    content: str


class TiktokenTokenizer:
    def __init__(self, encoding_name: str = "gpt2"):
        self.encoder = get_encoding(encoding_name)
        self.bos_token_id = self.encoder.encode("<|startoftext|>", allowed_special="all")[0]
        self.eos_token_id = self.encoder.encode("<|endoftext|>", allowed_special="all")[0]
        self.pad_token_id = self.encoder.encode("<|padding|>", allowed_special="all")[0]  # Assuming a pad token
        self.model_max_length = 2048  # Example max length

    def apply_chat_template(
            self,
            messages: List[Dict[str, str]],
            chat_template: str = None,
            add_special_tokens: bool = False,
            tokenize: bool = True,
            add_generation_prompt: bool = False,
    ):
        # Simple implementation for testing purposes
        # Concatenate messages with roles for the template
        text = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            text += f"{role}: {content}\n"

        if add_special_tokens:
            text = "<|startoftext|>" + text + "<|endoftext|>"

        if tokenize:
            return self.encoder.encode(text, allowed_special="all")
        else:
            return text

    def decode(self, tokens, skip_special_tokens=False):
        text = self.encoder.decode(tokens)
        if skip_special_tokens:
            text = text.replace("<|startoftext|>", "").replace("<|endoftext|>", "")
        return text


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
        self.tokenizer = TiktokenTokenizer()

        # Initialize the dataset
        self.dataset = ChatDataset(
            original_records=self.sample_records,
            tokenizer=self.tokenizer,
            max_tokens_count=50,
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

        print(input_text)
        exit()

        # Expected input text
        expected_input_text = apply_chat_template(
            self.sample_records[0]["messages"],
            add_special_tokens=False,
            tokenize=False,
        )

        # Check that the input text matches the expected text
        self.assertEqual(input_text.strip(), expected_input_text.strip())

        # Check labels
        # Since only_target_loss is True, labels for user messages should be -100
        # and labels for assistant messages should be the token ids
        # Let's reconstruct what labels should be
        messages = self.sample_records[0]["messages"]
        input_ids_list = []
        expected_labels = []
        for message in messages:
            message_input_ids = self.tokenizer.apply_chat_template(
                [message],
                add_special_tokens=True,
                tokenize=True,
                add_generation_prompt=False,
            )
            if message["role"] in ("assistant", "bot", "gpt"):
                expected_labels.extend(message_input_ids)
            else:
                expected_labels.extend([-100] * len(message_input_ids))
            input_ids_list.extend(message_input_ids)

        # Add global BOS and EOS tokens if specified
        if self.dataset.add_global_bos:
            input_ids_list = [self.tokenizer.bos_token_id] + input_ids_list
            expected_labels = [-100] + expected_labels
        if self.dataset.add_global_eos:
            input_ids_list.append(self.tokenizer.eos_token_id)
            expected_labels.append(self.tokenizer.eos_token_id)

        # Convert to tensors
        expected_input_ids = torch.LongTensor(input_ids_list)
        expected_labels = torch.LongTensor(expected_labels)

        # Check that input_ids and labels match expected values
        self.assertTrue(torch.equal(input_ids, expected_input_ids))
        self.assertTrue(torch.equal(labels, expected_labels))

        # Check attention_mask (should be all ones in this case)
        expected_attention_mask = torch.ones_like(input_ids)
        self.assertTrue(torch.equal(attention_mask, expected_attention_mask))


# Run the tests
if __name__ == '__main__':
    unittest.main()
