import unittest
from unittest.mock import patch, mock_open

from pydantic import ValidationError
from ruamel.yaml import YAML

from impruver.config import Config, load_yaml_with_references, log_config
from impruver.utils import get_logger

_log = get_logger()


class TestConfig(unittest.TestCase):

    def setUp(self):
        self.valid_yaml = """
        tokenizer:
            _component_: 'tokenizer_component'
            path: 'path/to/tokenizer'
        model:
            _component_: 'model_component'
            path: 'path/to/model'
            attn_implementation: 'scaled_dot_product'
            load_in_4bit: False
            load_in_8bit: False
        compile: False
        dataset:
          - _component_: 'dataset_component'
            train_on_input: True
        seed: 42
        shuffle: True
        optimizer:
            _component_: 'optimizer_component'
            lr: 0.001
        loss:
            _component_: 'loss_component'
        batch_size: 32
        epochs: 10
        max_steps_per_epoch: 1000
        gradient_accumulation_steps: 1
        resume_from_checkpoint: False
        optimizer_in_bwd: False
        device: 'cuda'
        enable_activation_checkpointing: True
        dtype: 'bf16'
        metric_logger:
            _component_: 'metric_logger_component'
            log_dir: 'logs'
            filename: 'logfile.log'
        output_dir: 'output'
        log_every_n_steps: 10
        log_peak_memory_stats: False
        """
        self.invalid_yaml = """
        tokenizer:
            _component_: 'tokenizer_component'
        model:
            _component_: 'model_component'
            path: 'path/to/model'
            attn_implementation: 'scaled_dot_product'
            load_in_4bit: False
            load_in_8bit: False
        compile: False
        dataset:
          - _component_: 'dataset_component'
            train_on_input: True
        seed: 42
        shuffle: True
        optimizer:
            _component_: 'optimizer_component'
            lr: 'not_a_float'
        loss:
            _component_: 'loss_component'
        batch_size: 32
        epochs: 10
        max_steps_per_epoch: 1000
        gradient_accumulation_steps: 1
        resume_from_checkpoint: False
        optimizer_in_bwd: False
        device: 'cuda'
        enable_activation_checkpointing: True
        dtype: 'bf16'
        metric_logger:
            _component_: 'metric_logger_component'
            log_dir: 'logs'
            filename: 'logfile.log'
        output_dir: 'output'
        log_every_n_steps: 10
        log_peak_memory_stats: False
        """

    def test_valid_config_loading(self):
        with patch("builtins.open", mock_open(read_data=self.valid_yaml)):
            config = Config.from_yaml("config.yaml")
            self.assertEqual(config.tokenizer.component, 'tokenizer_component')
            self.assertEqual(config.model.path, 'path/to/model')
            self.assertEqual(config.optimizer.lr, 0.001)

    def test_invalid_config_loading(self):
        with patch("builtins.open", mock_open(read_data=self.invalid_yaml)):
            with self.assertRaises(ValidationError):
                Config.from_yaml("config.yaml")

    def test_yaml_references_resolution(self):
        yaml_with_refs = """
        var: &var_value "value"
        some_key: "ref: ${var}"
        """
        expected_result = {
            "var": "value",
            "some_key": "ref: value"
        }
        with patch("builtins.open", mock_open(read_data=yaml_with_refs)):
            resolved_data = load_yaml_with_references("dummy_path")
            self.assertEqual(resolved_data, expected_result)



if __name__ == "__main__":
    unittest.main()
