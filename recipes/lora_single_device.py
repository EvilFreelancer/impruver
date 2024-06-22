import random
import logging
import time

import fire
import torch
from transformers import (
    # AutoTokenizer,
    # AutoModelForCausalLM,
    DataCollatorForTokenClassification,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from impruver.config import Config, log_config
from impruver.utils import dynamic_import, get_dtype, get_device, get_logger, set_seed
from recipes._train_interface import TrainInterface

_log: logging.Logger = get_logger()


class LoraSingleDevice(TrainInterface):
    _device: torch.device
    _dtype: torch.dtype
    _model = None
    _tokenizer = None
    _config: Config
    _dataset = []

    def __init__(self, cfg: Config):
        # Setup configuration
        self._config = cfg

        # Setup precision and device object
        self._device = get_device(self._config.device)
        self._dtype = get_dtype(self._config.dtype)

    def _setup_model_and_tokenizer(self):
        # Load tokenizer
        _tokenizer_cls = dynamic_import(self._config.tokenizer.component)
        _log.debug(f"Tokenizer class is {_tokenizer_cls}")
        self._tokenizer = _tokenizer_cls.from_pretrained(
            self._config.tokenizer.path
        )

        # If quantization is set, then use it
        _quantization_config = None
        if self._config.quantization:
            if self._config.quantization.bnb_4bit_compute_dtype is not None:
                self._config.quantization.bnb_4bit_compute_dtype = get_dtype(
                    self._config.quantization.bnb_4bit_compute_dtype)
            _quantization_config = BitsAndBytesConfig(**self._config.quantization.dict())
            _log.debug(f"Quantization config {_quantization_config}")

        # Load model
        _model_cls = dynamic_import(self._config.model.component)
        _log.debug(f"Model class is {_model_cls}")
        self._model = _model_cls.from_pretrained(
            self._config.model.path,
            quantization_config=_quantization_config,
            torch_dtype=self._dtype,
            attn_implementation=self._config.model.attn_implementation,
        )

        # If unsloth training is enabled
        # if self._config.unsloth:
        #     from unsloth.models._utils import prepare_model_for_kbit_training
        #     self._model = prepare_model_for_kbit_training(self._model)

        # If LoRA training is enabled
        if self._config.lora:
            from peft import get_peft_model, LoraConfig
            _lora_config = LoraConfig(**self._config.lora.dict())
            _log.debug(f"LoRA config {_lora_config}")
            self._model = get_peft_model(self._model, _lora_config)

    def _setup_datasets(self):
        if self._config.seed is not None:
            set_seed(self._config.seed)

        for _dataset in self._config.dataset:
            _dataset_cls = dynamic_import(_dataset.component)
            # TODO: need to support of multiple datasets
            self._dataset = _dataset_cls(
                tokenizer=self._tokenizer,
                source=_dataset.source,
                split=_dataset.split,
            )

        self._data_collator = DataCollatorForTokenClassification(self._tokenizer, pad_to_multiple_of=8)

    def setup(self):
        self._setup_model_and_tokenizer()
        self._setup_datasets()

    def train(self):
        _trainer_config = self._config.trainer.dict()
        _training_args = TrainingArguments(
            output_dir=self._config.output_dir,
            report_to="none",
            **_trainer_config
        )

        # print(_training_args)
        # exit()

        trainer = Trainer(
            model=self._model,
            args=_training_args,
            train_dataset=self._dataset,
            data_collator=self._data_collator,
        )

        # if trainer_config.get("report_to", "wandb") == "wandb":
        #     wandb.init(project="rulm_self_instruct", name=config_file)

        trainer.train()

        # Last step is config saving
        self.save_checkpoint()

    def save_checkpoint(self):
        self._model.save_pretrained(self._config.output_dir)
        self._tokenizer.save_pretrained(self._config.output_dir)


def recipe_main(cfg: str) -> None:
    config = Config.from_yaml(cfg)
    log_config(recipe_name="LoraSingleDevice", cfg=config)

    recipe = LoraSingleDevice(cfg=config)
    recipe.setup()
    recipe.train()
    # recipe.cleanup()


if __name__ == "__main__":
    fire.Fire(recipe_main)
