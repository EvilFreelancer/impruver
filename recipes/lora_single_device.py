import random

import fire
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForTokenClassification,
)

from impruver.config import Config, log_config
from impruver.utils import dynamic_import, get_dtype, get_device
from recipes._train_interface import TrainInterface


# from transformers import (
#     Trainer,
#     TrainingArguments,
#     logging,
#     TrainerCallback,
#     TrainerState,
#     TrainerControl,
#     BitsAndBytesConfig,
# )
# from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
# from unsloth.models._utils import prepare_model_for_kbit_training


class LoraSingleDevice(TrainInterface):
    _device: torch.device
    _dtype: torch.dtype
    _model = None
    _tokenizer = None
    _config: Config

    def __init__(self, cfg: Config):
        # Setup configuration
        self._config = cfg

        # Setup precision and device object
        self._device = get_device(self._config.device)
        self._dtype = get_dtype(self._config.dtype)

    def _setup_model_and_tokenizer(self):
        # Load tokenizer
        _tokenizer_cls = dynamic_import(self._config.tokenizer.component)
        self._tokenizer = _tokenizer_cls.from_pretrained(
            self._config.tokenizer.path
        )

        # Load model
        _model_cls = dynamic_import(self._config.model.component)
        self._model = _model_cls.from_pretrained(
            self._config.model.path,
            load_in_8bit=self._config.model.load_in_8bit,
            load_in_4bit=self._config.model.load_in_4bit,
            torch_dtype=self._dtype,
            attn_implementation=self._config.model.attn_implementation,
        ).to(self._device)

        # If unsloth training is enabled
        # if self._config.unsloth:
        #     from unsloth.models._utils import prepare_model_for_kbit_training
        #     self._model = prepare_model_for_kbit_training(self._model)

        # If LoRA training is enabled
        if self._config.lora:
            from peft import get_peft_model, LoraConfig

            print(self._config.lora.export())
            exit()

            _lora_config = LoraConfig(**self._config.lora.model_dump())

            print(_lora_config)
            exit()

            self._model = get_peft_model(self._model, lora_config)

    def _setup_datasets(self):
        train_file = self._config.dataset.train_file
        val_file = self._config.dataset.val_file
        max_tokens_count = self._config.dataset.max_tokens_count
        only_target_loss = self._config.dataset.only_target_loss
        sample_rate = self._config.dataset.sample_rate

        # train_records = read_jsonl(train_file)
        # val_records = read_jsonl(val_file)
        # random.shuffle(train_records)
        #
        # datasets = []
        # for records in (train_records, val_records):
        #     datasets.append(
        #         ChatDataset(
        #             records,
        #             self._tokenizer,
        #             max_tokens_count=max_tokens_count,
        #             sample_rate=sample_rate,
        #             only_target_loss=only_target_loss,
        #         )
        #     )
        # self._train_dataset, self._val_dataset = datasets
        # self._data_collator = DataCollatorForTokenClassification(self._tokenizer, pad_to_multiple_of=8)

    def setup(self):
        self._setup_model_and_tokenizer()
        # self._setup_datasets()

    # def train(self):
    #     ...
    #     # Last step is config saving
    #     # self.save_checkpoint()
    #
    # def cleanup(self):
    #     ...
    #
    # def save_checkpoint(self):
    #     self._model.save_pretrained(self._config.output_dir)
    #     self._tokenizer.save_pretrained(self._config.output_dir)


def recipe_main(cfg: str) -> None:
    config = Config.from_yaml(cfg)
    log_config(recipe_name="LoraSingleDevice", cfg=config)

    recipe = LoraSingleDevice(cfg=config)
    recipe.setup()
    # recipe.train()
    # recipe.cleanup()


if __name__ == "__main__":
    fire.Fire(recipe_main)
