import logging
import re
from typing import List, Union, Optional, Literal
import inspect
from pydantic import BaseModel, ValidationError, Field, create_model
from ruamel.yaml import YAML

from .utils import get_logger

_log: logging.Logger = get_logger()


def optional(*fields):
    def dec(cls):
        fields_dict = {}
        for field in fields:
            field_info = cls.__annotations__.get(field)
            if field_info is not None:
                fields_dict[field] = (Optional[field_info], None)
        OptionalModel = create_model(cls.__name__, **fields_dict)
        OptionalModel.__module__ = cls.__module__

        return OptionalModel

    if fields and inspect.isclass(fields[0]) and issubclass(fields[0], BaseModel):
        cls = fields[0]
        fields = cls.__annotations__
        return dec(cls)

    return dec


class TokenizerConfig(BaseModel):
    component: str = Field(..., alias='_component_')
    path: str


class ModelConfig(BaseModel):
    component: str = Field(..., alias='_component_')
    path: str
    attn_implementation: Optional[str] = None


class DatasetConfig(BaseModel):
    component: str = Field(..., alias='_component_')
    source: Optional[str] = None
    split: Optional[str] = None
    max_tokens_count: Optional[int] = None


class OptimizerConfig(BaseModel):
    component: str = Field(..., alias='_component_')
    lr: float


class QuantizationConfig(BaseModel):
    load_in_8bit: Optional[bool] = False
    load_in_4bit: Optional[bool] = False
    llm_int8_threshold: Optional[float] = 6.0
    llm_int8_skip_modules: Optional[str] = None
    llm_int8_enable_fp32_cpu_offload: Optional[bool] = False
    llm_int8_has_fp16_weight: Optional[bool] = False
    bnb_4bit_compute_dtype: Optional[str] = None
    bnb_4bit_quant_type: Optional[str] = "fp4"
    bnb_4bit_use_double_quant: Optional[bool] = False
    bnb_4bit_quant_storage: Optional[str] = None


class LoraConfig(BaseModel):
    r: Optional[int] = 8
    target_modules: Optional[Union[list[str], str]] = None
    lora_alpha: Optional[int] = 8
    lora_dropout: Optional[float] = 0.0
    bias: Optional[Literal["none", "all", "lora_only"]] = "none"
    modules_to_save: Optional[list[str]] = None


class LossConfig(BaseModel):
    component: str = Field(..., alias='_component_')


class MetricLoggerConfig(BaseModel):
    component: str = Field(..., alias='_component_')
    log_dir: Optional[str] = None
    filename: Optional[str] = None


class TrainerConfig(BaseModel):
    eval_strategy: str = "steps"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 128
    eval_steps: int = 8
    save_steps: int = 8
    logging_steps: int = 1
    learning_rate: float = 0.00005
    num_train_epochs: int = 2
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 8
    bf16: bool = False
    fp16: bool = False
    optim: str = "adamw_8bit"
    load_best_model_at_end: bool = True
    save_total_limit: int = 1
    seed: int = 42
    max_grad_norm: float = 1.0
    weight_decay: float = 0.05


class Config(BaseModel):
    tokenizer: TokenizerConfig
    model: ModelConfig
    compile: bool = False
    lora: Optional[LoraConfig] = None
    quantization: Optional[QuantizationConfig] = None
    # unsloth: Optional[bool] = False

    dataset: List[DatasetConfig]
    seed: Optional[int] = None
    shuffle: bool = False

    trainer: Optional[TrainerConfig] = None
    # optimizer: Optional[OptimizerConfig]
    # loss: Optional[LossConfig]
    # batch_size: int = 2
    # epochs: int = 3
    # max_steps_per_epoch: int | None = None
    # gradient_accumulation_steps: int = 1
    # resume_from_checkpoint: bool = False
    # optimizer_in_bwd: bool = False

    device: str = 'cuda'
    enable_activation_checkpointing: bool = True

    dtype: str = 'bf16'

    # metric_logger: Optional[MetricLoggerConfig]
    output_dir: Optional[str] = None

    # log_every_n_steps: int = 1
    # log_peak_memory_stats: bool = False

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    @classmethod
    def from_yaml(cls, path: str):
        config_data = load_yaml_with_references(path)
        if isinstance(config_data.get('dataset'), dict):
            config_data['dataset'] = [config_data['dataset']]
        _log.info(config_data)
        return cls(**config_data)

    def to_yaml(self, path: Optional[str] = None):
        yaml = YAML()
        config_data = self.dict(by_alias=True)
        if path is not None:
            with open(path, 'w') as file:
                yaml.dump(config_data, file)


def load_yaml_with_references(path):
    yaml = YAML()
    with open(path, 'r') as file:
        data = yaml.load(file)

    def resolve_references(node, references):
        if isinstance(node, dict):
            for key, value in node.items():
                node[key] = resolve_references(value, references)
        elif isinstance(node, list):
            return [resolve_references(item, references) for item in node]
        elif isinstance(node, str):
            matches = re.findall(r'\$\{(\w+)\}', node)
            for match in matches:
                if match in references:
                    node = node.replace(f"${{{match}}}", references[match])
        return node

    return resolve_references(data, data)


def log_config(recipe_name: str, cfg: Config) -> None:
    _log.info(msg=f"Running {recipe_name} with resolved config:\n\n{cfg.to_yaml()}")


if __name__ == "__main__":
    try:
        config = Config.from_yaml(path="config.yaml")
        print(config)
    except ValidationError as e:
        print(e)
