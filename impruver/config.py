import logging
import re
from typing import List, Union, Optional

from pydantic import BaseModel, ValidationError, Field
from ruamel.yaml import YAML

from .utils import get_logger

_log: logging.Logger = get_logger()


class TokenizerConfig(BaseModel):
    component: str = Field(..., alias='_component_')
    path: str


class ModelConfig(BaseModel):
    component: str = Field(..., alias='_component_')
    path: str
    attn_implementation: str | None = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False


class DatasetConfig(BaseModel):
    component: str = Field(..., alias='_component_')
    train_on_input: bool


class OptimizerConfig(BaseModel):
    component: str = Field(..., alias='_component_')
    lr: float


class LossConfig(BaseModel):
    component: str = Field(..., alias='_component_')


class MetricLoggerConfig(BaseModel):
    component: str = Field(..., alias='_component_')
    log_dir: Optional[str] = None
    filename: Optional[str] = None


class Config(BaseModel):
    tokenizer: TokenizerConfig
    model: ModelConfig
    compile: bool = False

    dataset: Union[DatasetConfig, List[DatasetConfig]]
    seed: Optional[int] = None
    shuffle: bool = True

    optimizer: OptimizerConfig
    loss: LossConfig
    batch_size: int = 2
    epochs: int = 3
    max_steps_per_epoch: int | None = None
    gradient_accumulation_steps: int = 1
    resume_from_checkpoint: bool = False
    optimizer_in_bwd: bool = False

    device: str = 'cuda'
    enable_activation_checkpointing: bool = True

    dtype: str = 'bf16'

    metric_logger: MetricLoggerConfig

    output_dir: str
    log_every_n_steps: int = 1
    log_peak_memory_stats: bool = False

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    @classmethod
    def from_yaml(cls, path: str):
        config_data = load_yaml_with_references(path)
        if isinstance(config_data.get('dataset'), dict):
            config_data['dataset'] = [config_data['dataset']]
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
