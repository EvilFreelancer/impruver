# Impruver: Framework for Training Large Language Models (LLMs)

[Русский](./README.md) | [中文](./README.zh.md) | **English**

A set of scripts and configurations for training Large Language Models (LLMs) independently.

Inspired by projects like [saiga](https://github.com/IlyaGusev/saiga),
[torchtune](https://github.com/pytorch/torchtune)
and [nanoGPT](https://github.com/karpathy/nanoGPT).

Features:

- Unified configuration in YAML format for dataset preparation, training, and inference
    - Allows specifying a tokenizer and model separately
    - Supports training models from scratch, full training, and LoRA/Peft fine-tuning
- Flexible dataset preparation system that enables combining multiple datasets, individually slicing and transforming
  each, and then merging and deduplicating them
    - Supports datasets in `instruct` or `chat` formats, converting them into OpenAI-style chat message formats
    - Enables training function call models with `function_call` and `function_response` roles
- Unlike other implementations, it uses classes from the `transformers` library. However, you can specify any other
  class for the model and/or tokenizer, and `impruver` will use them
- Supports distributed training using `accelerate`

For more details, check the project's [documentation](https://github.com/EvilFreelancer/impruver/wiki).

## Requirements

* Python 3.12
* Python Virtual Environment
* Nvidia GPU with 24GB VRAM (for GPUs with less VRAM, you can reduce the values of `per_device_*_batch_size`
  and/or `gradient_accumulation_steps`)
* Nvidia drivers and CUDA

## Installation

Install with a single command:

```shell
pip install impruver
```

This will add the `impruver` CLI utility to your PATH.

If you plan to train models using Flash Attention, also run:

```shell
pip install setuptools psutil torch flash-attn --no-build-isolation
```

## Available Configurations

Get a full list of training recipes and configurations by running:

```shell
impruver ls
```

You can copy a configuration locally:

```shell
impruver cp ruGPT-3.5/13B_lora_saiga2 ./ruGPT-3.5_13B_lora_saiga2.yaml
```

Learn more about [configurations](https://github.com/EvilFreelancer/impruver/wiki) in the project wiki.

## Usage

Before training a model, prepare and deduplicate the dataset, then split it into training and validation sample sets.

These tasks can be performed using the `compose_dataset` recipe with the specified configuration:

```shell
impruver run compose_dataset --config ./ruGPT-3.5_13B_lora_saiga2.yaml
```

Or by using a configuration from the default set:

```shell
impruver run compose_dataset --config ruGPT-3.5/13B_lora_saiga2
```

Next, run the `finetune` recipe to train the transformer model:

```shell
impruver run finetune --config ./ruGPT-3.5_13B_lora_saiga2.yaml
```

The training script supports logging to Weights and Biases (W&B). By default, this is disabled, but you can enable it by
adding the `--report-to=wandb` option to the training command.

Once training is complete, you can launch an interactive chat session using the `chat` recipe:

```shell
impruver run chat ./ruGPT-3.5_13B_lora_saiga2.yaml
```

To exit the chat shell, use the `Ctrl+D` or `Ctrl+C` keyboard shortcuts.

## License

This project is distributed under the MIT license. See the [LICENSE](./LICENSE) file for details.

## Citation

```
@misc{impruver2024sources,
    author       = {Pavel Rykov},
    title        = {{Impruver: Framework for Training Large Language Models}},
    howpublished = {\url{https://github.com/EvilFreelancer/impruver}},
    year         = {2024}
}
```
