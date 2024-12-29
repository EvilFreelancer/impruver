# Impruver：大语言模型（LLM）训练框架

[Русский](./README.md) | **中文** | [English](./README.en.md)

一套用于自行训练大语言模型（LLM）的脚本和配置。

灵感来源于以下项目：[saiga](https://github.com/IlyaGusev/saiga),
[torchtune](https://github.com/pytorch/torchtune),
[nanoGPT](https://github.com/karpathy/nanoGPT).

具有以下功能：

- 使用统一的 YAML 配置文件完成数据集准备、模型训练和推理的全过程
    - 可分别指定分词器和模型
    - 可根据需要定义从零开始的模型配置
- 灵活的数据集准备系统，支持组合多个数据集，并对每个数据集分别进行切片、转换，然后进行合并和去重
    - 支持`instruct`或`chat`任意格式的数据集，系统会将其转换为类似OpenAI的chat格式（messages） 
    - 可以训练支持`function_call`和`function_response`角色的函数调用模型
- 与其他实现不同，使用`transformers`包中的类，但可以指定其他自定义的模型和/或分词器类路径，`impruver`将自动使用这些类
- 支持使用`accelerate`进行分布式训练

您可以参考项目的[文档](https://github.com/EvilFreelancer/impruver/wiki)以获取更多信息。

## 推荐的系统要求

* Python 3.12
* Python虚拟环境
* 配备24GB显存的Nvidia GPU（对于显存较小的显卡，可适当减小`per_device_*_batch_size`和/或`gradient_accumulation_steps`的值）
* Nvidia驱动程序和CUDA

## 安装方法

只需运行以下命令即可完成安装：

```shell
pip install impruver
```

安装完成后，`impruver`命令行工具将会加入PATH。

如果需要使用`Flash Attention`进行模型训练，还需要执行以下命令：

```shell
pip install setuptools psutil torch flash-attn --no-build-isolation
```

## 可用配置

执行以下命令可查看所有训练配方和配置列表：

```shell
impruver ls
```

您可以将配置复制到本地：

```shell
impruver cp ruGPT-3.5/13B_lora_saiga2 ./ruGPT-3.5_13B_lora_saiga2.yaml
```
更多关于[配置](https://github.com/EvilFreelancer/impruver/wiki)的信息，请参阅项目Wiki。

## 使用说明

在开始训练模型之前，需要准备并去重数据集，然后将其分为训练集和验证集。

可以通过执行`compose_dataset`配方并指定配置文件来完成这些任务：

```shell
impruver run compose_dataset --config ./ruGPT-3.5_13B_lora_saiga2.yaml
```

或者使用自带的标准配置文件：

```shell
impruver run compose_dataset --config ruGPT-3.5/13B_lora_saiga2
```

接下来运行`finetune`配方以训练Transformer模型：

```shell
impruver run finetune --config ./ruGPT-3.5_13B_lora_saiga2.yaml
```

训练脚本支持将日志发送到Weights and Biases，但默认情况下是禁用的。要启用该功能，请在训练命令中添加`--report-to=wandb`选项。

训练完成后，可通过`chat`配方启动交互式聊天：

```shell
impruver run chat ./ruGPT-3.5_13B_lora_saiga2.yaml
```

要退出聊天界面，请使用`Ctrl+D`或`Ctrl+C`快捷键。

## 许可证

本项目以`MIT`许可证发布。详细信息见[LICENSE](./LICENSE)文件。

## 引用

```
@misc{impruver2024sources,
    author       = {Pavel Rykov},
    title        = {{Impruver: Framework for Training Large Language Models}},
    howpublished = {\url{https://github.com/EvilFreelancer/impruver}},
    year         = {2024}
}
```
