# Impruver: Фреймворк для обучения Больших Языковых Моделей (LLM)

Набор скриптов и конфигураций для самостоятельного обучения Больших Языковых Моделей (БЯМ) или же на английском Large
Language Models (LLM).

Вдохновлён проектами: [saiga](https://github.com/IlyaGusev/saiga),
[impruver](https://github.com/EvilFreelancer/impruver),
[nanoGPT](https://github.com/karpathy/nanoGPT).

Обладает следующими возможностями:

- Единая конфигурация для подготовки датасетов, запуска обучения и инференса в формате YAML;
- Гибкая система подготовки датасетов, позволяющая скомбинировать несколько датасетов, каждый из них индивидуально
  нарезать и преобразовать, после чего выполнить слияние и дедупликацию;
- Предусмотрены возможности обучения моделей с нуля (from scratch), full-train дообучения и LoRA/Peft дообучения;
- В отличие от иных реализаций использует классы из пакета `transformers`, однако, можно указать путь до любого другого
  класса описывающего модель и/или токенизатор и impruver будет использовать уже их;
- Поддерживает возможность распределённого обучения при помощи `accelerate`.

## Рекомендации

* Python 3.12
* Python Virtual Environment
* Nvidia GPU с 24Гб VRAM (на видеокартах с меньшим объёмом VRAM можно уменьшить `train_batch_size`)
* Драйвера Nvidia и CUDA

## Как установить

Клонируем репозиторий и подготавливаем окружение:

```shell
git clone https://github.com/EvilFreelancer/impruver.git
cd impruver
python3 -m venv venv
. venv/bin/activate
```

Если планируется обучение моделей поддерживающих Flash Attention, то устанавливать нужно так:

```shell
pip install "torch>=2.4.1"
pip install setuptools psutil
pip install "flash-attn>=2.6.3" --no-build-isolation
pip install -r requirements.txt
```

Если будете обучать модель без Flash Attention, то понадобится только это выполнить:

```shell
pip install -r requirements.txt
```

> В планах сделать из `impruver` полноценный пакет, чтобы можно было просто `pip install impruver` делать.

## Доступные конфигурации

В директории [configs](/configs) имеется набор готовых конфигураций, каждый из них оптимизирован для обучения
модели на одной видеокарте, память которой равна 24Гб, хотя маленькие модели можно обучать и на меньших объёмах памяти
просто уменьшая размер `per_device_*_batch_size` и/или `gradient_accumulation_steps`.

| Модель                                                                       | Тип модели                                                                   | Конфигурации                                                                                       |
|------------------------------------------------------------------------------|------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| [ruGPT3.5-13B](https://huggingface.co/ai-forever/ruGPT-3.5-13B)              | Saiga 2                                                                      | [lora](/configs/ruGPT35_13B_lora_saiga2.yaml)                                                      |
| [ruGPT3.5-13B](https://huggingface.co/ai-forever/ruGPT-3.5-13B)              | function call                                                                | [lora](/configs/ruGPT35_13B_lora_fc.yaml)                                                          |
| [nanoGPT](https://github.com/karpathy/nanoGPT) (обучение с нуля)             | Alpaca                                                                       | [full-train](/configs/nanoGPT_full_alpaca.yaml)                                                    |
| [rugpt3large](https://huggingface.co/ai-forever/rugpt3large_based_on_gpt2)   | Saiga 2                                                                      | [full-train](/configs/rugpt3large_full_saiga2.yaml), [lora](/configs/rugpt3large_lora_saiga2.yaml) |
| [rugpt3large](https://huggingface.co/ai-forever/rugpt3large_based_on_gpt2)   | function call                                                                | [full-train](/configs/rugpt3large_full_fc.yaml), [lora](/configs/rugpt3large_lora_fc.yaml)         |
| [rugpt3medium](https://huggingface.co/ai-forever/rugpt3medium_based_on_gpt2) | Saiga 2                                                                      | [full-train](/configs/rugpt3medium_full_saiga2.yaml)                                               |
| [rugpt3medium](https://huggingface.co/ai-forever/rugpt3medium_based_on_gpt2) | function call                                                                | [full-train](/configs/rugpt3medium_full_fc.yaml)                                                   |
| [rugpt3small](https://huggingface.co/ai-forever/rugpt3small_based_on_gpt2)   | Saiga 2                                                                      | [full-train](/configs/rugpt3small_full_saiga2.yaml)                                                |
| [rugpt3small](https://huggingface.co/ai-forever/rugpt3small_based_on_gpt2)   | function call                                                                | [full-train](/configs/rugpt3small_full_fc.yaml)                                                    |
| [zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)        | [zephyr-python-ru](https://huggingface.co/MexIvanov/zephyr-python-ru) analog | [lora](/configs/zephyr_7B_lora_python-ru.yaml)                                                     |

Подробнее о том из чего состоят конфигурации и как их описывать сказано в [документации](/docs/Конфигурация).

## Сборка датасета

Прежде чем приступить к обучению модели необходимо подготовить и дедуплицировать датасет обучения:

```shell
python3 compose_dataset.py configs/ruGPT35_13B_lora_saiga2.yaml
```

## Обучение на одной машине с одной видеокартой

Запускаем обучение модели вот так:

```shell
python3 train_transformers.py configs/ruGPT35_13B_lora_saiga2.yaml
```

Скрипт тренировки поддерживает режим отправки логов в Weights and Biases, но по умолчанию данный функционал отключен,
для того чтобы включить данный функционал нужно добавить опцию `--report-to=wandb`  в команду запуска обучения:

```shell
python3 train_transformers.py configs/ruGPT35_13B_lora_saiga2.yaml --report-to=wandb
```

## Инференс обученной модели

По завершению обучения можно взять интерактивный чат

```shell
python3 infer_transformer.py configs/ruGPT35_13B_lora_saiga2.yaml
```

## Обучение в режиме (D)DP - (Distributed) Data Parallel

Если у вас на сервере несколько видеокарт то потребуется выполнить ряд дополнительных настроек.

Для начала необходимо пройти небольшой опрос, выполни команду:

```shell
accelerate config
```

Пример вопросов и ответов для сервера с тремя видеокартами RTX 4070 Ti.

```
In which compute environment are you running? This machine
Which type of machine are you using? Multi-GPU
How many different machines will you use (in total with all the processes)? 1
Do you wish to use DeepSpeed? No
How many processes in total will you use? 3
Do you wish to use FP16 or BF16 (mixed precision)? bf16
Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). Yes
```

Остальное по умолчанию, просто прожимайте Enter пока не закончится.

Далее в YAML-конфигурацию потребуется добавить стройки вида:

```yaml
ddp:
  ddp_find_unused_parameters: false
```

После этого можно будет запустить обучение:

```shell
accelerate launch train_transformers.py configs/ruGPT35_13B_lora_saiga2.yaml
```

Опция `--report-to=wandb` в таком формате тоже поддерживается.

Далее смотрите в `nvidia-smi` или `nvitop`, модель должна будет запуститься и разлиться на все видеокарты.

## Лицензия

Этот проект лицензирован под лицензией `MIT`. Подробности см. в файле [LICENSE](./LICENSE).

## Цитирование

```
@misc{impruver2024sources,
    author       = {Pavel Rykov},
    title        = {{Impruver: Framework for Training Large Language Models}},
    howpublished = {\url{https://github.com/EvilFreelancer/impruver}},
    year         = {2024}
}
```
