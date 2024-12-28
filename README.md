# Impruver: Фреймворк для обучения Больших Языковых Моделей (LLM)

Набор скриптов и конфигураций для самостоятельного обучения Больших Языковых Моделей (БЯМ) или же на английском Large
Language Models (LLM).

Вдохновлён проектами: [saiga](https://github.com/IlyaGusev/saiga),
[torchtune](https://github.com/pytorch/torchtune),
[nanoGPT](https://github.com/karpathy/nanoGPT).

Обладает следующими возможностями:

- Единая конфигурация для подготовки датасетов, запуска обучения и инференса в формате YAML
    - Возможно указать отдельно токенизатор и модель
    - При необходимости можно описать конфигурацию модели from scratch
- Гибкая система подготовки датасетов, позволяющая скомбинировать несколько датасетов, каждый из них индивидуально
  нарезать и преобразовать, после чего выполнить слияние и дедупликацию
    - Можно использовать датасеты типа `instruct` или `chat` произвольного формата, система преобразует их в
      OpenAI-подобный chat типа messages
    - Можно обучать function call модели с ролями function_call и function_response
- Предусмотрены возможности обучения моделей с нуля (from scratch), full-train дообучения и LoRA/Peft дообучения
- В отличие от иных реализаций использует классы из пакета `transformers`, однако, можно указать путь до любого другого
  класса описывающего модель и/или токенизатор и `impruver` будет использовать уже их
- Поддерживает возможность распределённого обучения при помощи `accelerate`

При необходимости вы можете изучить [документацию](https://github.com/EvilFreelancer/impruver/wiki) проекта.

## Рекомендованные системные требования

* Python 3.12
* Python Virtual Environment
* Nvidia GPU с 24Гб VRAM (на видеокартах с меньшим объёмом VRAM можно уменьшить размер `per_device_*_batch_size`
  и/или `gradient_accumulation_steps`)
* Драйвера Nvidia и CUDA

## Как установить

Устанавливается одной командой:

```shell
pip install impruver
```

После чего в PATH станет доступна утилита командной строки `impruver`.

Если же планируется обучение моделей с использованием Flash Attention, то надо будет выполнить ещё и:

```shell
pip install setuptools psutil torch flash-attn --no-build-isolation
```

## Доступные конфигурации

Полный список рецептов обучения и конфигураций можно посмотреть выполнив:

```shell
impruver ls
```

Вы можете скопировать конфигурацию локально:

```shell
impruver cp ruGPT-3.5/13B_lora_saiga2 ./ruGPT-3.5_13B_lora_saiga2.yaml
```

Подробнее про [конфигурации](https://github.com/EvilFreelancer/impruver/wiki) в wiki проекта.

## Как пользоваться

Прежде чем приступить к обучению модели необходимо подготовить и дедуплицировать датасет, после чего разделить
его на тренировочный и валидационный наборы сэмплов.

Все эти задачи можно выполнить запустив рецепт `compose_dataset` и указав необходимую конфигурацию:

```shell
impruver run compose_dataset --config ./ruGPT-3.5_13B_lora_saiga2.yaml
```

Или используя конфигурацию, идущую в стандартной поставке:

```shell
impruver run compose_dataset --config ruGPT-3.5/13B_lora_saiga2
```

Далее запускаем рецепт `finetune` для обучения трансфорфмерной модели:

```shell
impruver run finetune --config ./ruGPT-3.5_13B_lora_saiga2.yaml
```

Скрипт тренировки поддерживает режим отправки логов в Weights and Biases, но по умолчанию данный функционал отключен,
для того чтобы включить данный функционал нужно добавить опцию `--report-to=wandb` в команду запуска обучения.

По завершению обучения при помощи рецепта `chat` можно запустить интерактивный чат:

```shell
impruver run chat ./ruGPT-3.5_13B_lora_saiga2.yaml
```

## Лицензия

Данный проект распространяется под лицензией `MIT`. Подробности в файле [LICENSE](./LICENSE).

## Цитирование

```
@misc{impruver2024sources,
    author       = {Pavel Rykov},
    title        = {{Impruver: Framework for Training Large Language Models}},
    howpublished = {\url{https://github.com/EvilFreelancer/impruver}},
    year         = {2024}
}
```
