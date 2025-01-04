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

## Рекомендованные системные требования

* Python 3.12
* Python Virtual Environment
* Nvidia GPU с 24Гб VRAM (на видеокартах с меньшим объёмом VRAM можно уменьшить размер `per_device_*_batch_size`
  и/или `gradient_accumulation_steps`)
* Драйвера Nvidia и CUDA

## Доступные конфигурации

В директории [configs](/recipes/configs) имеется набор готовых конфигураций, каждый из них оптимизирован для обучения
модели на одной видеокарте, память которой равна 24Гб, хотя маленькие модели можно обучать и на меньших объёмах памяти
просто уменьшая размер `per_device_*_batch_size` и/или `gradient_accumulation_steps`.

| Модель                                                                | Тип модели                                                                   | Конфигурации                                                                                                                 |
|-----------------------------------------------------------------------|------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| [ruGPT3.5-13B](https://huggingface.co/ai-forever/ruGPT-3.5-13B)       | Saiga 2                                                                      | [lora](/recipes/configs/ruGPT-3.5/13B_lora_saiga2.yaml)                                                                      |
| [ruGPT3.5-13B](https://huggingface.co/ai-forever/ruGPT-3.5-13B)       | function call                                                                | [lora](/recipes/configs/ruGPT-3.5/13B_lora_fc.yaml)                                                                          |
| [nanoGPT](https://github.com/karpathy/nanoGPT) (обучение с нуля)      | Alpaca                                                                       | [full-train](/recipes/configs/nanoGPT/30M_full_alpaca.yaml)                                                                  |
| [rugpt3large](https://huggingface.co/ai-forever/based_on_gpt2)        | Saiga 2                                                                      | [full-train](/recipes/configs/rugpt3large/760M_full_saiga2.yaml), [lora](/recipes/configs/rugpt3large/760M_lora_saiga2.yaml) |
| [rugpt3large](https://huggingface.co/ai-forever/based_on_gpt2)        | function call                                                                | [full-train](/recipes/configs/rugpt3large/760M_full_fc.yaml), [lora](/recipes/configs/rugpt3large/760M_lora_fc.yaml)         |
| [rugpt3medium](https://huggingface.co/ai-forever/based_on_gpt2)       | Saiga 2                                                                      | [full-train](/recipes/configs/rugpt3medium/457M_full_saiga2.yaml)                                                            |
| [rugpt3medium](https://huggingface.co/ai-forever/based_on_gpt2)       | function call                                                                | [full-train](/recipes/configs/rugpt3medium/457M_full_fc.yaml)                                                                |
| [rugpt3small](https://huggingface.co/ai-forever/based_on_gpt2)        | Saiga 2                                                                      | [full-train](/recipes/configs/rugpt3small/125M_full_saiga2.yaml)                                                             |
| [rugpt3small](https://huggingface.co/ai-forever/based_on_gpt2)        | function call                                                                | [full-train](/recipes/configs/rugpt3small/125M_full_fc.yaml)                                                                 |
| [zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) | [zephyr-python-ru](https://huggingface.co/MexIvanov/zephyr-python-ru) analog | [lora](/recipes/configs/zephyr/7B_lora_python-ru.yaml)                                                                       |

Полный список рецептов обучения и конфигураций можно посмотреть выполнив:

```shell
impruver ls
```

Вы можете скопировать конфигурацию локально:

```shell
impruver cp ruGPT-3.5/13B_lora_saiga2 ./ruGPT-3.5_13B_lora_saiga2.yaml
```
