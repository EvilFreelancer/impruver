# impruver

Прототип проекта для обучения своих больших языковых моделей (LLM) с гибкой системой подготовки датасетов.

## Как установить

Клонируем репозиторий и подготавливаем окружение:

```shell
git clone https://github.com/EvilFreelancer/impruver.git
cd impruver
python3 -m venv venv
. venv/bin/activate
```

Если планируется обучение моделей, поддерживающих Flash Attention, то сначала это:

```shell
pip install "torch>=2.4.1"
pip install setuptools psutil
pip install "flash-attn>=2.6.3" --no-build-isolation
pip install -r requirements.txt
pip install .
```

Если будете обучать модель без Flash Attention то понадобится только это выполнить: 

```shell
pip install -r requirements.txt
pip install .
```

(тут ещё пока всё сложно, я планирую сделать из impruver полноценный пакет, который можно будет установить через pip)

## Сборка датасета

Далее собираем датасет:

```shell
python3 compose_dataset.py configs/ruGPT35_13B_lora.yaml
```

## Обучение на одной машине с одной видеокартой 

Запускаем обучение модели вот так:

```shell
python3 train_transformers.py configs/ruGPT35_13B_lora.yaml
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
accelerate launch train_transformers.py configs/ruGPT35_13B_lora.yaml
```

Далее смотрите в `nvidia-smi` или `nvitop`, модель должна будет запуститься и разлиться на все видеокарты.

---

Про loss функцию:

![](https://b2633864.smushcdn.com/2633864/wp-content/uploads/2019/08/keras_learning_rate_finder_header.png?lossy=2&strip=1&webp=1)
