# impruver

Прототип проекта для обучения своих больших языковых моделей (LLM) с гибкой системой подготовки датасетов.

## Как использовать

Клонируем репозиторий и подготавливаем окружение:

```shell
git clone https://github.com/EvilFreelancer/impruver.git
cd impruver
python3 -m venv venv
. venv/bin/activate
```

Далее ставим пакеты:

```shell
pip install "torch>=2.4.1"
pip install "flash-attn>=2.6.3" --no-build-isolation
pip install -r requirements.txt
pip install .
```

(тут ещё пока всё сложно, я планирую сделать из impruver полноценный пакет, который можно будет установить через pip)

Далее собираем датасет:

```shell
python3 compose_dataset.py configs/rugpt3small_based_on_gpt2.yaml ./tr.jsonl ./val.jsonl
```

Запускаем обучение модели:

```shell
python3 train_transformers.py configs/rugpt3small_based_on_gpt2.yaml ./tr.jsonl ./val.jsonl ./output
```

---

Про loss функцию:

![](https://b2633864.smushcdn.com/2633864/wp-content/uploads/2019/08/keras_learning_rate_finder_header.png?lossy=2&strip=1&webp=1)
