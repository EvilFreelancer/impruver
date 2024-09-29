# converter

Функция конвертации отдельного элемента необходима для того, чтобы привести все входящие данные к общему виду.

Например, у нас имеется датасет на HuggingFace со стриктурой вида:

```json lines
{"conversation":[{"from":"human","value":"text"},{"from":"gpt","value":"text"}]}
{"conversation":[{"from":"human","value":"text"},{"from":"gpt","value":"text"}]}
{"conversation":[{"from":"human","value":"text"},{"from":"gpt","value":"text"}]}
```

А класс `ChatDataset` может работать только со следующей структурой данных:

```json lines
{"messages":[{"role":"user","value":"text"},{"role":"assistant","value":"text"}]}
{"messages":[{"role":"user","value":"text"},{"role":"assistant","value":"text"}]}
{"messages":[{"role":"user","value":"text"},{"role":"assistant","value":"text"}]}
```

Для того чтобы преобразовать сообщения из первой формы во вторую и нужна функция конвертация.

Пример реализации:

```python
def conversations_to_messages(conversation: dict) -> list:
    messages = []
    for item in conversation['conversations']:
        messages.append({
            "role": 'user' if item['from'] == 'human' else 'assistant',
            "content": item['value']
        })
    return messages
```

Передав такую функцию в конструктор `ChatDataset` мы можем быть спокойны за то, что данные будут в правильном формате.

```python
from datasets import load_dataset
from transformers import AutoTokenizer
from impruver.dataset import ChatDataset

# Подгружаем токенизатор из модели gpt2
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Скачиваем датасет с данными в кривом формате
# https://huggingface.co/datasets/mahiatlinux/Reflection-Dataset-ShareGPT-v2
hf_dataset = load_dataset('mahiatlinux/Reflection-Dataset-ShareGPT-v2', split='train')

# Инициализируем класс ChatDataset передав в него функцию конвертации
chat_dataset = ChatDataset(
    original_records=list(hf_dataset),
    tokenizer=tokenizer,
    converter=conversations_to_messages,
    max_tokens_count=1024
)
```

Вызов `print(chat_dataset[0])` отобразит примерно следующее:

```json lines
{"messages":[{"role":"user","value":"text"},{"role":"assistant","value":"text"}]}
```
