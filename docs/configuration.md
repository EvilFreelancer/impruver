# Конфигурация

Далее мы разберём как описывать конфигурацию.

## Секция `datasets`

В данной секции перечислен список датасетов которые должны быть использованы в процессе обучения, при этом
предполагается, что все используемые датасеты загружены на HuggingFace.

Допустим мы планируем обучить модель на комбинации из датасета инструкций и диалогов.

```yaml
datasets:
    # Датасет с инструкциями, применяем конвертер `instruction_to_messages` 
  - name: IlyaGusev/ru_turbo_alpaca
    converter: impruver.instruction_to_messages

    # Датасет с function call примерами, из него нам надо взять первые 1000 элементов сплита train
  - name: korotkov/glaive-function-calling-v2-ru-parsed
    split: train[:1000]

    # Датасет с чатами, применяем `dialog_to_messages` и фильтруем все чаты, количество токенов в которых меньше 1024 токенов
  - name: IlyaGusev/oasst1_ru_main_branch
    converter: impruver.dialog_to_messages
    max_tokens_count: 1024
```

### Параметр `converter`

Отвечает за то какую функцию конвертации элементов датасета потребуется использовать, для того
чтобы преобразовать сэмплы из датасета в формат пригодный для обучения модели.

Если конвертен не указано, то предполагается, что датасет уже имеет правильный формат и в колонке messages содержатся
массивы вида:

```json lines
[{"role": "system", "content": "system text"}, {"role": "user", "content": "user text"}, {"role": "assistant", "content": "assistant text"}]
```

При желании вы можете описать свои функции конвертации и указать в конфигурации к ним путь относительно скрипта для
комбинирования датасета, или же например функции из других пакетов сигнатурах которых совместима с `impruver`.

Подробнее о конвертерах [тут](./converter.md).

### Параметр `split`

Отвечает за то, какой сплит из датасета использовать, если его не указывать то будет использоваться скрипт `train`.

Помимо этого можно указать и диапазон сэмплов, которые вы хотите использовать при обучении модели, скажем требуется
взять 1000 первым сэмплов из датасета:

```yml
datasets:
  - name: korotkov/glaive-function-calling-v2-ru-parsed
    split: train[:1000]
```

### Параметр `max_tokens_count`

Отвечает за то, сколько максимум токенов разрешено в одном сэмпле, если не указано, то используется соответствующий
параметров токенизатора, если он не задан и в токенизаторе, то брать значение по умолчанию из конфигурации.

```yaml
datasets:
  - name: korotkov/glaive-function-calling-v2-ru-parsed
    max_tokens_count: 1024
```


## Секция `tokenizer`

Данная секция описывает, то какой токенизатор использовать в процессе комбинирования датасета и обучения модели.

```yaml
tokenizer:
  class: transformers.AutoTokenizer
  name: ai-forever/rugpt3large_based_on_gpt2
```

### Параметр `class`

Отвечает за то какой класс необходимо использовать для инициализации токенизатора.

В качестве класса можно указать путь до любого другого подходящего вам класса, например:

```yaml
tokenizer:
  class: transformers.GPT2Tokenizer
  name: openai-community/gpt2
```

Так же можно указать путь до пакета содержащего класс и название класса на файловой системе, важно только, чтобы его
сигнатура была совместима с форматом классов представленных в пакете `transformers`. 

```yaml
tokenizer:
  class: ru-gpts.src.xl_wrapper.GPT2Tokenizer
  name: ai-forever/rugpt3xl
```

### Параметр `name`

Позволяет указать токенизатор из какой модели необходимо использовать в процессе обучения, можно указать
как `repo_id` на HuggingFace, так и путь до директории (абсолютный или относительный) в которой будут находиться
конфигурация токенизатора, его словарь и список специальных токенов.

```yaml
tokenizer:
  class: transformers.GPT2Tokenizer
  name: ./models/gpt2
```
