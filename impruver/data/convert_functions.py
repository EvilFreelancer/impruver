import random


def conversations_to_messages(sample: dict) -> list:
    """
    Simple function to convert a conversation to a list of messages.

    Args:
        sample (dict): a "conversation" object

    Returns:
        list: a list of "messages" compatible with the OpenAI format
    """
    messages = []
    for item in sample['conversations']:
        messages.append({
            "role": 'user' if item['from'] == 'human' else 'assistant',
            "content": item['value']
        })
    return messages


def instruction_to_messages(sample: dict, skip_labels: list = ["bad_output"]) -> list | None:
    """
    Simple function to convert an instruction to a list of messages.

    Args:
        sample (dict): instruction object in format {"instruction": "text", "input": "text", "output": "text"}
        skip_labels (list, optional): list of labels to skip, defaults to ["bad_output"]

    Returns:
        list: a list of messages in the format {"role": "user"/"assistant", "content": "text"}
    """
    if "label" in sample and sample["input"] in skip_labels:
        print(sample)
        return None
    instruction = sample["instruction"]
    if "input" in sample and sample["input"]:
        instruction += "\nДано: " + sample["input"]
    return [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": sample["output"]}
    ]


def dialog_to_messages(sample: dict) -> list | None:
    """
    Converts a dialog structure into a list of messages compatible with the OpenAI format.

    Args:
        sample (dict): a dictionary containing the dialog data

    Returns:
        list: a list of messages in the format {"role": "user"/"assistant", "content": "text"}
    """
    messages = []
    roles = sample['messages']['role']
    contents = sample['messages']['content']
    if (len(roles) != len(contents)) or (len(roles) < 2):
        return None
    for role, content in zip(roles, contents):
        messages.append({
            "role": 'user' if role == 'user' else 'assistant',
            "content": content
        })
    return messages


def build_char_system_messages(char):
    name = char["name"]
    greeting = char["greeting"]
    example_dialogue = char["example_dialogue"]
    context = ""
    if random.random() < 0.5:
        context += f"Ты {name}. "
    context += f"{char['context']}"
    chat = []
    if random.random() < 0.2:
        context += f"\nПриветствие: {greeting}"
        chat.append({"role": "char", "content": greeting})
    if random.random() < 0.2:
        mapping = {"user": "Пользователь", "char": "Персонаж"}
        example_messages = [f'{mapping[m["role"]]}: {m["content"]}' for m in example_dialogue]
        context += "\nПример диалога:\n" + "\n".join(example_messages)
    chat.insert(0, {"role": "system", "content": context})
    return chat


def char_dialog_to_messages(sample: dict) -> list:
    messages_list = []
    for dialogue in sample["dialogues"]:
        if dialogue["model_name"] != "gpt-4":
            continue
        chat = dialogue["chat"]
        for message in chat:
            if message["role"] == "char":
                message["role"] = "assistant"
            if message["role"] == "operator" or message["role"] == "user":
                message["role"] = "user"
        system_messages = build_char_system_messages(sample)
        messages = system_messages + chat
        messages_list.append(messages)
    return messages_list
