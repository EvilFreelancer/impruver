import random


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
