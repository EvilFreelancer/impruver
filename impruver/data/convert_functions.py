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


def dialog_to_messages(sample) -> list | None:
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
