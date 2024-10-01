from sympy.polys.subresultants_qq_zz import rotate_l


def conversations_to_messages(sample: dict) -> list:
    """
    Simple function to convert a conversation to a list of messages.

    Example:
        {"conversation": [{"from": "human", "value": "human text"}, {"from": "bot", "value": "bot text"}]}

    Args:
        sample (dict): a conversation dictionary object

    Returns:
        list: a list of messages compatible with the OpenAI format
        [{"role": "user", "content": "human text"}, {"role": "assistant", "content": "bot text"}]
    """
    messages = []
    for item in sample['conversations']:
        if item['from'] == 'human':
            role = 'user'
        elif item['from'] == 'bot':
            role = 'assistant'
        elif item['from'] == 'system':
            role = 'system'
        else:
            raise ValueError(f"Invalid 'from' value: {item['from']}")
        messages.append({
            "role": role,
            "content": item['value']
        })
    return messages