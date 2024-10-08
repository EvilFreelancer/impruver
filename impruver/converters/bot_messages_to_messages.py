def bot_messages_to_messages(sample: dict) -> list:
    """
    Simple function to convert a conversation to a list of messages.

    Example:
        {"conversation": [{"role": "user", "content": "human text"}, {"role": "bot", "content": "bot text"}]}

    Args:
        sample (dict): a conversation dictionary object

    Returns:
        list: a list of messages compatible with the OpenAI format
        [{"role": "user", "content": "human text"}, {"role": "assistant", "content": "bot text"}]
    """
    messages = []
    for item in sample['messages']:
        if item['role'] == 'user':
            role = 'user'
        elif item['role'] == 'human':
            role = 'user'
        elif item['role'] == 'gpt':
            role = 'assistant'
        elif item['role'] == 'bot':
            role = 'assistant'
        elif item['role'] == 'system':
            role = 'system'
        else:
            raise ValueError(f"Invalid 'role' value: {item['role']}")
        messages.append({
            "role": role,
            "content": item['content']
        })
    return messages
