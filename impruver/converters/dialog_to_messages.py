def dialog_to_messages(sample: dict) -> list | None:
    """
    Converts a dialog structure into a list of messages compatible with the OpenAI format.

    Example:
        {"messages": {"roles": ["user", "assistant"], "content": ["user text", "assistant text"]}}

    Args:
        sample (dict): a dictionary containing the dialog data

    Returns:
        list: a list of messages compatible with the OpenAI format
        [{"role": "user", "content": "user text"}, {"role": "assistant", "content": "assistant text"}]
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
