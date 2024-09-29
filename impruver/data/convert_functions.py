def conversations_to_messages(conversation: dict) -> list:
    """
    Simple function to convert a conversation to a list of messages.

    Args:
        conversation (dict): a "conversation" object

    Returns:
        list: a list of "messages" compatible with the OpenAI format
    """
    messages = []
    for item in conversation['conversations']:
        messages.append({
            "role": 'user' if item['from'] == 'human' else 'assistant',
            "content": item['value']
        })
    return messages
