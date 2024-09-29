def conversations_to_messages(conversation: dict) -> list:
    messages = []
    for item in conversation['conversations']:
        content = item['value']
        role = 'assistant'
        if item['from'] == 'human':
            role = 'user'
        if item['from'] == 'gpt':
            role = 'assistant'
        messages.append({"role": role, "content": content})
    return messages
