def reasoning_to_messages(sample: dict, mapping: dict = None) -> list | None:
    """
    Convert a dataset sample to a list of messages with optional field mapping.

    Args:
        sample (dict): Dataset sample with keys like {"instruction": "text", "input": "text", "output": "text"}.
        mapping (dict, optional): Mapping for field names, e.g., {"instruction": "question", "output": "answer"}.

    Returns:
        list | None: List of messages compatible with OpenAI format, or None if sample is skipped.
    """
    # Apply mapping or use defaults
    instruction_key = mapping.get("instruction", "instruction") if mapping else "instruction"
    reasoning_key = mapping.get("reasoning", "reasoning") if mapping else "reasoning"
    output_key = mapping.get("output", "output") if mapping else "output"
    system_key = mapping.get("system", None) if mapping else None

    instruction = sample.get(instruction_key, "")
    reasoning = sample.get(reasoning_key, "")

    output = ""
    if reasoning and reasoning != "":
        output += f"<think>\n{reasoning}\n</think>\n\n"
    output += sample.get(output_key, "")

    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output}
    ]

    if system_key and system_key in sample:
        messages.insert(0, {"role": "system", "content": sample.get(system_key, "")})

    return messages
