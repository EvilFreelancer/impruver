def instruction_to_messages(sample: dict, skip_labels: list = ["bad_output"], mapping: dict = None) -> list | None:
    """
    Convert a dataset sample to a list of messages with optional field mapping.

    Args:
        sample (dict): Dataset sample with keys like {"instruction": "text", "input": "text", "output": "text"}.
        skip_labels (list, optional): List of labels to skip, defaults to ["bad_output"].
        mapping (dict, optional): Mapping for field names, e.g., {"instruction": "question", "output": "answer"}.

    Returns:
        list | None: List of messages compatible with OpenAI format, or None if sample is skipped.
    """
    # Apply mapping or use defaults
    instruction_key = mapping.get("instruction", "instruction") if mapping else "instruction"
    input_key = mapping.get("input", "input") if mapping else "input"
    output_key = mapping.get("output", "output") if mapping else "output"
    system_key = mapping.get("system", None) if mapping else None

    if "label" in sample and sample.get(input_key) in skip_labels:
        return None

    instruction = sample.get(instruction_key, "")
    if input_key in sample and sample[input_key]:
        instruction += "\n" + sample[input_key]

    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": sample.get(output_key, "")}
    ]

    if system_key and system_key in sample:
        messages.insert(0, {"role": "system", "content": sample.get(system_key, "")})

    return messages
