def instruction_to_messages(sample: dict, skip_labels: list = ["bad_output"]) -> list | None:
    """
    Simple function to convert an instruction to a list of messages.

    Example:
        {"instruction": "instruction text", "input": "input text", "output": "output text"}

    Args:
        sample (dict): instruction object in format {"instruction": "text", "input": "text", "output": "text"}
        skip_labels (list, optional): list of labels to skip, defaults to ["bad_output"]

    Returns:
        list: a list of messages compatible with the OpenAI format
        [{"role": "user", "content": "instruction text\ninput text"}, {"role": "assistant", "content": "output text"}]
    """
    if "label" in sample and sample["input"] in skip_labels:
        return None
    instruction = sample["instruction"]
    if "input" in sample and sample["input"]:
        instruction += "\n" + sample["input"]
    return [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": sample["output"]}
    ]
