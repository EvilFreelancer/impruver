import yaml
import fire

import torch
from transformers import logging, GenerationConfig, BitsAndBytesConfig

from impruver.utils import set_seed, get_dtype, dynamic_import
from impruver.conversation import get_prompt, generate, ChatHistory


def chat(
    config: str,
    output_dir: str | None = None,
    history_limit: int = 10,
    system_prompt: str | None = None,
    seed: int = 42,
    max_new_tokens: int = 200,
    repetition_penalty: float = 1.2,
    do_sample: bool = True,
    temperature: float = 0.5,
    top_p: float = 0.6,
    top_k: int = 40,
):
    set_seed(seed)
    logging.set_verbosity_info()

    #
    # Load configuration
    #

    # Read config from disk
    with open(config, "r") as r:
        config = yaml.safe_load(r)

    # Path where model will be saved
    output_dir = output_dir
    if output_dir is None:
        output_dir = config['output_dir']

    # Get settings of Peft/LoRA adapter
    lora_config = config.get("lora")

    # Class to work with Tokenizer
    model_class = "transformers.AutoModelForCausalLM"
    if "class" in config["model"]:
        model_class = config["model"]["class"]

    # Class to work with Tokenizer
    tokenizer_class = "transformers.AutoTokenizer"
    if "class" in config["tokenizer"]:
        tokenizer_class = config["tokenizer"]["class"]

    # Settings related to bitsandbytes and useful only with LoRA adapter training
    load_in_4bit = False
    if "load_in_4bit" in config["model"]:
        load_in_4bit = bool(config["model"]["load_in_4bit"])
    load_in_8bit = False
    if "load_in_8bit" in config["model"]:
        load_in_8bit = bool(config["model"]["load_in_8bit"])

    # Get ddp settings from config if available
    ddp_config = config.get("ddp", {})

    #
    # Tokenizer preparation
    #

    # Init tokenizer object
    tokenizer_obj = dynamic_import(tokenizer_class)
    tokenizer = tokenizer_obj.from_pretrained(output_dir)

    #
    # Model preparation
    #

    # Data Type is bfloat16 by default
    dtype = torch.bfloat16
    if "dtype" in config["model"]:
        dtype = get_dtype(config["model"]["dtype"])

    # Quantization related settings, can be used only in combination with LoRA adapter
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )

    # Generator config
    generation_config = GenerationConfig.from_pretrained(output_dir)
    generation_config.max_new_tokens = max_new_tokens
    generation_config.repetition_penalty = repetition_penalty
    generation_config.do_sample = do_sample
    generation_config.temperature = temperature
    generation_config.top_p = top_p
    generation_config.top_k = top_k

    # Attention implementation
    attn_implementation = None
    if "attn_implementation" in config["model"]:
        attn_implementation = config["model"]["attn_implementation"]

    # Init model object
    model_obj = dynamic_import(model_class)

    # Read model from folder with trained checkpoints
    model = model_obj.from_pretrained(
        output_dir,
        generation_config=generation_config,
        quantization_config=quantization_config,
        device_map=None if ddp_config else "auto",  # need to be disabled for DDP
        torch_dtype=dtype,
        attn_implementation=attn_implementation
    )

    # If we've trained a LoRA adapter
    if lora_config:
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model=model,
            model_id=output_dir,
            torch_dtype=dtype,
        )

    #
    # Chat loop
    #

    # Start chat loop
    chat_history = ChatHistory(history_limit, system_prompt)
    while True:
        user_message = input("User: ")

        # Reset chat command
        if user_message.strip() == "/reset":
            chat_history = ChatHistory(history_limit, system_prompt)
            print("History reset completed!")
            continue

        # Skip empty messages from user
        if user_message.strip() == "":
            continue

        # Add user message to chat history
        chat_history.add_user_message(user_message)

        # Get list of messages
        prompt = get_prompt(tokenizer, chat_history.get_messages(), True)

        # Generate response
        output = generate(model, tokenizer, prompt, generation_config)

        # Save response to chat history as assistant's message
        chat_history.add_assistant_message(output)
        print("Assistant:", output)


if __name__ == "__main__":
    fire.Fire(chat)
