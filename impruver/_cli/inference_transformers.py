import yaml
import fire
import torch
from transformers import logging, GenerationConfig, BitsAndBytesConfig

from impruver.utils import set_seed, get_dtype, dynamic_import

FUNCTION = """
{
    "name": "get_news_headlines",
    "description": "Запросить заголовки новостей на конкретную тематику только если пользователь об этом попросил",
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Тематика новостей"
            }
        },
        "required": [ "topic" ]
    } 
}
"""
DEFAULT_SYSTEM_PROMPT = f"Ты полезный помощник с доступом к следующим функциям. Используй их при необходимости:\n{FUNCTION}"




def infer(
    config_file: str,
    output_dir: str = None,
    seed: int = 42,
    history_limit: int = 10,
):
    set_seed(seed)
    logging.set_verbosity_info()

    #
    # Load configuration
    #

    # Read config from disk
    with open(config_file, "r") as r:
        config = yaml.safe_load(r)

    # Path where model will be saved
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
    generation_config.max_new_tokens = 200
    generation_config.repetition_penalty = 1.2
    generation_config.do_sample = True

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
    chat_history = ChatHistory(history_limit=history_limit)
    while True:
        user_message = input("User: ")

        # Reset chat command
        if user_message.strip() == "/reset":
            chat_history = ChatHistory(history_limit=history_limit)
            print("History reset completed!")
            continue

        # Skip empty messages from user
        if user_message.strip() == "":
            continue

        # Add user message to chat history
        chat_history.add_user_message(user_message)

        # Get list of messages
        prompt = get_prompt(tokenizer, chat_history.get_messages(), True)
        # print("==============================")
        # print(prompt)
        # print("==============================")
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            generation_config=generation_config
        )
        chat_history.add_assistant_message(output)
        print("Bot:", output)
        # print()
        # print("==============================")
        # print()


if __name__ == "__main__":
    fire.Fire(infer)
