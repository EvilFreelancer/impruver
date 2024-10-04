import yaml
import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForTokenClassification, GenerationConfig
from transformers import Trainer, TrainingArguments, logging, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

from impruver.utils import set_seed, read_jsonl, get_dtype, dynamic_import
from impruver.data import apply_chat_template, DEFAULT_CHAT_TEMPLATE

FUNCTION = """{
    'name': 'calculate_loan_payment',
    'description': 'Рассчитать ежемесячный платеж по кредиту',
    'parameters': {
        'type': 'object',
        'properties': {
            'principal': {
                'type': 'number',
                'description': 'Основная сумма кредита'
            },
            'interest_rate': {
                'type': 'number',
                'description': 'Годовая процентная ставка'
             },
            'loan_term': {
                'type': 'integer',
                'description': 'Срок кредита в годах'
            }
        },
        'required': ['principal', 'interest_rate', 'loan_term']
    }
}"""
DEFAULT_SYSTEM_PROMPT = f"Ты полезный помощник с доступом к следующим функциям. Используй их при необходимости:\n{FUNCTION}"


class ChatHistory:
    def __init__(self, history_limit: int = None):
        self.history_limit = history_limit
        # self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.system_prompt = None
        self.messages = []
        if self.system_prompt is not None:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def add_message(self, role, message):
        self.messages.append({"role": role, "content": message})
        self.trim_history()

    def add_user_message(self, message):
        self.add_message("user", message)

    def add_assistant_message(self, message):
        self.add_message("assistant", message)

    def trim_history(self):
        appendix = 0
        if self.system_prompt is not None:
            appendix = 1
        if self.history_limit is not None and len(self.messages) > self.history_limit + appendix:
            overflow = len(self.messages) - (self.history_limit + appendix)
            self.messages = [self.messages[0]] + self.messages[overflow + appendix:]

    def get_messages(self) -> list:
        return self.messages


def get_prompt(tokenizer, messages, add_generation_prompt=False):
    if (
            hasattr(tokenizer, 'apply_chat_template')
            and hasattr(tokenizer, 'chat_template')
            and tokenizer.chat_template is not None
    ):
        prompt = tokenizer.apply_chat_template(
            messages,
            chat_template=DEFAULT_CHAT_TEMPLATE,
            add_special_tokens=False,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    else:
        # Assume apply_chat_template is a function you have defined
        prompt = apply_chat_template(
            messages,
            chat_template=DEFAULT_CHAT_TEMPLATE,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            # tokenizer=tokenizer,
        )

    return prompt


def generate(model, tokenizer, prompt, generation_config):
    data = tokenizer(prompt, return_tensors="pt")
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data,
        generation_config=generation_config
    )[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()


def infer(
        config_file: str,
        output_dir: str = None,
        seed: int = 42,
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

    # Read repo_id of tokenizer or path to configs on disk#
    tokenizer_name = None
    if "name" in config["tokenizer"]:
        tokenizer_name = config["tokenizer"]["name"]
    elif "name" in config["model"]:
        tokenizer_name = config["model"]["name"]

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
    generation_config.max_new_tokens = 1024
    generation_config.repetition_penalty = 1.2

    # Attention implementation
    attn_implementation = None
    if "attn_implementation" in config["model"]:
        attn_implementation = config["model"]["attn_implementation"]

    # Init model object
    model_obj = dynamic_import(model_class)

    # Read model from folder with trained checkpoints
    model = model_obj.from_pretrained(
        output_dir,
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
    chat_history = ChatHistory(history_limit=10)
    while True:
        user_message = input("User: ")

        # Reset chat command
        if user_message.strip() == "/reset":
            chat_history = ChatHistory(history_limit=10)
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
