import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

# path = "/home/pasha/Documents/Repository/gpt/impruver/test_trainer"
path = "ai-forever/rugpt3small_based_on_gpt2"

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map=device,
)

prompt = "def hello_world():"
model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
print(tokenizer.batch_decode(generated_ids)[0])
