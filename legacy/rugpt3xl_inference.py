import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda"

path = "/home/pasha/Documents/Repository/gpt/impruver/models/rugpt3xl-hf"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.float16,
    device_map=device
)

prompt = "Кто был президентом США в 2020? "
model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

generated_ids = model.generate(
    **model_inputs,
    # do_sample=True,
    # temperature=0.9,
    # num_return_sequences=5,
    # max_length=50,
    # no_repeat_ngram_size=3,
    # repetition_penalty=2.,
)

print(tokenizer.batch_decode(generated_ids)[0])
