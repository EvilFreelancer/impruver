from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
# model = AutoModelForCausalLM.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")

# path = "ai-forever/rugpt3small_based_on_gpt2"
path = "/home/pasha/Documents/Repository/gpt/impruver/test_trainer"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path)

prompt = "Привет! Меня зовут - Супер Король!"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

generated_ids = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)

print(tokenizer.batch_decode(generated_ids)[0])
