import yaml
import fire
from transformers import AutoTokenizer


def test(config_path: str):
    with open(config_path, "r") as r:
        config = yaml.safe_load(r)
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['name'])

    input_ids = [1, 1072, 444, 203, 16676, 17649, 30175, 5152, 7788]
    output = tokenizer.decode(input_ids)
    print(output)


if __name__ == "__main__":
    fire.Fire(test)
