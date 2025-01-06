Прототип модели использующих 1.5bit линейные слои из
проекта [BitNet-Transformers](https://github.com/Beomi/BitNet-Transformers) для
обучения LLM с архитектурой Llama.

В процессе обучения задействован словарь токенизатора
от [Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct](https://huggingface.co/Vikhrmodels/Vikhr-Llama-3.2-1B-Instruct).

Обучать вот так:

```shell
impruver run finetune --config custom/llama/110M_full_alpaca.yaml
```

## Ссылки

* [Reducing Transformer Depth on Demand with Structured Dropout](https://arxiv.org/abs/1909.11556)
* [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)
* [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)
* https://github.com/Beomi/BitNet-Transformers
