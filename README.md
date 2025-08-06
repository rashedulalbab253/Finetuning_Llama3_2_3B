# Fine-tuning Llama 3.2 3B with QLoRA (Unsloth) on Google Colab

This project demonstrates how to fine-tune the [Llama 3.2 3B Instruct](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct) model using [Unsloth](https://github.com/unslothai/unsloth) for efficient, low-VRAM QLoRA (Quantized Low-Rank Adapter) training on Google Colab (T4 GPU). The workflow covers the full process: environment setup, model loading, dataset preparation, QLoRA fine-tuning, and running inference with the finetuned model.

---

## ⚠️ Note

> My previous GitHub account was unexpectedly suspended. This project was originally created earlier and has been re-uploaded here. All work was done gradually over time, and original commit history has been preserved where possible.

## Features

- **Unsloth**: Enables fast, memory-efficient fine-tuning of large language models.
- **QLoRA**: Uses the QLoRA (Quantized Low-Rank Adapter) approach for parameter-efficient fine-tuning, allowing large models to be trained with minimal GPU resources.
- **Llama 3.2 3B**: State-of-the-art instruction-tuned model.
- **Dataset**: Uses [`mlabonne/FineTome-100k`](https://huggingface.co/datasets/mlabonne/FineTome-100k), a high-quality ShareGPT-style dataset.
- **Chat Template Standardization**: Ensures consistent input formatting.
- **Colab Ready**: All steps are designed for easy execution on Google Colab with a T4 GPU.

---

## Workflow

### 1. Install Dependencies

```python
pip install unsloth transformers trl
```

### 2. Import Libraries

```python
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
```

### 3. Load Base Model

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True
 )
```

### 4. Apply QLoRA (PEFT) for Fine-tuning

> **Note:** This notebook uses QLoRA (Quantized Low-Rank Adapter) fine-tuning, which allows you to efficiently train large models by adding trainable adapters to a quantized backbone model. This is achieved with Unsloth's `get_peft_model` method.

```python
model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
```

### 5. Standardize Chat Template

```python
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
```

### 6. Load and Preprocess Dataset

```python
dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = standardize_sharegpt(dataset)
```

- The dataset has 100,000 samples with fields: `conversations`, `source`, `score`.
- Standardization ensures ShareGPT format consistency.

#### Example Conversation Format

```python
dataset[0]
# {
#   'conversations': [...],
#   'source': 'infini-instruct-top-500k',
#   'score': 5.21,
#   ...
# }
```

### 7. Convert Conversations to Model Input

```python
dataset = dataset.map(
    lambda examples: {
        "text": [
            tokenizer.apply_chat_template(convo, tokenize=False)
            for convo in examples["conversations"]
        ]
    },
    batched=True
)
```

### 8. Set Up Trainer

```python
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir="outputs"
    ),
)
```

### 9. Train the Model

```python
trainer.train()
```

- **Note**: For demonstration, training runs for only 60 steps (not a full epoch).

### 10. Save Finetuned Model

```python
model.save_pretrained("finetuned_model")
```

### 11. Reload Finetuned Model for Inference

```python
inference_model, inference_tokenizer = FastLanguageModel.from_pretrained(
    model_name="./finetuned_model",
    max_seq_length=2048,
    load_in_4bit=True
)
```

### 12. Run Inference

```python
text_prompts = [
    "What is investment?"
]

for prompt in text_prompts:
    formatted_prompt = inference_tokenizer.apply_chat_template([{
        "role": "user",
        "content": prompt
    }], tokenize=False)
    model_inputs = inference_tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    generated_ids = inference_model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=inference_tokenizer.pad_token_id
    )
    response = inference_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
```

---

## Notes & Tips

- **QLoRA**: This notebook implements QLoRA (Quantized Low-Rank Adapter) finetuning, which is ideal for training large LLMs on consumer GPUs. Only a small percentage of the model parameters are updated via LoRA adapters, while the main model remains in 4-bit quantized mode.
- **VRAM**: The entire flow is optimized for Google Colab T4 (15GB VRAM). For larger batch sizes or longer sequences, a larger GPU is recommended.
- **Dependencies**: Unsloth patches critical libraries for efficiency. For best results, restart the Colab runtime after installation.
- **Custom Data**: To use your own dataset, adapt the loading and mapping steps accordingly. Make sure the data follows a multi-turn conversation structure.

---

## References

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Llama-3.2-3B-Instruct Model Card](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct)
- [FineTome-100k Dataset](https://huggingface.co/datasets/mlabonne/FineTome-100k)
- [Parameter-Efficient Fine-Tuning (PEFT)](https://github.com/huggingface/peft)

---

## License

Open source for research and educational purposes. Please check individual model and dataset licenses for commercial use.

---
