"""
基座模型：基于 Hugging Face Transformers 的最小推理示例。
默认从环境变量 BASE_MODEL_PATH 或 ./models/base 加载权重。
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.environ.get("BASE_MODEL_PATH", os.path.join(_ROOT, "models", "base"))


def predict(messages, model, tokenizer):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=2048)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16)

    test_texts = {
        "instruction": "你是一个农业专家，你需要根据用户的问题，给出带有思考的回答。",
        "input": "水稻施肥顺序应按照什么顺序进行？",
    }

    messages = [
        {"role": "system", "content": test_texts["instruction"]},
        {"role": "user", "content": test_texts["input"]},
    ]

    print(predict(messages, model, tokenizer))


if __name__ == "__main__":
    main()
