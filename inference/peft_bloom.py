import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_or_path = "bigscience/bloomz-7b1"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

max_memory = {0: "1GIB", 1: "1GIB", 2: "2GIB", 3: "10GIB", "cpu": "30GB"}
peft_model_id = '/home/bruce/llms-bloom-training/train/peft/output'
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto", max_memory=max_memory)
model = PeftModel.from_pretrained(model, peft_model_id, device_map="auto", max_memory=max_memory)

print(model.hf_device_map)

model.eval()
inputs = tokenizer('hello, how are you ?', return_tensors="pt")
print(inputs)

with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
    print(outputs)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))