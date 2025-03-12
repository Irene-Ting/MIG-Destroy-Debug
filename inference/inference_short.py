import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import infer_auto_device_map, dispatch_model
import ctypes

model_name = "bigscience/bloom-1b7"
cache_dir = "/workspace/.cache"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", max_memory={0: "2GiB", "cpu": "200GB"}, cache_dir=cache_dir, torch_dtype=torch.float16)

inputs = tokenizer("Explain quantum physics.", return_tensors="pt").to("cuda")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0]))

del model, tokenizer, inputs, output
gc.collect() 
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
torch.cuda.synchronize()