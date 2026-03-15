# MAKE SURE GPU IS CLEAN

import torch
import gc

def clear_gpu_memory():
    
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() 
        
    print(f"Memoria reservada tras limpieza: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"Memoria asignada tras limpieza: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

clear_gpu_memory()

from transformers import AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int4WeightOnlyConfig, quantize_

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", torch_dtype=torch.bfloat16, device_map="cpu")
#model.to("cuda")

vram_usada = torch.cuda.memory_allocated() / (1024**3)
print(f"Modelo cargado exitosamente. VRAM asignada: {vram_usada:.2f} GB")


#FOR CHATTING 

# while True:

#     prompt = input("Dime qlq:")
#     if prompt.lower() == 'exit': break

#     message = tokenizer(prompt, return_tensors="pt")
#     message = message.to("cuda")

#     response = model.generate(
#         input_ids=message.input_ids,
#         attention_mask=message.attention_mask,
#         max_new_tokens=30
#     )

#     text=tokenizer.batch_decode(response,skip_special_tokens=True)[0]

#     print(text)

print("parts of the model: ")
for name, module in model.named_modules():
    if "." not in name:
        print(name)


layers = model.model.layers
config = Int4WeightOnlyConfig(
    group_size=32,
    int4_packing_format="tile_packed_to_4d",
    int4_choose_qparams_algorithm="hqq",
)
for layer in layers:
    layer = layer.to("cuda")

    quantize_(layer, config)

    del layer

    gc.collect()
    torch.cuda.empty_cache()

print("cuantizacion capas finito")

model.model.embed_tokens.to("cuda")
model.model.norm.to("cuda")
model.lm_head.to("cuda")


vram_usada = torch.cuda.memory_allocated() / (1024**3)
print(f"Modelo cargado exitosamente. VRAM asignada: {vram_usada:.2f} GB")

# CUANTIZED FOR TRAINING

# while True:

#     prompt = input("Dime qlq:")
#     if prompt.lower() == 'exit': break

#     message = tokenizer(prompt, return_tensors="pt")
#     message = message.to("cuda")

#     response = model.generate(
#         input_ids=message.input_ids,
#         attention_mask=message.attention_mask,
#         max_new_tokens=30
#     )

#     text=tokenizer.batch_decode(response,skip_special_tokens=True)[0]

#     print(text)

torch.save(model.state_dict(), "./models/meta-llama-3-8b-int4.pth")