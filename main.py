from src.adapters import Int4Quantizer
import torch
from torchao.quantization import Int4WeightOnlyConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import clear_gpu_memory

#just in case... lets empty gpu hehe :)

clear_gpu_memory()

# lets do it! we try 4int quantization ;)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", dtype=torch.bfloat16, device_map="cpu")
config = Int4WeightOnlyConfig(
    group_size=32,
    int4_packing_format="tile_packed_to_4d",
    int4_choose_qparams_algorithm="hqq",
)

my_quantized_model = Int4Quantizer(model, tokenizer, config)

# device?
first_param_device = next(my_quantized_model.model.parameters()).device
print(f"device pre quantization: {first_param_device}")

my_quantized_model.quantize()
my_quantized_model.move_to_device("cuda")
# my_quantized_model.chat()
my_quantized_model.save("./models/quantized_llama3_8b_int4.pth")

# device?
first_param_device = next(my_quantized_model.model.parameters()).device
print(f"device post quantization: {first_param_device}")
