import torch
vram_usada = torch.cuda.memory_allocated() / (1024**3)

print(f"VRAM asignada: {vram_usada:.2f} GB")


