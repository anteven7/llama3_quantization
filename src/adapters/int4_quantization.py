import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.interfaces import QuantizerInterface
import torch
from torchao.quantization import Int4WeightOnlyConfig, quantize_
from typing import Any, Union
from pathlib import Path

class Int4Quantizer(QuantizerInterface):

    def __init__(self, model: torch.nn.Module, tokenizer: AutoTokenizer, config: Int4WeightOnlyConfig):
        
       super().__init__(model)
       self.config = config
       self.tokenizer = tokenizer

    def quantize(self) -> None:

        #check where te model is located
        if next(self.model.parameters()).is_cuda:
            device = "cuda"
        else:
            device = "cpu"

        #move device to cpu for better quantization (currently on L4 - 22 GB vram)
        if device == "cuda":
            self.model.to("cpu")

        print("Cuantizando el modelo...")
        layers = self.model.model.layers
        for i in range(len(layers)):

            layers[i] = layers[i].to("cuda") # moving the layer to gpu
            
            quantize_(layers[i], self.config) # we quantize the layer
        
            layers[i] = layers[i].to("cpu") # moving the layer back to cpu
            
            torch.cuda.empty_cache() # cleaning cache and garbage collection to free up memory
            gc.collect()
            
            # later for inference we can move the model to gpu again, but for now we keep it on cpu to save vram
        print("Cuantización finalizada!")

    def move_to_device(self, device: str) -> None:

        if device not in ["cpu", "cuda"]:
            raise ValueError("Device must be 'cpu' or 'cuda'")
        self.model.to(device)
        vram_usada = torch.cuda.memory_allocated() / (1024**3)
        print(f"Modelo cargado en cuda exitosamente. VRAM asignada: {vram_usada:.2f} GB")   

        
    def save(self, output_path: Union[str, Path]) -> None:
        # Saving the quantized model using torch.save, which is compatible with torchao models
        torch.save(self.model.state_dict(), output_path)    
    
    def chat(self):
        print("localizacion del modelo:", next(self.model.parameters()).device)

        if next(self.model.parameters()).is_cuda:
            device = "cuda"
        else:
            device = "cpu"

        if device == "cpu":
            self.model.to("cuda")
        
        print("el modelo esta en cuda para chatear:", next(self.model.parameters()).device)

        while True:

            prompt = input("Hola! Soy Llama 3, cuantizado a INT4. Dime algo: ")
            if prompt.lower() == 'exit': break

            message = self.tokenizer(prompt, return_tensors="pt")
            message = message.to("cuda")

            response = self.model.generate(
                input_ids=message.input_ids,
                attention_mask=message.attention_mask,
                max_new_tokens=30
            )

            text=self.tokenizer.batch_decode(response,skip_special_tokens=True)[0]

            print(text)

        pass
