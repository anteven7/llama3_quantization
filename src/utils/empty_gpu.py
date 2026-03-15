import torch
import gc

def clear_gpu_memory():
    # 1. Eliminar variables globales que puedan ocupar espacio (opcional/manual)
    # Por ejemplo: del model, del optimizer, del data
    
    # 2. Forzar la recolección de basura de Python
    gc.collect()
    
    # 3. Vaciar el caché de PyTorch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() # Limpia la comunicación entre procesos si aplica
        
    print(f"Memoria reservada tras limpieza: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"Memoria asignada tras limpieza: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
