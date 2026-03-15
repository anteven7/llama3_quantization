from abc import ABC, abstractmethod
from typing import Any, Union
from pathlib import Path
import torch.nn as nn

class QuantizerInterface(ABC):
    
    def __init__(self, model: nn.Module):

        self.model = model
        
    @abstractmethod
    def quantize(self) -> None:
        """
        Aplica el algoritmo de cuantización al modelo.
        
        Args:
            model (nn.Module): Modelo base instanciado en PyTorch/Transformers.
            kwargs: Parámetros específicos del algoritmo.
            
        Returns:
            nn.Module: El modelo con los pesos cuantizados.
        """
        pass

    @abstractmethod
    def save(self) -> None:
        """
        Guarda el modelo cuantizado en disco utilizando el formato compatible
        con el algoritmo subyacente.
        
        Args:
            model (nn.Module): El modelo previamente cuantizado.
            output_path (Union[str, Path]): Ruta destino del archivo/directorio.
        """
        pass

