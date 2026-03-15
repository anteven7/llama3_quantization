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
        """
        pass

    @abstractmethod
    def save(self) -> None:
        """
        Guarda el modelo cuantizado en disco utilizando el formato compatible
        con el algoritmo subyacente.
        """
        pass

