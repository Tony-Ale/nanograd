from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from neural_net import Sequential, Layer
    from engine import Nodes

class module(ABC):
    """Main module for neural networks"""

    call_thread_count = True # boolean to ensure that the server, client and gui are placed once into different threads

    @abstractmethod
    def forward(self):
        """
        The forward function must be implemented by all class inheriting from this abstract class
        """
    def parameters(self):
        # for model classes inheriting nn.module
        parameters = {'weights': [], 'bias': []}
        layers = []
        dict_layers = vars(self)
        for key, val in dict_layers.items():
            layers.append(val)

        for layer in layers:
            if isinstance(layer, module):
                parameters['weights'].append(layer.w)
                parameters['bias'].append(layer.b)

        return parameters
    
    def __repr__(self, dict_layers=None) -> str:
        out = ""
        if not dict_layers:
            dict_layers = vars(self)
        for key, val in dict_layers.items():
            if isinstance(val, module):
                out += f"({key}): {val.__class__.__name__}(input= {val.input}, output= {val.output})\n"
            else:
                out += f"({key}): {val.__class__.__name__}()\n"
        return out

        
    def backward(self:Sequential|Layer):
        self.forward_pass.backward()
        return self
    
    def zero_grad(self):
        params:dict[str, list[Nodes]] = self.parameters()
        for key, val in params.items():
            for param in val:
                param.grad = np.zeros(shape=param.grad.shape)
    
    

if __name__ == "__main__":
    pass

