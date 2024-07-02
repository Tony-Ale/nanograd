from __future__ import annotations
from .logger import LoggerSetup
from .engine import Nodes
from .operations import matmul
from .module import module
from .activations_base_class import Activations
import numpy as np
import math

class Layer(module):

    visualize = False
    debug = False
    logger = LoggerSetup.setup_logger(name = __name__)

    @classmethod
    def set_debugger(cls, debug:bool):
        cls.debug = debug

    def __init__(self, input, output) -> None:
        self.input = input 
        self.output = output

        self.w = Nodes(data=np.random.uniform(-math.sqrt(1/input), math.sqrt(1/input), size=(input, output))) # Following pytorch method of initilaization of a Linear layer
        self.b = Nodes(data=np.random.uniform(-math.sqrt(1/input), math.sqrt(1/input), size=(output,)))

        self.forward_pass:Nodes = None
        self.prev_layer_output = None
    
    def __call__(self, x:np.ndarray|Nodes):
        return self.forward(x)

    def __repr__(self) -> str:
        if self.debug or LoggerSetup.debug:
            return f"Weights: {self.w}\nBias: {self.b}\nforward pass: {self.forward_pass}"
        else:
            return f"Layer(input= {self.input}, output= {self.output})"

    def forward(self, x:np.ndarray|Nodes|Layer):
        if self.visualize: # To keep track of previous layer input if visualize is enabled.
            self.prev_layer_output = x

        x = x.forward_pass if isinstance(x, Layer) else x # case whereby this function is used as chain, that is the output of one is the input of the next
        x = x.forward_pass if isinstance(x, Activations) else x
        x = Nodes(data=x) if isinstance(x, np.ndarray) else x

        self.forward_pass = matmul(x, self.w) + self.b
        return self
    
    def parameters(self):
        params = {'weights': [], 'bias': []}
        params['weights'].append(self.w)
        params['bias'].append(self.b)
        return params 

class Sequential(module):
    debug = False
    logger = LoggerSetup.setup_logger(name = __name__)

    @classmethod
    def set_debugger(cls, debug:bool):
        cls.debug = debug

    def __init__(self, layers:list[Layer]):
        self.layers = layers
        self.forward_pass = None
    
    def forward(self, x:np.ndarray|Nodes|Layer):
        x = x.forward_pass if isinstance(x, Layer) else x # case whereby this function is used as chain, that is the output of one is the input of the next
        for layer in self.layers:
            layer.forward(x)
            x = layer.forward_pass
        self.forward_pass = x
        return self
    
    def parameters(self):
        params = {'weights': [], 'bias': []}
        for layer in self.layers:
            params['weights'].append(layer.w)
            params['bias'].append(layer.b)
        return params 
            

    def __repr__(self) -> str:
        if self.debug or LoggerSetup.debug:
            Layer.set_debugger(True)
            return '\n\n'.join(repr(layer) for layer in self.layers)
        else:
            keys = [i for i in range(len(self.layers))]
            dict_layers = dict(zip(keys, self.layers))
            return super().__repr__(dict_layers=dict_layers)


if __name__ == "__main__":
    LoggerSetup.set_debugger(False)
    Layer.set_debugger(False)
    Sequential.set_debugger(True)

    layer1 = Layer(3, 2)

    # printing out the structure of layer1
    print(f"----Structure of Layer1----\n{layer1}\n")

    # To print weights of layer 1
    print(f"----Weights of layer1----\n{layer1.w}\n")

    layer2 = Layer(2, 1)

    z = Sequential([layer1, layer2,])

    print(f"----Structure of Sequential----\n{z}")
    