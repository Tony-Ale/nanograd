from .engine import Nodes
from .logger import LoggerSetup
from .neural_net import Layer
from .operations import sum, exp
from .activations_base_class import Activations
import numpy as np

class ReLU(Activations):
    
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, input:Nodes|Layer|np.ndarray):
        self.forward_pass = self.ReLU(input)
        return self

    def ReLU(self, input:Nodes|Layer|np.ndarray):
        if Layer.visualize:
            self.prev_layer_output = input

        input = input.forward_pass if isinstance(input, Layer) else input 
        input = Nodes(input) if isinstance(input, np.ndarray) else input

        logical_matrix = input.data >= np.zeros(input.data.shape)

        # note that i cant do logical_matrix * input because __rmul__ does not support it 
        out = input * logical_matrix # all values >= zero remains the same, others less than zero are turned to zero

        if Nodes.debug or LoggerSetup.debug:
            out.operation = 'ReLU'

        return out
    
class SoftMax(Activations):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, input:Nodes|Layer|np.ndarray) :
        self.forward_pass = self.SoftMax(input)
        return self
    
    def SoftMax(self, input:Nodes|Layer|np.ndarray):
        if Layer.visualize:
            self.prev_layer_output = input
            
        input = input.forward_pass if isinstance(input, Layer) else input 
        input = Nodes(input) if isinstance(input, np.ndarray) else input

        expinput = exp(input)
        sum_expinput = sum(expinput, axis=1, keepdims=True)
        output = expinput/sum_expinput

        if Nodes.debug or LoggerSetup.debug:
            output.operation = 'SoftMax'
        return output

