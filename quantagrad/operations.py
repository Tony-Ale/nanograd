import numpy as np
from .engine import Nodes
 
def matmul(self:Nodes, other:Nodes):
    """Matrix multiplication"""
    return Nodes.__matmul__(self, other)

def sum(input:Nodes, axis=None, keepdims=True):
    out = Nodes(np.sum(input.data, axis=axis, keepdims=keepdims), (input,), operation='sum', operands_ids=(input.id, f"sum--axis:{axis}"))

    def _backward():
        input.grad += (out.grad * np.ones(input.grad.shape))
    
    out._backward = _backward

    return out

def exp(input:Nodes):
    out = Nodes(np.exp(input.data), (input,), operation='exp', operands_ids=(input.id,))

    def _backward():
        input.grad += np.exp(input.data) * out.grad
    
    out._backward = _backward

    return out 

def log(input:Nodes):
    out = Nodes(np.log(input.data), (input,), operation='log', operands_ids=(input.id,))

    def _backward():
        input.grad += (np.ones(input.data.shape) / input.data) * out.grad
    out._backward = _backward

    return out

if __name__ == "__main__":
    input  = Nodes(np.array([1, 2, 3,]))
    input2 = Nodes(np.array([1.2,2.0, 7.0]))
    weight1 = Nodes(np.array([[1, 2], [3, 4], [5, 6]]))
    weight2 = Nodes(np.array([[3], [2]]))
    out_exp = exp(weight1)
    output = matmul((input2 + input), matmul(weight1,  weight2))
    print(out_exp.backward())
