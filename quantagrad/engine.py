from __future__ import annotations
from .logger import LoggerSetup
import numpy as np
import itertools

class Nodes():

    debug = False
    logger = LoggerSetup.setup_logger(name = __name__)
    id_iter = itertools.count()

    @classmethod
    def set_debugger(cls, debug:bool):
        cls.debug = debug

    def __init__(self, data:np.ndarray|list, children=(), id:int= None, operation:str= None, operands_ids: tuple=None):
        """operation: Operation that formed the node"""
        self.data = data.astype(float) if isinstance(data, np.ndarray) else np.array(data, dtype=float)
        self.grad = np.zeros(self.data.shape, dtype=float)

        if self.debug:
            self.id = next(self.id_iter)
            self.operation = operation
            self.operands_ids = operands_ids
        else:
            self.id=None

        if self.data.ndim == 1:
            self.change_shape()

        self.prev_nodes:set[Nodes] = set(children)
        self._backward = lambda: None

    def __repr__(self) -> str:
        if self.debug:
            return f"Nodes(id= {self.id}, data={self.data.tolist()}, grad={self.grad.tolist()}, operation= {self.operation}, operands={self.operands_ids})\n"
        else:
            return f"Nodes(data={self.data.tolist()}, grad={self.grad.tolist()})\n"

    def __add__(self, other):
        other = [other] if isinstance(other, (int, float)) else other
        other = other if isinstance(other, Nodes) else Nodes(other)
        sum = Nodes(self.data + other.data, (self, other), operation='+', operands_ids=(self.id, other.id))
        def _backward():
            self_data, other_data, self_grad, other_grad, broadcast_shape = Nodes.possible_broadcasting(self, other)
            for row in range(sum.grad.shape[0]):
                for col in range(sum.grad.shape[1]):
                    self_grad[row, col] += sum.grad[row, col]
                    other_grad[row, col] += sum.grad[row, col]

            if self.data.shape != other.data.shape: # perform action if broadcasting was carried out
                self.broadcasted_grad_to_grad(nodes=[self, other], broadcasted_data=[self_grad, other_grad], broadcast_shape=broadcast_shape)
        sum._backward = _backward

        return sum
    
    def __mul__(self, other):   
        """Element wise multiplication"""

        other = [other] if isinstance(other, (int, float)) else other
        other = other if isinstance(other, Nodes) else Nodes(other)
        
        # Multiplication for arrays/matrix with the same shape: it does element wise multiplication
        mul = Nodes(self.data * other.data, (self, other), operation='*', operands_ids=(self.id, other.id))

        def _backward():
            self_data, other_data, self_grad, other_grad, broadcast_shape = Nodes.possible_broadcasting(self, other)
            # perform grad calculation
            for row in range(self_data.shape[0]):
                for col in range(self_data.shape[1]):
                    self_grad[row, col] += other_data[row, col] * mul.grad[row, col]
                    other_grad[row, col] += self_data[row, col] * mul.grad[row, col]
            # Transform self_grad, other_grad to self.grad and other.grad using the right shapes
            if self.data.shape != other.data.shape: # perform action if broadcasting was carried out
                self.broadcasted_grad_to_grad(nodes=[self, other], broadcasted_data=[self_grad, other_grad], broadcast_shape=broadcast_shape)
        mul._backward = _backward

        return mul
    
    def __matmul__(self, other):
        """Matrix multiplication"""
        other = other if isinstance(other, Nodes) else Nodes(other)
        mul = Nodes(np.matmul(self.data,  other.data), (self, other), operation='mat_mul', operands_ids=(self.id, other.id))
        
        def _backward():

            #implementing Einstein summation technique for calculating gradient

            n = self.data.shape[1]
            new_shape = (self.data.shape[0], other.data.shape[1])
            for i in range(new_shape[0]):
                for k in range(new_shape[1]):
                    for j in range(n):
                        self.grad[i, j] += other.data[j, k] * mul.grad[i, k]
                        other.grad[j, k] += self.data[i, j] * mul.grad[i, k]

        mul._backward = _backward

        return mul

    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "The exponent must either be an int or a float type"
        out = Nodes(np.power(self.data, other), (self,), operation='^', operands_ids=(self.id, f"pow= {other}"))

        def _backward():
            self.grad += other * np.power(self.data, other-1) * out.grad

        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other**-1
    
    def change_shape(self):
        self.data = self.data.reshape(1, -1)
        self.grad = self.grad.reshape(1, -1)

    def __rmul__(self, other):
        # __rmul__ can only handle when other is an integer or a floating point number 
        # it cannot handle multiplication with numpy arrays because of numpy implementation of __rmul__
        # i can peharps handle multiplication with numpy arrays, were the numpy array comes first using another function
        other = Nodes([other])
        return self * other
    
    def __sub__(self, other:Nodes): # self- other
        return self + (-other)
    
    def __rsub__(self, other): # other - self
        return other + (-self)
    
    def __radd__(self, other): # other + self
        other = Nodes([other])
        return other + self 

    def __neg__(self): # -self 
        return self * -1
    
    def __getitem__(self, indices):
        out = Nodes(self.data[indices], (self,), operation='indexing', operands_ids=(self.id, f"indices: {indices}"))

        def _backward():
            self.grad[indices] += out.grad.reshape(self.grad[indices].shape)
        
        out._backward = _backward
        return out
    
    def broadcasted_grad_to_grad(self, nodes:list[Nodes], broadcasted_data:list[np.ndarray], broadcast_shape):
        """
        nodes: it is a list of size 2 containing the Nodes that arithemetic operations were performed on 
        broadcasted_data: it is a list of size 2, containing the new writable broadcasted grad of the result of the operation
        """
        for node, bdc_grad in zip(nodes, broadcasted_data): 
            # when both rows and columns were broadcasted
            if broadcast_shape[0] != node.grad.shape[0] and broadcast_shape[1] != node.grad.shape[1]:
                node.grad += bdc_grad.sum() 
            # No broadcasting was done 
            elif broadcast_shape == node.grad.shape:
                node.grad += bdc_grad
            # when rows is broadcasted
            elif broadcast_shape[0] != node.grad.shape[0]:
                node.grad += bdc_grad.sum(axis=0, keepdims=True)
            # when column is broadcasted
            elif broadcast_shape[1] != node.grad.shape[1]:
                node.grad += bdc_grad.sum(axis=1, keepdims=True)
    @staticmethod
    def possible_broadcasting(self:Nodes, other:Nodes):
        """To know whether to try broadcasting"""

        # Broadcast if possible 
        if self.data.shape != other.data.shape:
            broadcast_shape = np.broadcast(self.data, other.data).shape
            self_data = np.broadcast_to(self.data, broadcast_shape)
            other_data = np.broadcast_to(other.data, broadcast_shape)
            self_grad = np.broadcast_to(self.grad, broadcast_shape)
            other_grad = np.broadcast_to(other.grad, broadcast_shape)

            # to make self_grad and other_grad writable
            self_grad = np.ones(broadcast_shape) * self_grad
            other_grad = np.ones(broadcast_shape) * other_grad
        else:
            self_data = self.data
            other_data = other.data
            self_grad = self.grad
            other_grad = other.grad
            broadcast_shape = self.data.shape

        return self_data, other_data, self_grad, other_grad, broadcast_shape
    
    def backward(self):
        visited = set()
        sorted_nodes = []

        def topo_sort(node):
            if node not in visited:
                visited.add(node)
                for child in node.prev_nodes:
                    topo_sort(child)
                sorted_nodes.append(node)
        
        topo_sort(self)

        self.grad = np.ones(self.data.shape)

        for node in reversed(sorted_nodes):
            node._backward()
            
            if self.debug or LoggerSetup.debug:
                self.logger.debug(node)

        #print(sorted_nodes)


            

if __name__ == '__main__':

    Nodes.set_debugger(True)
    LoggerSetup.set_debugger(False)

    node1 = Nodes(np.array([1.0,])) 
    node2 = Nodes(np.array([[2], [3]]))


    k = node1 + node2

    print(k.backward())
    
        