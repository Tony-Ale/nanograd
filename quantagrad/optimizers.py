from .engine import Nodes
class SGD:
    def __init__(self, layers:dict[str, list[Nodes]], lr=0.01, alpha=0):
        self.layers = layers
        self.lr = lr
        self.alpha = alpha

    def sgd(self, layers:dict[str, list[Nodes]], lr=0.01):
        for key, val in layers.items():
            for params in val:
                params.data -= lr*(params.grad + self.alpha*params.grad)


    def step(self):
        self.sgd(self.layers, self.lr)
    
