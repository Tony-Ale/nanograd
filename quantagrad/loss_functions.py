import numpy as np
from .engine import Nodes
from .operations import exp, sum, log
from .neural_net import Layer
class CrossEntropyLoss():
    def __call__(self, predictions:np.ndarray, targets:np.ndarray):
        return self.crossentropyloss(predictions, targets)
    def crossentropyloss(self, predictions:Nodes|np.ndarray|Layer, targets:np.ndarray)->Nodes:
        """
        targets: Indices of targets, it is a row matrix.
        predictions: it is a matrix, where each row contains the predictions per sample in a  batch.
        e.g [prediction for class 1, prediction for class 2...]
        """
        predictions = Nodes(predictions) if isinstance(predictions, np.ndarray) else predictions
        predictions = predictions.forward_pass if isinstance(predictions, Layer) else predictions
        softmax = SoftMax()
        probs = softmax(predictions)

        # number of samples 
        N = predictions.data.shape[0]  # or it can be targets.shape[1]; why column is beacuse targets is reshaped to a row vector i.e 1 by n vector 
        
        logits = -log(probs[range(N), targets]) # indices of true value/target
        
        output = sum(logits)/N

        return output
    
class SoftMax():
    def __call__(self, input):
        return self.softmax(input)
    def softmax(self, input:Nodes)->Nodes:
        expinput = exp(input)
        sum_expinput = sum(expinput, axis=1, keepdims=True)
        output = expinput/sum_expinput
        return output
    
    def softmax2(self, input:Nodes):
        c = Nodes(np.max(input.data, axis=1, keepdims=True))

        expinput = exp(input - c)
        sum_expinput =  sum(expinput, axis=1, keepdims=True)

        log_sum_exp = log(sum_expinput)

        y = c + log_sum_exp

        output = exp(input - y)

        return output