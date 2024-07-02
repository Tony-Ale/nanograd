
# quantagrad

![Let's get activated](./neural_net_pic.jpeg)

An Autograd engine built for fun. Implements backpropagation and a small neural networks library on top of it with a PyTorch-like API. Potentially useful for educational purposes.

### Installation

```bash
pip install quantagrad
```

### Example usage 1

Below is an example showing how it can be used:

```python
from quantagrad.engine import Value

node1 = Nodes(np.array([1.0,])) 
node2 = Nodes(np.array([[2], [3]]))


k = node1 + node2

print(k.backward())
```
### Example usage 2
```python
from quantagrad.neural_net import Layer, Sequential

layer1 = Layer(3, 2)

# printing out the structure of layer1
print(f"----Structure of Layer1----\n{layer1}\n")

# To print weights of layer 1
print(f"----Weights of layer1----\n{layer1.w}\n")

layer2 = Layer(2, 1)

z = Sequential([layer1, layer2,])

print(f"----Structure of Sequential----\n{z}")
```
### Training a neural net
```python
"""How to set up a model for training"""
from quantagrad.module import module
from quantagrad.neural_net import Layer
from quantagrad.activations import ReLU
from quantagrad.loss_functions import CrossEntropyLoss
from quantagrad.optimizers import SGD

class digitNetwork(module):
    def __init__(self):
        self.fc1 = Layer(2, 60)
        self.fc2 = Layer(60, 2)
        self.relu = ReLU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
model = digitNetwork()
criterion = CrossEntropyLoss()
optim = SGD(model.parameters(), lr=0.01, alpha=0)
print(model)
```

The notebook `demo.ipynb` provides a full demo of training a MLP classifier using crossentropy loss and stochastic gradient descent

### License

MIT
