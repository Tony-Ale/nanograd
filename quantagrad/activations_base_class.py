class Activations:
    """
    Base class for activation functions
    All Activation functions are designed in such a way whereby calling the class as a function returns the object instance
    but calling direct the activation function in the class returns a Node Object
    """
    def __init__(self) -> None:
        self.forward_pass = None 
        self.prev_layer_output = None