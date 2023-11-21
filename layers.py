from typing import Callable
import numpy as np

class Layer():
    """
    Represents a layer in a neural network.

    Attributes:
    - inputs: The input values to the layer.
    - weights: The weights associated with the inputs.
    - activation: The activation function applied to the net input.
    - outputs: The output values of the layer.
    - errors: The error values of the layer.

    Methods:
    - __init__: Initializes the Layer object.
    - net_input: Calculates the net input of the layer.
    - forward: Performs the forward pass of the layer.
    """
    
    def __init__(self, weights: np.ndarray, activation: Callable[[np.ndarray], np.float32]):
        self.activation = activation
        self.weights = weights
        self.net: np.ndarray
        self.outputs: np.float32
        self.delta: np.ndarray
        self.grad_weights: np.ndarray
        self.activation_prime: Callable[[Callable[[np.ndarray], np.float32]], Callable[[np.ndarray], np.float32]]

 

    def forward(self, inputs) -> np.ndarray:
        """
        Performs the forward pass of the layer.

        Returns:
        - The output values of the layer.
        """
        self.net =self.weights @ inputs  
        self.outputs = self.activation(self.net)
        return self.outputs
    
    def backward(self, prev_error: np.ndarray) -> np.ndarray:
            """
            Performs the backward pass of the layer.

            Args:
                prev_error (np.ndarray): The error gradient of the previous layer.

            Returns:
                np.ndarray: The error gradient of the previous layer.
            """
            act_prime = self.activation_prime(self.net)
            self.delta =  prev_error * act_prime
            self.grad_weights = np.outer(self.delta, self.inputs) # to update the weights
            self.error = self.weights.T @ self.delta # to pass to the previous layer
            return self.error


if __name__=='__main__':
    weights= np.random.rand(3,2)
    print(weights)
    def identity(x):
        return x
    input=np.random.rand(2,1)
    layer = Layer(weights=weights, activation=identity)
    output= layer.forward(input)
    print(f'output:{output}')

    