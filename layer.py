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
    - net: Calculates the net input of the layer.
    - forward: Performs the forward pass of the layer.
    """
    
    def __init__(self, inputs: np.ndarray, weights: np.ndarray, activation: Callable[[np.ndarray], np.float32]):
        self.inputs = inputs
        self.activation = activation
        self.weights = weights
        self.net: np.ndarray
        self.outputs: np.float32
        self.delta: np.ndarray
        self.delta_weights: np.ndarray
        self.activation_prime: Callable[[Callable[[np.ndarray], np.float32]], Callable[[np.ndarray], np.float32]]

    def compute_net(self) -> np.ndarray:
        """
        Calculates the net input of the layer.

        Returns:
        - The net input value.
        """
        return np.matmul(self.weights, self.inputs)

    def forward(self) -> np.ndarray:
        """
        Performs the forward pass of the layer.

        Returns:
        - The output values of the layer.
        """
        self.net = self.compute_net()
        self.outputs = self.activation(self.net)
        return self.outputs
    
    def backward(self, delta_k: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass of the layer.

        Args:
            delta_k (np.ndarray): The error gradient of the following layer.

        Returns:
            np.ndarray: The error gradient of the layer.
        """
        act_prime = self.activation_prime(self.net)
        np.multiply(delta_k, act_prime, out=self.delta)
        self.delta = np.matmul(self.weights.T, self.delta)          # computing the error on the present layer
        self.delta_weights = np.outer(self.delta, self.inputs)     # updating the weights
        return self.delta
