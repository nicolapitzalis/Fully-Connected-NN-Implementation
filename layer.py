from typing import Callable
import numpy as np

class Layer():
    """
    A class representing a neural network layer.

    Args:
        input_size (int): The size of the input to the layer.
        output_size (int): The size of the output from the layer.
        activation (Callable[[np.ndarray], np.float32], optional): The activation function to be applied to the layer's output. Defaults to None.
        activation_prime (Callable[[np.ndarray], np.float32], optional): The derivative of the activation function. Defaults to None.

    Attributes:
        input_size (int): The size of the input to the layer.
        output_size (int): The size of the output from the layer.
        activation (Callable[[np.ndarray], np.float32]): The activation function to be applied to the layer's output.
        activation_prime (Callable[[np.ndarray], np.float32]): The derivative of the activation function.
        weight (np.ndarray): The weight of the layer.
        input (np.ndarray): The input to the layer.
        bias (np.ndarray): The bias of the layer.
        net (np.ndarray): The net input to the layer.
        output (np.ndarray): The output of the layer.
        delta (np.ndarray): The error of the layer.
        delta_weight (np.ndarray): The weight updates of the layer.
        delta_bias (np.ndarray): The bias updates of the layer.
    """

    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 activation: Callable[[np.ndarray], np.float32] = None, 
                 activation_prime: Callable[[np.ndarray], np.float32] = None):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.activation_prime = activation_prime
        self.weight = self.weight_init()
        self.input = np.ndarray = None
        self.bias: np.ndarray = None
        self.net: np.ndarray = None
        self.output: np.ndarray = None
        self.delta: np.ndarray = np.zeros(self.output_size)
        self.delta_weight: np.ndarray = np.zeros((self.output_size, self.input_size))
        self.delta_bias: np.ndarray = np.zeros(self.output_size)


    def set_input(self, input_data: np.ndarray):
        """
        Sets the input data for the layer.
        """
        self.input = input_data
        self.output = input_data

    def compute_net(self) -> np.ndarray:
        """
        Computes the net input to the layer.

        Returns:
            np.ndarray: The net input to the layer.
        """
        return np.matmul(self.weight, self.input)

    def forward(self) -> np.ndarray:
        """
        Performs the forward pass of the layer.

        Returns:
            np.ndarray: The output of the layer.
        """
        self.net = self.compute_net()
        self.output = self.activation(self.net)
        return self.output
    
    def backward(self, delta_k: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass of the layer.

        Args:
            delta_k (np.ndarray): The error of the next layer.

        Returns:
            np.ndarray: The error of the current layer.
        """
        act_prime = self.activation_prime(self.net)
        np.multiply(delta_k, act_prime, out=self.delta)
        self.delta = np.matmul(self.weight.T, self.delta)          # computing the error on the present layer
        self.delta_weight = np.outer(self.delta, self.input)      # updating the weight
        return self.delta
    
    def weight_init(self) -> np.ndarray:
        """
        Initializes the weight of the layer.

        Returns:
            np.ndarray: The initialized weight.
        """

        # TODO: implement different weight initialization schemes

        weight = np.random.randn(self.output_size, self.input_size)
        return weight

