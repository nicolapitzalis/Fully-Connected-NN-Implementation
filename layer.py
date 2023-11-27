from math_functions.weight import he_init, xavier_init, zero_init
from math_functions.activation import FunctionClassEnum, pick_function_class, pick_activation
import numpy as np

class Layer():
    """
    A class representing a neural network layer.

    Args:
        input_size (int): The size of the input to the layer.
        output_size (int): The size of the output from the layer.
        activation_name (str): The name of the activation function to be applied to the layer's output.

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
                 activation_name: str):
        self.input_size = input_size
        self.output_size = output_size
        self.activation, self.activation_prime = pick_activation(activation_name)
        self.weight = self.weight_init()
        self.input = np.ndarray = None
        self.net: np.ndarray = None
        self.output: np.ndarray = None
        self.bias: np.ndarray = np.zeros(self.output_size)
        self.delta: np.ndarray = np.zeros(self.output_size)
        self.delta_weight: np.ndarray = np.zeros((self.output_size, self.input_size))
        self.delta_bias: np.ndarray = np.zeros(self.output_size)


    def set_input(self, input_data: np.ndarray):
        """
        Sets the input data for the layer.

        Args:
            input_data (np.ndarray): The input data for the layer.
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
    
    def backward(self, prev_delta: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass of the layer.

        Args:
            prev_delta (np.ndarray): The error of the next layer.

        Returns:
            np.ndarray: The error of the current layer.
        """
        act_prime = self.activation_prime(self.net)
        np.multiply(prev_delta, act_prime, out=self.delta)
        self.delta = np.matmul(self.weight.T, self.delta)           # computing the error on the present layer
        self.delta_weight += np.outer(self.delta, self.input)       # updating the weight (for generalized batch version)
        
        # TODO: check if the bias update is correct
        self.delta_bias += np.multiply(self.delta, self.bias)       # updating the bias (for generalized batch version)
        return self.delta
    
    def weight_init(self) -> np.ndarray:
        """
        Initializes the weight of the layer.

        Returns:
            np.ndarray: The initialized weight.
        """

        if self.activation.__name__ in pick_function_class(FunctionClassEnum.RELU_LIKE.value):
            weight = he_init(self.input_size, self.output_size)

        if self.activation.__name__ in pick_function_class(FunctionClassEnum.SIGMOIDAL_LIKE.value):
            weight = xavier_init(self.input_size, self.output_size)

        if self.activation.__name__ in pick_function_class(FunctionClassEnum.IDENTITY.value):
            weight = zero_init(self.input_size, self.output_size)

        return weight

