import numpy as np

from math_functions.weight import he_init, xavier_init, zero_init
from math_functions.activation import FunctionClassEnum, pick_function_class, pick_activation

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
        delta_w_old (np.ndarray): The previous Delta_w of the layer.
        delta_w_bias_old (np.ndarray): The previous Delta_w_bias of the layer.
    """

    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 activation_type_value: int):
        self.input_size = input_size
        self.output_size = output_size
        self.activation, self.activation_prime = pick_activation(activation_type_value)
        self.weight = self.weight_init()
        self.weight_mod = self.weight.copy()
        self.input: np.ndarray = None
        self.net: np.ndarray = None
        self.output: np.ndarray = None
        self.bias: np.ndarray = np.zeros((self.output_size, 1))
        self.bias_mod = self.bias.copy()
        self.error: np.ndarray = np.zeros((self.output_size, 1))
        self.delta_weight: np.ndarray = np.zeros((self.output_size, self.input_size))
        self.delta_bias: np.ndarray = np.zeros((self.output_size, 1))
        self.delta_w_old: np.ndarray = np.zeros((self.output_size, self.input_size))
        self.delta_w_bias_old: np.ndarray = np.zeros((self.output_size, 1))

    def nesterov(self, mom_alpha: float):
        """
        Computes the new modified weights for backprop computation, if Nesterov is used.
        Else the function isn't called and weight_mod and bias_mod are equal to the original ones.
        """
        self.weight_mod = self.weight + mom_alpha * self.delta_w_old
        self.bias_mod = self.bias + mom_alpha * self.delta_w_bias_old

    def compute_net(self) -> np.ndarray:
        """
        Computes the net input to the layer.

        Returns:
            np.ndarray: The net input to the layer.
        """
        return np.matmul(self.weight_mod, self.input) + self.bias_mod

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the layer.

        Returns:
            np.ndarray: The output of the layer.
        """
        self.input = input_data
        self.net = self.compute_net()
        self.output = self.activation(self.net)
        return self.output
    
    def backward(self, prev_error: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass of the layer.

        Args:
            prev_delta (np.ndarray): The error of the next layer.

        Returns:
            np.ndarray: The error of the current layer.
        """
        delta = np.zeros((self.output_size, 1))
        act_prime = self.activation_prime(self.net)
        np.multiply(prev_error, act_prime, out=delta)
        self.delta_weight += np.outer(delta, self.input)       # updating the weight (for generalized batch version)
        self.delta_bias += delta                               # updating the bias (for generalized batch version)
        self.error = np.matmul(self.weight_mod.T, delta)           # computing the error on the present layer
        
        return self.error
    
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

    def update_weight(self, learning_rate: float, reg_lambda: float, mom_alpha: float, batch_size: int):
        """
        Updates the weight of the layer.

        Args:
            learning_rate (float): The learning rate.
            reg_lambda (float): Tykhonov regularization parameter.
            mom_alpha (float): momentum parameter.
        """

        self.delta_weight /= batch_size
        self.delta_bias /= batch_size

        delta_w_new = -learning_rate * self.delta_weight + mom_alpha * self.delta_w_old
        self.delta_w_old = delta_w_new.copy()
        self.weight += delta_w_new - 2 * reg_lambda*self.weight
        self.weight_mod = self.weight.copy()
       
        delta_w_bias_new = -learning_rate * self.delta_bias + mom_alpha * self.delta_w_bias_old
        self.delta_w_bias_old = delta_w_bias_new.copy()
        self.bias += delta_w_bias_new
        self.bias_mod = self.bias.copy()
       
        self.delta_weight.fill(0)
        self.delta_bias.fill(0)
