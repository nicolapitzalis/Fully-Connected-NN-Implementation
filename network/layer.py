import numpy as np

from math_functions.weight import he_init, xavier_init, zero_init
from math_functions.activation import FunctionClassEnum, pick_function_class, pick_activation

class Layer():
    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 activation_type_value: int):
        """
        Initializes a Layer object.

        Args:
            input_size (int): The size of the input to the layer.
            output_size (int): The size of the output from the layer.
            activation_type_value (int): The value representing the activation function type.

        Returns:
            None
        """
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
        Applies Nesterov momentum to update the weights and biases.

        Args:
            mom_alpha (float): The momentum coefficient.

        Returns:
            None
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
        Performs forward propagation through the layer.

        Args:
            input_data (np.ndarray): The input data.

        Returns:
            np.ndarray: The output of the layer.
        """
        self.input = input_data
        self.net = self.compute_net()
        self.output = self.activation(self.net)
        return self.output
    
    def backward(self, prev_error: np.ndarray) -> np.ndarray:
        """
        Performs backward propagation through the layer.

        Args:
            prev_error (np.ndarray): The error from the next layer.

        Returns:
            np.ndarray: The error for the current layer.
        """
        delta = np.zeros((self.output_size, 1))
        act_prime = self.activation_prime(self.net)
        np.multiply(prev_error, act_prime, out=delta)
        self.delta_weight += np.outer(delta, self.input)
        self.delta_bias += delta
        self.error = np.matmul(self.weight_mod.T, delta)
        
        return self.error
    
    def weight_init(self) -> np.ndarray:
        """
        Initializes the weights of the layer based on the activation function.

        Returns:
            np.ndarray: The initialized weights.
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
        Updates the weights and biases of the layer.

        Args:
            learning_rate (float): The learning rate.
            reg_lambda (float): The regularization parameter.
            mom_alpha (float): The momentum coefficient.
            batch_size (int): The size of the batch.

        Returns:
            None
        """
        self.delta_weight /= batch_size
        self.delta_bias /= batch_size

        delta_w_new = -learning_rate * self.delta_weight + mom_alpha * self.delta_w_old
        self.delta_w_old = delta_w_new.copy()
        self.weight += delta_w_new - 2 * reg_lambda * self.weight
        self.weight_mod = self.weight.copy()
       
        delta_w_bias_new = -learning_rate * self.delta_bias + mom_alpha * self.delta_w_bias_old
        self.delta_w_bias_old = delta_w_bias_new.copy()
        self.bias += delta_w_bias_new
        self.bias_mod = self.bias.copy()
       
        self.delta_weight.fill(0)
        self.delta_bias.fill(0)
