from typing import Callable, List
import numpy as np
from layer import Layer

class NeuralNetwork():
    """
    A class representing a neural network.

    Parameters:
    - hidden_layer_sizes (List[int]): List of integers representing the number of units in each hidden layer.
    - n_hidden_layers (int): Number of hidden layers in the network.
    - n_output_units (int): Number of output units in the network.
    - task (str): The type of task the network is designed for (e.g., classification, regression).
    - loss (Callable[[np.ndarray, np.ndarray], np.float32]): Loss function used to compute the network's error.
    - loss_prime (Callable[[np.ndarray, np.ndarray], np.float32]): Derivative of the loss function.
    - activation_hidden_name (str): Name of the activation function used in the hidden layers.
    - activation_output_name (str): Name of the activation function used in the output layer.
    - learning_rate (float): Learning rate used in the network's training algorithm.
    """

    def __init__(self, 
                 hidden_layer_sizes: List[int], 
                 n_hidden_layers: int, 
                 n_output_units: int,
                 task: str,
                 loss: Callable[[np.ndarray, np.ndarray], np.float32], 
                 loss_prime: Callable[[np.ndarray, np.ndarray], np.float32],
                 activation_hidden_name: str,
                 activation_output_name: str,
                 learning_rate: float):
        """
        Initialize the NeuralNetwork object.

        Args:
        - hidden_layer_sizes (List[int]): List of integers representing the number of units in each hidden layer.
        - n_hidden_layers (int): Number of hidden layers in the network.
        - n_output_units (int): Number of output units in the network.
        - task (str): The type of task the network is designed for (e.g., classification, regression).
        - loss (Callable[[np.ndarray, np.ndarray], np.float32]): Loss function used to compute the network's error.
        - loss_prime (Callable[[np.ndarray, np.ndarray], np.float32]): Derivative of the loss function.
        - activation_hidden_name (str): Name of the activation function used in the hidden layers.
        - activation_output_name (str): Name of the activation function used in the output layer.
        - learning_rate (float): Learning rate used in the network's training algorithm.
        """
        
        self.layers: List[Layer] = []
        self.n_features: int = None
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_hidden_layers = n_hidden_layers
        self.n_output_units = n_output_units
        self.task = task
        self.loss = loss
        self.loss_prime = loss_prime
        self.activation_hidden_name = activation_hidden_name
        self.activation_output_name = activation_output_name
        self.learning_rate = learning_rate

    def _add_layer(self, input_size: int, output_size: int, activation_name: str = None):
        """
        Add a layer to the neural network.

        Args:
        - input_size (int): Number of input units for the layer.
        - output_size (int): Number of output units for the layer.
        - activation_name (str): Name of the activation function used in the layer.
        """
        self.layers.append(Layer(input_size, output_size, activation_name))

    def _network_architecture(self):
        """
        Define the architecture of the neural network by adding layers.
        """
        # Add input layer
        self._add_layer(self.n_features, self.n_features)
        
        # Add hidden layers
        for i in range(self.n_hidden_layers):
            if i == 0:
                self._add_layer(self.hidden_layer_sizes[i], self.n_features, self.activation_hidden_name)
            else:
                self._add_layer(self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1], self.activation_hidden_name)
        
        # Add output layer
        self._add_layer(self.hidden_layer_sizes[-1], self.n_output_units, self.activation_output_name)
        
    def _predict_outputs(self, features: np.ndarray) -> np.ndarray:
        """
        Predict the outputs of the neural network for the given features.

        Args:
        - features (np.ndarray): Input features for prediction.

        Returns:
        - output (np.ndarray): Predicted outputs of the neural network.
        """
        self.layers[0].set_input(features)

        for i in range(1, len(self.layers)):
            self.layers[i].input = self.layers[i-1].output
            output = self.layers[i].forward()
        
        return output

    # def fit(X: np.ndarray, Y: np.ndarray)
