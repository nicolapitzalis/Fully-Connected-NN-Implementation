import numpy as np

from typing import List
from layer import Layer
from math_functions.loss import pick_loss

class NeuralNetwork():
    def __init__(self, 
                 hidden_layer_sizes: List[int], 
                 n_hidden_layers: int, 
                 n_output_units: int,
                 task: str,
                 loss_type_value: int, 
                 activation_hidden_type_value: int,
                 activation_output_type_value: int,
                 learning_rate: float):
        
        self.layers: List[Layer] = []
        self.n_features: int = None
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_hidden_layers = n_hidden_layers
        self.n_output_units = n_output_units
        self.task = task
        self.loss, self.loss_prime = pick_loss(loss_type_value)
        self.activation_hidden_type_value = activation_hidden_type_value
        self.activation_output_type_value = activation_output_type_value
        self.learning_rate = learning_rate

    def _add_layer(self, input_size: int, output_size: int, activation_name: str = None):
        self.layers.append(Layer(input_size, output_size, activation_name))

    def _network_architecture(self):
        # Add input layer
        self._add_layer(self.n_features, self.n_features)
        
        # Add hidden layers
        for i in range(self.n_hidden_layers):
            if i == 0:
                self._add_layer(self.hidden_layer_sizes[i], self.n_features, self.activation_hidden_type_value)
            else:
                self._add_layer(self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1], self.activation_hidden_type_value)
        
        # Add output layer
        self._add_layer(self.hidden_layer_sizes[-1], self.n_output_units, self.activation_output_type_value)
        
    def _predict_outputs(self, features: np.ndarray) -> np.ndarray:
        self.layers[0].set_input(features)

        for i in range(1, len(self.layers)):
            self.layers[i].input = self.layers[i-1].output
            output = self.layers[i].forward()
        
        return output

    # def fit(X: np.ndarray, Y: np.ndarray)
