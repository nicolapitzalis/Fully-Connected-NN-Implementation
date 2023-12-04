import numpy as np

from typing import List
from layer import Layer
from math_functions.loss import pick_loss

class NeuralNetwork():
    def __init__(self, 
                 n_hidden_layers: int, 
                 hidden_layer_sizes: List[int], 
                 n_output_units: int,
                 loss_type_value: int, 
                 activation_hidden_type_value: int,
                 activation_output_type_value: int,
                 learning_rate: float):
        
        self.layers: List[Layer] = []
        self.n_features: int = None
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_output_units = n_output_units
        self.loss, self.loss_prime = pick_loss(loss_type_value)
        self.activation_hidden_type_value = activation_hidden_type_value
        self.activation_output_type_value = activation_output_type_value
        self.learning_rate = learning_rate

    def _add_layer(self, input_size: int, output_size: int, activation_type_value: int = None):
        self.layers.append(Layer(input_size, output_size, activation_type_value))

    def _network_architecture(self):        
        # Add hidden layers
        for i in range(self.n_hidden_layers):
            # first takes features as input
            if i == 0:
                self._add_layer(self.n_features, self.hidden_layer_sizes[i], self.activation_hidden_type_value)
            else:
                self._add_layer(self.hidden_layer_sizes[i-1], self.hidden_layer_sizes[i], self.activation_hidden_type_value)
        
        # Add output layer
        self._add_layer(self.hidden_layer_sizes[-1], self.n_output_units, self.activation_output_type_value)
        
    def _forward_propagation(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def _backward_propagation(self, y: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            y = layer.backward(y)
        return y

    def _update_weights(self):
        for layer in self.layers:
            layer.update_weight(self.learning_rate)

    def _cut_treshold(self, y: np.ndarray, treshold: float):
        y[y < treshold] = 0
        y[y >= treshold] = 1
        return y

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int, batch_size: int):
        self.n_features = X.shape[1]
        self._network_architecture()

        for epoch in range(epochs):
            training_loss = 0
            for i in range(0, X.shape[0], batch_size):
                for x, y in zip(X[i:i+batch_size], Y[i:i+batch_size]):
                    output = self._forward_propagation(x)
                    output = self._cut_treshold(output, 0.5)
                    training_loss += self.loss(y_true=y, y_pred=output)
                    error = self.loss_prime(y_true=y, y_pred=output)
                    self._backward_propagation(error)
                    self._update_weights()
                
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, loss: {training_loss}")