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
        # Add input layer
        self._add_layer(self.n_features, self.n_features)
        
        # Add hidden layers
        for i in range(self.n_hidden_layers):
            if i == 0:
                self._add_layer(self.n_features, self.hidden_layer_sizes[i], self.activation_hidden_type_value)
            else:
                self._add_layer(self.hidden_layer_sizes[i-1], self.hidden_layer_sizes[i], self.activation_hidden_type_value)
        
        # Add output layer
        self._add_layer(self.hidden_layer_sizes[-1], self.n_output_units, self.activation_output_type_value)
        
    def _predict_outputs(self, features: np.ndarray) -> np.ndarray:
        self.layers[0].set_input(features)

        for i in range(1, len(self.layers)):
            self.layers[i].input = self.layers[i-1].output
            output = self.layers[i].forward()
        
        return output
    
    # TODO: Add discretization of output

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int, batch_size: int):
        self.n_features = X.shape[1]
        self._network_architecture()

        for epoch in range(epochs):
            train_loss = 0
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                Y_batch = Y[i:i+batch_size]

                for x, y in zip(X_batch, Y_batch):
                    output = self._predict_outputs(x)
                    train_loss += self.loss(y_true=y, y_pred=output)
                    loss_prime = self.loss_prime(y_true=y, y_pred=output)

                    for i in reversed(range(1, len(self.layers))):
                        loss_prime = self.layers[i].backward(loss_prime)
                        self.layers[i].update_weight(self.learning_rate)

            if epoch % 10 == 0:
                print('Epoch: {0}, loss: {1}'.format(epoch, train_loss))
