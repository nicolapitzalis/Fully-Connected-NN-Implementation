import numpy as np

from typing import List
from layer import Layer
from math_functions.loss import pick_loss
from network_utility import format_data, evaluate, Metrics

class NeuralNetwork():
    def __init__(self, 
                 n_hidden_layers: int, 
                 hidden_layer_sizes: List[int],
                 n_output_units: int,
                 training_loss_type_value: int,
                 validation_loss_type_value: int,
                 activation_hidden_type_value: int,
                 activation_output_type_value: int,
                 learning_rate: float,
                 epochs: int,
                 batch_size: int,
                 classification: bool = True,
                 verbose: bool = False):
        
        self.layers: List[Layer] = []
        self.n_features: int = None
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_output_units = n_output_units
        self.training_loss, self.training_loss_prime = pick_loss(training_loss_type_value)
        self.validation_loss, self. validation_loss_prime= pick_loss(validation_loss_type_value)
        self.activation_hidden_type_value = activation_hidden_type_value
        self.activation_output_type_value = activation_output_type_value
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.classification = classification
        self.verbose = verbose
        self.training_losses: List[np.float64] = []
        self.training_accuracy: List[np.float64] = []
        self.validation_losses: List[np.float64] = []
        self.validation_accuracy: List[np.float64] = []

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
        
    def _forward_propagation(self, data: np.ndarray) -> np.ndarray:
        data = format_data(data)
        for layer in self.layers:
            data = layer.forward(data)
        return data.T
    
    def _backward_propagation(self, error: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            error = layer.backward(error)
        return error

    def _update_weights(self):
        for layer in self.layers:
            layer.update_weight(self.learning_rate)

    def train_net(self, x_matrix: np.ndarray, y_matrix: np.ndarray, x_val_matrix: np.ndarray = None, y_val_matrix: np.ndarray = None):
        n_samples = x_matrix.shape[0]
        self.n_features = x_matrix.shape[1]
        n_batches = np.ceil(n_samples / self.batch_size)
        self._network_architecture()

        # iterating over epochs
        for epoch in range(self.epochs):
            x_batches = np.array_split(x_matrix, n_batches)
            y_batches = np.array_split(y_matrix, n_batches)
            training_loss = 0
            validation_loss = 0

            # iterating over batches
            for x_batch, y_batch in zip(x_batches, y_batches):

                # iterating over samples in batch
                for x, y in zip(x_batch, y_batch):
                    output = self._forward_propagation(x)                                            # computing the forward pass 
                    error = evaluate(y_true=y, y_pred=output, metrics=Metrics.LOSS.value)            # computing the error on the output layer
                    self._backward_propagation(error)                                                # computing the backward pass
                    self._update_weights()                                                           # updating the weights and biases
                    training_loss += self.training_loss(y_true=y, y_pred=output)                     # computing the loss on the present sample
            
            training_loss /= self.batch_size                                                         # computing the average loss on the batch
            training_accuracy = evaluate(y_true=y_matrix, y_pred=self._forward_propagation(x_matrix), metrics=Metrics.ACCURACY.value)
            self.training_losses.append(training_loss)
            self.training_accuracy.append(training_accuracy)
            
            # validation
            if x_val_matrix is not None and y_val_matrix is not None:
                output = self._forward_propagation(x_val_matrix)
                validation_loss = self.validation_loss(y_true=y_val_matrix, y_pred=output)
                validation_accuracy = evaluate(y_true=y_val_matrix, y_pred=output, metrics=Metrics.ACCURACY.value)
                self.validation_losses.append(validation_loss)
                self.validation_accuracy.append(validation_accuracy)
            
            if self.verbose:
                formatted_output = "Epoch: {:<5} Training Loss: {:<30} Training Accuracy: {:<30} Validation Loss: {:<30} Validation Accuracy: {:<30}"
                print(formatted_output.format(epoch+1, training_loss, training_accuracy, validation_loss, validation_accuracy))

        return self