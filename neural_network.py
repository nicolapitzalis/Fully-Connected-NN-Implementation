import numpy as np

from typing import List
from layer import Layer
from math_functions.activation import ActivationFunction
from math_functions.loss import LossFunction, pick_loss
from network_utility import format_data, evaluate, Metrics

class NeuralNetwork():
    def __init__(self, 
                 n_hidden_layers: int, 
                 hidden_layer_sizes: List[int],
                 n_output_units: int,
                 training_loss_type_value: int = LossFunction.MSE.value,
                 validation_loss_type_value: int = LossFunction.MSE.value,
                 evaluation_metric_type_value: int = LossFunction.MEE.value,
                 activation_hidden_type_value: int = ActivationFunction.SIGMOID.value,
                 activation_output_type_value: int = ActivationFunction.IDENTITY.value,
                 learning_rate: float = 0.01,
                 Lambda: float=0,
                 Alpha: float=0,
                 epochs: int = 100,
                 batch_size: int = 1,
                 classification: bool = True,
                 early_stopping: bool = False,
                 patience: int = 10,
                 tollerance: float = 0.01,
                 verbose: bool = False):
                 
                 

        
        self.layers: List[Layer] = []
        self.n_features: int = None
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_output_units = n_output_units
        self.training_loss, self.training_loss_prime = pick_loss(training_loss_type_value)
        self.validation_loss = pick_loss(validation_loss_type_value)[0]
        self.evaluation_metric = pick_loss(evaluation_metric_type_value)[0]
        self.activation_hidden_type_value = activation_hidden_type_value
        self.activation_output_type_value = activation_output_type_value
        self.learning_rate = learning_rate
        self.Lambda=Lambda
        self.Alpha=Alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.classification = classification
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.patience = patience
        self.tollerance = tollerance
        self.training_losses: List[np.float64] = []
        self.training_accuracy: List[np.float64] = []
        self.validation_losses: List[np.float64] = []
        self.early_stopping_losses: List[np.float64] = []
        self.validation_accuracy: List[np.float64] = []
        self.confusion_matrix: np.ndarray = None

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
            layer.update_weight(self.learning_rate, self.Lambda, self.Alpha)

    def train_net(self, train_data: np.ndarray, train_target: np.ndarray, val_data: np.ndarray = None, val_target: np.ndarray = None):
        self.n_features = train_data.shape[1]
        self._network_architecture()

    def weights_norm(self):
        norm=0
        for layer in self.layers:
            norm+=np.linalg.norm(layer.weight, 'fro')**2
        return norm
    
        # early stopping initialization
        if self.early_stopping:
            split_index = len(train_data) * 2 // 3
            train_data, early_stopping_data = train_data[:split_index], train_data[split_index:]
            train_target, early_stopping_target = train_target[:split_index], train_target[split_index:]
            best_weights = [0] * len(self.layers)
            best_bias = [0] * len(self.layers)
            patience = self.patience

        n_samples = train_data.shape[0]
        n_batches = np.ceil(n_samples / self.batch_size)

        # iterating over epochs
        for epoch in range(self.epochs):
            x_batches = np.array_split(train_data, n_batches)
            y_batches = np.array_split(train_target, n_batches)
            training_loss = 0
            validation_loss = 0

            # iterating over batches
            for x_batch, y_batch in zip(x_batches, y_batches):

                # iterating over samples in batch
                for x, y in zip(x_batch, y_batch):
                    output = self._forward_propagation(x)
                    error = self.training_loss_prime(y_true=y, y_pred=output)
                    self._backward_propagation(error)
                    self._update_weights()
                    training_loss += self.training_loss(y_true=y, y_pred=output)
            
            training_loss /= self.batch_size
            training_loss += Lambda*self.weights_norm()
            training_accuracy = evaluate(y_true=train_target, y_pred=self._forward_propagation(train_data), metric_type_value=Metrics.ACCURACY.value)
            self.training_losses.append(training_loss)
            self.training_accuracy.append(training_accuracy)
            
            # stopping decision
            if self.early_stopping:
                output = self._forward_propagation(early_stopping_data)
                es_loss = self.validation_loss(y_true=early_stopping_target, y_pred=output)

                # if loss is decreasing by a very small amount, we stop
                if epoch >= self.patience:
                    if abs(es_loss - self.early_stopping_losses[-1]) >= self.tollerance:
                        patience = self.patience
                        for i, layer in enumerate(self.layers):
                            best_weights[i] = layer.weight
                            best_bias[i] = layer.bias
                    else:
                        patience -= 1
                        if patience == 0:
                            for i, layer in enumerate(self.layers):
                                layer.weight = best_weights[i]
                                layer.bias = best_bias[i]
                            break

                self.early_stopping_losses.append(es_loss)
                
            # validation
            if val_data is not None and val_target is not None:
                output = self._forward_propagation(val_data)
                validation_loss = self.validation_loss(y_true=val_target, y_pred=output)
                self.validation_losses.append(validation_loss)

                validation_accuracy = evaluate(y_true=val_target, y_pred=output, metric_type_value=Metrics.ACCURACY.value)
                self.validation_accuracy.append(validation_accuracy)

                if self.classification:
                    self.confusion_matrix = evaluate(y_true=val_target, y_pred=output, metric_type_value=Metrics.CONFUSION_MATRIX.value)
            
            if self.verbose:
                formatted_output = "Epoch: {:<5} Training Loss: {:<30} Training Accuracy: {:<30} Validation Loss: {:<30} Validation Accuracy: {:<30}"
                print(formatted_output.format(epoch+1, training_loss, training_accuracy, validation_loss, validation_accuracy))

        return self
    
    def predict_and_evaluate(self, data: np.ndarray, target: np.ndarray) -> np.ndarray:
        return evaluate(y_true=target, y_pred=self._forward_propagation(data), metric_type_value=Metrics.ACCURACY.value)