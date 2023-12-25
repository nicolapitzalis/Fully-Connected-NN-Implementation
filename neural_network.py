from typing import List
from layer import Layer
from math_functions.loss import pick_loss, mse, mee
from math_functions.function_enums import Metrics, LossFunction, ActivationFunction
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

class NeuralNetwork():
    def __init__(self, 
                 n_hidden_layers: int,
                 hidden_layer_sizes: List[int],
                 n_output_units: int,
                 training_loss_type_value: int = LossFunction.MSE.value,
                 validation_loss_type_value: int = LossFunction.MSE.value,
                 evaluation_metric_type_value: int = Metrics.MEE.value,
                 activation_hidden_type_value: int = ActivationFunction.SIGMOID.value,
                 activation_output_type_value: int = ActivationFunction.IDENTITY.value,
                 learning_rate: float = 0.01,
                 reg_lambda: float = 0,
                 mom_alpha: float = 0,
                 nesterov: bool = False,
                 epochs: int = 100,
                 batch_size: int = 1,
                 classification: bool = True,
                 early_stopping: bool = False,
                 fast_stopping: bool = True,
                 linear_decay: bool = False,
                 patience: int = 10,
                 tollerance: float = 0.01,
                 tao: int = 300,
                 verbose: bool = False):
                 
        self.layers: List[Layer] = []
        self.n_features: int = None
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_output_units = n_output_units
        self.training_loss, self.training_loss_prime = pick_loss(training_loss_type_value)
        self.validation_loss = pick_loss(validation_loss_type_value)[0]
        self.evaluation_metric_type_value = evaluation_metric_type_value
        self.activation_hidden_type_value = activation_hidden_type_value
        self.activation_output_type_value = activation_output_type_value
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.final_lr = learning_rate * 0.01
        self.current_lr = learning_rate
        self.reg_lambda = reg_lambda
        self.normalized_reg_lambda = reg_lambda
        self.mom_alpha = mom_alpha
        self.nesterov = nesterov
        self.epochs = epochs
        self.batch_size = batch_size
        self.classification = classification
        self.early_stopping = early_stopping
        self.fast_stopping = fast_stopping
        self.linear_decay = linear_decay
        self.verbose = verbose
        self.patience = patience
        self.tollerance = tollerance
        self.tao = tao
        self.training_losses: List[np.float64] = []
        self.training_evaluations: List[np.float64] = []
        self.validation_losses: List[np.float64] = []
        self.validation_evaluations: List[np.float64] = []
        self.confusion_matrix: np.ndarray = None

    def _format_data(self, data: np.ndarray) -> np.ndarray:
        # if single sample, make it np broadcastable
        if data.ndim == 1:
            return data.reshape(data.shape[0], 1)
        # if matrix, transpose it
        return data.T

    def _binary_discretizer(self, x: np.ndarray) -> np.ndarray:
        if self.activation_output_type_value == ActivationFunction.SIGMOID.value:
            return np.where(x >= 0.5, 1, 0)
        elif self.activation_output_type_value == ActivationFunction.TANH.value:
            return np.where(x >= 0, 1, 0)

    def _evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, metric_type_value: int) -> float:
        if metric_type_value == Metrics.ACCURACY.value:
            return accuracy_score(y_true=y_true, y_pred=self._binary_discretizer(y_pred))
        
        if metric_type_value == Metrics.CONFUSION_MATRIX.value:
            return confusion_matrix(y_true=y_true, y_pred=self._binary_discretizer(y_pred))
        
        if metric_type_value == Metrics.MEE.value:
            return mee(y_true=y_true, y_pred=y_pred)
        
        if metric_type_value == Metrics.MSE.value:
            return mse(y_true=y_true, y_pred=y_pred)

    def _add_layer(self, input_size: int, output_size: int, activation_type_value: int = None):
        self.layers.append(Layer(input_size, output_size, activation_type_value))

    def _network_architecture(self):
        self.layers = []
        # Add hidden layers
        for i in range(self.n_hidden_layers):
            # first takes features as input
            if i == 0:
                self._add_layer(self.n_features, self.hidden_layer_sizes[i], self.activation_hidden_type_value)
            else:
                self._add_layer(self.hidden_layer_sizes[i-1], self.hidden_layer_sizes[i], self.activation_hidden_type_value)
        
        # Add output layer
        self._add_layer(self.hidden_layer_sizes[-1], self.n_output_units, self.activation_output_type_value)

    def _nesterov_momentum(self):
        for layer in self.layers:
            layer.nesterov(self.mom_alpha)
        
    def _forward_propagation(self, data: np.ndarray) -> np.ndarray:
        data = self._format_data(data)
        for layer in self.layers:
            data = layer.forward(data)
        return data.T
    
    def _backward_propagation(self, error: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            error = layer.backward(error)
        return error
    
    def _update_learning_rate(self, epoch: int):
        alpha = epoch/self.tao
        self.current_lr = (1-alpha) * self.initial_lr + alpha * self.final_lr

    def _update_weights(self):
        for layer in self.layers:
            layer.update_weight(self.current_lr, self.normalized_reg_lambda, self.mom_alpha, self.batch_size)
    
    def _weights_norm(self):
        norm=0
        for layer in self.layers:
            norm+=np.linalg.norm(layer.weight, 'fro')**2
        return norm

    def train_net(self, train_data: np.ndarray, train_target: np.ndarray, val_data: np.ndarray = None, val_target: np.ndarray = None):
        
        # initialization__________________________________________________________________________________________________________________________________
        n_samples = train_data.shape[0]
        self.n_features = train_data.shape[1]
        n_batches = np.ceil(n_samples / self.batch_size)
        self.normalized_reg_lambda = self.reg_lambda * (self.batch_size / n_samples)      # normalizing l2 reg term for batch size
        
        # fresh initialization per training
        self.training_losses = []
        self.training_evaluations = []
        self.validation_losses = []
        self.validation_evaluations = []
        
        self._network_architecture()        # network initiliazation
        
        # early stopping initialization
        if self.early_stopping:
            best_weights = [0] * len(self.layers)
            best_bias = [0] * len(self.layers)
            patience = self.patience
            slow_decrease_condition = True
        #_________________________________________________________________________________________________________________________________________________

        # training________________________________________________________________________________________________________________________________________
        for epoch in range(self.epochs):
            training_loss = 0
            validation_loss = 0

            x_batches = np.array_split(train_data, n_batches)
            y_batches = np.array_split(train_target, n_batches)
            
            # update learning rate if linear decay is enabled
            if self.linear_decay:
                self._update_learning_rate(epoch)

            # iterating over batches
            for x_batch, y_batch in zip(x_batches, y_batches):

                # iterating over samples in batch
                for x, y in zip(x_batch, y_batch):
                    
                    # nesterov momentum
                    if self.nesterov:
                        self._nesterov_momentum()
                    
                    output = self._forward_propagation(x)
                    error = self.training_loss_prime(y_true=y, y_pred=output)
                    self._backward_propagation(error.T)
                    self._update_weights()
                    training_loss += self.training_loss(y_true=y, y_pred=output)
            
            # computing training loss and accuracy
            training_loss /= self.batch_size
            training_loss += self.normalized_reg_lambda * self._weights_norm()
            self.training_losses.append(training_loss)
            training_evaluation = self._evaluate(y_true=train_target, y_pred=self._forward_propagation(train_data), metric_type_value=self.evaluation_metric_type_value)
            self.training_evaluations.append(training_evaluation)
            #____________________________________________________________________________________________________________________________________________
            
            # validation_________________________________________________________________________________________________________________________________
            if val_data is not None and val_target is not None:
                output = self._forward_propagation(val_data)
                validation_loss = self.validation_loss(y_true=val_target, y_pred=output)

                if self.early_stopping:
                    # if loss is decreasing by a very small amount, we stop
                    if epoch >= self.patience:
                        if self.fast_stopping:
                            slow_decrease_condition = abs(validation_loss - self.validation_losses[-1]) >= self.tollerance
                        # forces it to enter at least once in order to set the best weight
                        if (validation_loss < self.validation_losses[-1] and slow_decrease_condition) or epoch == self.patience:
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
                
                #computing validation loss and accuracy
                self.validation_losses.append(validation_loss)
                validation_evaluation = self._evaluate(y_true=val_target, y_pred=self._forward_propagation(val_data), metric_type_value=self.evaluation_metric_type_value)
                self.validation_evaluations.append(validation_evaluation)
            #____________________________________________________________________________________________________________________________________________
            
            # print epoch's info
            if self.verbose:
                formatted_output = "Epoch: {:<5} Training Loss: {:<30} Training Evaluation: {:<30} Validation Loss: {:<30} Validation Evaluation: {:<30}"
                print(formatted_output.format(epoch+1, training_loss, training_evaluation, validation_loss, validation_evaluation))


    def predict_and_evaluate(self, data: np.ndarray, target: np.ndarray, metric_type_value: int) -> np.ndarray:
        return self._evaluate(y_true=target, y_pred=self._forward_propagation(data), metric_type_value=metric_type_value)
