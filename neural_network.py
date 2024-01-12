from typing import List
from layer import Layer
from math_functions.loss import pick_loss
from math_functions.function_enums import Metrics, LossFunction, ActivationFunction
from neural_network_utility import format_data, evaluate
import numpy as np
import copy

class NeuralNetwork():
    def __init__(self, 
                 hidden_layer_sizes: List[int],
                 n_output_units: int,
                 weight_init_uniform_range: float = None,
                 training_loss_type_value: int = LossFunction.MSE.value,
                 validation_loss_type_value: int = LossFunction.MSE.value,
                 evaluation_metric_type_value: int = Metrics.MEE.value,
                 activation_hidden_type_value: int = ActivationFunction.SIGMOID.value,
                 activation_output_type_value: int = ActivationFunction.IDENTITY.value,
                 learning_rate: float = 0.1,
                 reg_lambda: float = 0,
                 mom_alpha: float = 0.75,
                 nesterov: bool = False,
                 epochs: int = 1000,
                 batch_size: int = 1,
                 classification: bool = False,
                 early_stopping: bool = True,
                 fast_stopping: bool = False,
                 linear_decay: bool = False,
                 patience: int = 20,
                 tolerance: float = 0.1,
                 tao: int = 500,
                 verbose: bool = False):
                 
        self.layers: List[Layer] = []
        self.n_features: int = None
        self.n_hidden_layers = len(hidden_layer_sizes)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_output_units = n_output_units
        self.weight_init_uniform_range = weight_init_uniform_range
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
        self.tolerance = tolerance
        self.tao = tao
        self.training_losses: List[np.float64] = []
        self.training_evaluations: List[np.float64] = []
        self.validation_losses: List[np.float64] = []
        self.validation_evaluations: List[np.float64] = []
        self.confusion_matrix: np.ndarray = None
        self.best_epoch: int = None
        self.loss_at_increase_start: float = None
        self.starting_params = copy.deepcopy(self.__dict__)

    def _add_layer(self, input_size: int, output_size: int, activation_type_value: int = None):
        if self.weight_init_uniform_range is not None:
            self.layers.append(Layer(input_size, output_size, activation_type_value, self.weight_init_uniform_range))
        else:
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
        data = format_data(data)
        for layer in self.layers:
            data = layer.forward(data)
        return data.T
    
    def _backward_propagation(self, error: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            error = layer.backward(error)
        return error
    
    def _update_learning_rate(self, epoch: int):
        if epoch <= self.tao:
            alpha = epoch/self.tao
            self.current_lr = (1-alpha) * self.initial_lr + alpha * self.final_lr
        else:
            self.current_lr = self.final_lr

    def _update_weights(self):
        for layer in self.layers:
            layer.update_weight(self.current_lr, self.normalized_reg_lambda, self.mom_alpha, self.batch_size)
    
    def _weights_norm(self):
        norm=0
        for layer in self.layers:
            norm+=np.linalg.norm(layer.weight, 'fro')**2
        return norm

    def train_net(self, train_data: np.ndarray, train_target: np.ndarray, val_data: np.ndarray = None, val_target: np.ndarray = None, tr_loss_stopping_point: float = None):
        
        # initialization_______________________________________________________________________________________________________________
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
            best_weights_mod = [0] * len(self.layers)
            best_bias_mod = [0] * len(self.layers)
            patience = self.patience
        #________________________________________________________________________________________________________________________________

        # training_______________________________________________________________________________________________________________________
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
            training_loss /= n_samples
            training_loss += self.normalized_reg_lambda * self._weights_norm()
            self.training_losses.append(training_loss)
            training_evaluation = evaluate(y_true=train_target, y_pred=self._forward_propagation(train_data), metric_type_value=self.evaluation_metric_type_value, activation_output_type_value=self.activation_output_type_value)
            self.training_evaluations.append(training_evaluation)

            # when retraining for early stopping, we achieve the same level of fitting
            if tr_loss_stopping_point is not None and training_loss <= tr_loss_stopping_point:
                break

            #____________________________________________________________________________________________________________________________
            
            # validation_________________________________________________________________________________________________________________
            if val_data is not None and val_target is not None:
                output = self._forward_propagation(val_data)
                validation_loss = self.validation_loss(y_true=val_target, y_pred=output)

                if self.early_stopping:
                    # if loss is decreasing by a very small amount, we stop
                    if epoch >= self.patience:
                        loss_change = validation_loss - self.validation_losses[-1]
                        
                        if loss_change > 0:
                            # If loss has increased and we are not already tracking, set the start.
                            if self.loss_at_increase_start is None:
                                self.loss_at_increase_start = validation_loss
                        elif loss_change < 0:
                            # If loss has decreased, reset the start of the increase.
                            self.loss_at_increase_start = None

                        if self.fast_stopping:
                            # If fast stopping is on, we stop if the decrease in loss is not substantial (less than the tolerance).
                            # Note that a decrease in loss would result in a negative loss_change, so we check if it's more than a negative tolerance.
                            tolerance_condition = loss_change > -self.tolerance
                        else:
                            # For non-fast stopping, we check against the loss at the start of the increase
                            if self.loss_at_increase_start is not None:
                                tolerance_condition = validation_loss - self.loss_at_increase_start > self.tolerance
                            else:
                                tolerance_condition = False  # No increase has started, so we don't stop

                        # If the loss is less than the best so far, set the best epoch and save the weights.
                        if validation_loss < min(self.validation_losses) or epoch == self.patience:
                            self.best_epoch = epoch
                            for i, layer in enumerate(self.layers):
                                best_weights[i] = layer.weight.copy()
                                best_bias[i] = layer.bias.copy()
                                best_weights_mod[i] = layer.weight_mod.copy()
                                best_bias_mod[i] = layer.bias_mod.copy()

                        if not tolerance_condition:
                            # Continue training because:
                            # - When fast stopping is on, the loss decrease is substantial (more than the tolerance).
                            # - When fast stopping is off, the loss hasn't increased more than the tolerance.
                            # - Regardless of fast stopping, it also continues if the epoch has reached exactly the patience limit.
                            # This is where you would typically save the current weights as the best if the loss is less than the best so far.
                            patience = self.patience
                        else:
                            # Stop training because:
                            # - When fast stopping is on and the loss decrease is too small (less than the tolerance).
                            # - When fast stopping is off and the loss increases more than the tolerance.
                            # This is where you would typically reduce patience or revert to the best weights if patience has run out.
                            patience -= 1
                            if patience == 0:
                                for i, layer in enumerate(self.layers):
                                    layer.weight = best_weights[i]
                                    layer.weight_mod = best_weights_mod[i]
                                    layer.bias = best_bias[i]
                                    layer.bias_mod = best_bias_mod[i]
                                break
                    
                    # if we reached the last epoch, we revert to the best weights
                    if epoch == self.epochs - 1:
                        for i, layer in enumerate(self.layers):
                            layer.weight = best_weights[i]
                            layer.weight_mod = best_weights_mod[i]
                            layer.bias = best_bias[i]
                            layer.bias_mod = best_bias_mod[i]
                
                #computing validation loss and accuracy
                self.validation_losses.append(validation_loss)
                validation_evaluation = evaluate(y_true=val_target, y_pred=self._forward_propagation(val_data), metric_type_value=self.evaluation_metric_type_value, activation_output_type_value=self.activation_output_type_value)
                self.validation_evaluations.append(validation_evaluation)
            #___________________________________________________________________________________________________________________________
            
            # print epoch's info
            if self.verbose:
                formatted_output = "Epoch: {:<5} Training Loss: {:<30} Training Evaluation: {:<30} Validation Loss: {:<30} Validation Evaluation: {:<30}"
                print(formatted_output.format(epoch, training_loss, training_evaluation, validation_loss, validation_evaluation))

        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        return self._forward_propagation(data)

    def predict_and_evaluate(self, data: np.ndarray, target: np.ndarray, metric_type_value: int) -> np.ndarray:
        return evaluate(y_true=target, y_pred=self._forward_propagation(data), metric_type_value=metric_type_value, activation_output_type_value=self.activation_output_type_value)
