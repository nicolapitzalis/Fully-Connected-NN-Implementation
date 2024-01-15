import numpy as np
from math_functions.function_enums import ActivationFunction, Metrics
from math_functions.loss import mse, mee
from sklearn.metrics import accuracy_score, confusion_matrix

def format_data(data: np.ndarray) -> np.ndarray:
    """
    Formats the input data.

    Args:
        data: The input data as a numpy array.

    Returns:
        The formatted data as a numpy array.
    """
    # if single sample, make it np broadcastable
    if data.ndim == 1:
        return data.reshape(data.shape[0], 1)
    # if matrix, transpose it
    return data.T

def binary_discretizer(x: np.ndarray, activation_output_type_value: int) -> np.ndarray:
    """
    Discretizes the input array based on the activation output type.

    Args:
        x: The input array as a numpy array.
        activation_output_type_value: The value representing the activation output type.

    Returns:
        The discretized array as a numpy array.
    """
    if activation_output_type_value == ActivationFunction.SIGMOID.value:
        return np.where(x >= 0.5, 1, 0)
    elif activation_output_type_value == ActivationFunction.TANH.value:
        return np.where(x >= 0, 1, 0)

def evaluate(y_true: np.ndarray, y_pred: np.ndarray,  metric_type_value: int, activation_output_type_value: int = ActivationFunction.SIGMOID.value) -> float:
    """
    Evaluates the predicted values based on the specified metric type.

    Args:
        y_true: The true values as a numpy array.
        y_pred: The predicted values as a numpy array.
        metric_type_value: The value representing the metric type.
        activation_output_type_value: The value representing the activation output type.

    Returns:
        The evaluation result as a float.
    """
    if metric_type_value == Metrics.ACCURACY.value:
        return accuracy_score(y_true=y_true, y_pred=binary_discretizer(y_pred, activation_output_type_value))
    
    if metric_type_value == Metrics.CONFUSION_MATRIX.value:
        return confusion_matrix(y_true=y_true, y_pred=binary_discretizer(y_pred, activation_output_type_value))
    
    if metric_type_value == Metrics.MEE.value:
        return mee(y_true=y_true, y_pred=y_pred)
    
    if metric_type_value == Metrics.MSE.value:
        return mse(y_true=y_true, y_pred=y_pred)