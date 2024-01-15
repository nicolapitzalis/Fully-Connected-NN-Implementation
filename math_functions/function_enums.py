from enum import Enum

class LossFunction(Enum):
    """
    Enum class for loss functions.
    """
    MSE = 1
    MEE = 2
    
class FunctionClassEnum(Enum):
    IDENTITY = 1
    SIGMOIDAL_LIKE = 2
    RELU_LIKE = 3

class ActivationFunction(Enum):
    IDENTITY = 1
    RELU = 2
    LEAKY_RELU = 3
    SOFTPLUS = 4
    TANH = 5
    SIGMOID = 6
    SOFTMAX = 7

class Metrics(Enum):
    ACCURACY = 1
    CONFUSION_MATRIX = 2
    MEE = 3
    MSE = 4

def get_metric_name(metric_type_value: int) -> callable:
    if metric_type_value == Metrics.ACCURACY.value:
        return 'accuracy'
    if metric_type_value == Metrics.CONFUSION_MATRIX.value:
        return 'confusion_matrix'
    if metric_type_value == Metrics.MEE.value:
        return 'mee'
    if metric_type_value == Metrics.MSE.value:
        return 'mse'