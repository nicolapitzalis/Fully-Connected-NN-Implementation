from enum import Enum
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

class Metrics(Enum):
    ACCURACY = 1
    CONFUSION_MATRIX = 2

def format_data(data: np.ndarray) -> np.ndarray:
    # if single sample, make it np broadcastable
    if data.ndim == 1:
        return data.reshape(data.shape[0], 1)
    # if matrix, transpose it
    return data.T

def _binary_discretizer(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0.5, 1, 0)

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, metric_type_value: int) -> float:
    if metric_type_value == Metrics.ACCURACY.value:
        return accuracy_score(y_true=y_true, y_pred=_binary_discretizer(y_pred))
    if metric_type_value == Metrics.CONFUSION_MATRIX.value:
        return confusion_matrix(y_true=y_true, y_pred=_binary_discretizer(y_pred))
    