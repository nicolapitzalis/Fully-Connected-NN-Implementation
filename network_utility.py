from enum import Enum
import numpy as np
from sklearn.metrics import accuracy_score

class Metrics(Enum):
    ACCURACY = 1
    LOSS = 2

def format_data(data: np.ndarray) -> np.ndarray:
    # if single sample, make it np broadcastable
    if data.ndim == 1:
        return data.reshape(data.shape[0], 1)
    # if matrix, transpose it
    return data.T

def _binary_discretizer(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0.5, 1, 0)

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, metrics: int, loss_function=None) -> float:
    if metrics == Metrics.ACCURACY.value:
        return accuracy_score(y_true=y_true, y_pred=_binary_discretizer(y_pred))
    if metrics == Metrics.LOSS.value:
        return loss_function(y_true=y_true, y_pred=y_pred)