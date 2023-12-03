from enum import Enum
from typing import Callable, Tuple
import numpy as np

class LossFunction(Enum):
    """
    Enum class for loss functions.
    """
    MSE = 1
    MEE = 2

def mse (y_true: np.ndarray, y_pred: np.ndarray) -> np.float32:
    """
    Mean Squared Error (MSE) loss function.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        np.float32: MSE loss.
    """
    if y_true.ndim == 1:
        return np.mean((y_true - y_pred) ** 2)
    return np.mean(np.sum((y_true - y_pred) ** 2, axis=1))

def mee (y_true: np.ndarray, y_pred: np.ndarray) -> np.float32:
    """
    Mean Euclidean Error (MEE) loss function.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        np.float32: MEE loss.
    """
    if y_true.ndim == 1:
        return np.mean(np.sqrt(np.sum(((y_true - y_pred) ** 2))))
    return np.mean(np.sqrt(np.sum(((y_true - y_pred) ** 2), axis=1)))

def mse_prime (y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Derivative of the Mean Squared Error (MSE) loss function.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        np.ndarray: Derivative of MSE loss.
    """
    return np.multiply((-2 / y_true.shape[0]), (y_true -y_pred))

def mee_prime (y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Derivative of the Mean Euclidean Error (MEE) loss function.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        np.ndarray: Derivative of MEE loss.
    """

    if y_true.ndim == 1:
        d = np.sqrt(np.sum(((y_true - y_pred) ** 2)))
        if d == 0:
            return np.zeros(y_true.shape)
        return np.multiply((y_true - y_pred), np.reciprocal(d))

    d = np.sqrt(np.sum(((y_true - y_pred) ** 2), axis=1))
    non_zero_d = d > 0
    gradient = np.zeros(y_true.shape)
    gradient[non_zero_d, :] = np.multiply((y_true[non_zero_d, :] - y_pred[non_zero_d, :]), np.reciprocal(d[non_zero_d] * y_true.shape[0]))
    return -gradient

def pick_loss(loss_type_value: int) -> Tuple[Callable[[np.ndarray], np.float32], Callable[[np.ndarray], np.ndarray]]:
    """
    Pick a loss function based on the loss_type_value.

    Args:
        loss_type_value (int): The value of the loss function type.

    Returns:
        Tuple[Callable[[np.ndarray], np.float32], Callable[[np.ndarray], np.ndarray]]: The loss function and its derivative.
    """
    if loss_type_value == LossFunction.MSE.value:
        return mse, mse_prime
    elif loss_type_value == LossFunction.MEE.value:
        return mee, mee_prime
    else:
        raise ValueError("Invalid loss function type value.")