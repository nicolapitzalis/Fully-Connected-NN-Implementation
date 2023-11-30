from enum import Enum
from typing import Callable, Tuple
import numpy as np

class LossFunction(Enum):
    """
    Enum class for loss functions.
    """
    MSE = 1
    MEE = 2

def mse (Y_true: np.ndarray, Y_pred: np.ndarray) -> np.float32:
    """
    Mean Squared Error (MSE) loss function.

    Args:
        Y_true (np.ndarray): True values.
        Y_pred (np.ndarray): Predicted values.

    Returns:
        np.float32: MSE loss.
    """
    return np.mean(np.sum((Y_true - Y_pred) ** 2, axis=1))

def mee (Y_true: np.ndarray, Y_pred: np.ndarray) -> np.float32:
    """
    Mean Euclidean Error (MEE) loss function.

    Args:
        Y_true (np.ndarray): True values.
        Y_pred (np.ndarray): Predicted values.

    Returns:
        np.float32: MEE loss.
    """
    return np.mean(np.sqrt(np.sum(((Y_true - Y_pred) ** 2), axis=1)))

def mse_prime (Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """
    Derivative of the Mean Squared Error (MSE) loss function.

    Args:
        Y_true (np.ndarray): True values.
        Y_pred (np.ndarray): Predicted values.

    Returns:
        np.ndarray: Derivative of MSE loss.
    """
    return np.multiply((-2 / Y_true.shape[0]), (Y_true -Y_pred))

def mee_prime (Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """
    Derivative of the Mean Euclidean Error (MEE) loss function.

    Args:
        Y_true (np.ndarray): True values.
        Y_pred (np.ndarray): Predicted values.

    Returns:
        np.ndarray: Derivative of MEE loss.
    """
    d = np.sqrt(np.sum((Y_true - Y_pred) ** 2, axis=1))
    non_zero_d = d > 0
    gradient = np.zeros(Y_true.shape)
    gradient[non_zero_d, :] = (Y_true[non_zero_d, :] - Y_pred[non_zero_d, :]) / (d[non_zero_d] * Y_true.shape[0])
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