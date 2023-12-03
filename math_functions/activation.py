from typing import Callable, List, Tuple
from enum import Enum
import numpy as np

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

def identity(x: np.ndarray) -> np.ndarray:
    """
    Identity activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    return x

def identity_prime(x: np.ndarray) -> np.ndarray:
    """
    Derivative of the identity activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    return np.ones_like(x)

def relu(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit (ReLU) activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    return np.maximum(0, x)

def relu_prime(x: np.ndarray) -> np.ndarray:
    """
    Derivative of the Rectified Linear Unit (ReLU) activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    return np.where(x > 0, 1, 0)

def leaky_relu(x: np.ndarray) -> np.ndarray:
    """
    Leaky ReLU activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    return np.maximum(0.01 * x, x)

def leaky_relu_prime(x: np.ndarray) -> np.ndarray:
    """
    Derivative of the Leaky ReLU activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    return np.where(x > 0, 1, 0.01)

def softplus(x: np.ndarray) -> np.ndarray:
    """
    Softplus activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    return np.log(1 + np.exp(x))

def softplus_prime(x: np.ndarray) -> np.ndarray:
    """
    Derivative of the Softplus activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    return sigmoid(x)

def tanh(x: np.ndarray) -> np.ndarray:
    """
    Hyperbolic tangent (tanh) activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    return np.tanh(x)

def tanh_prime(x: np.ndarray) -> np.ndarray:
    """
    Derivative of the Hyperbolic tangent (tanh) activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    return 1 - np.tanh(x) ** 2

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    """
    Derivative of the Sigmoid activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    shifted_exp = np.exp(x - np.max(x))         # Overflow protection
    return shifted_exp / np.sum(shifted_exp)

def softmax_prime(x: np.ndarray) -> np.ndarray:
    """
    Derivative of the Softmax activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    f = softmax(x)
    return np.diagflat(f) - np.outer(f, f)    # Jacobian matrix of softmax (s_i - s_iˆ2 in diag, -s_i * s_j in off-diag)


def pick_activation(activation_type_value: int) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """
    Pick the activation function and its derivative based on the given activation type value.

    Parameters:
    activation_type_value (int): Value representing the activation function type.

    Returns:
    Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]: Tuple containing the activation function and its derivative.
    """
    if activation_type_value is None:
        return None, None
    if activation_type_value == ActivationFunction.IDENTITY.value:
        return identity, identity_prime
    elif activation_type_value == ActivationFunction.RELU.value:
        return relu, relu_prime
    elif activation_type_value == ActivationFunction.LEAKY_RELU.value:
        return leaky_relu, leaky_relu_prime
    elif activation_type_value == ActivationFunction.SOFTPLUS.value:
        return softplus, softplus_prime
    elif activation_type_value == ActivationFunction.TANH.value:
        return tanh, tanh_prime
    elif activation_type_value == ActivationFunction.SIGMOID.value:
        return sigmoid, sigmoid_prime
    elif activation_type_value == ActivationFunction.SOFTMAX.value:
        return softmax, softmax_prime
    else:
        raise ValueError("Invalid activation function name.")

def pick_function_class(class_value: int) -> List[str]:
    """
    Pick activation functions based on the given class value.

    Parameters:
    class_value (int): Value representing the activation function class.

    Returns:
    List[str]: List of activation function names.
    """
    if class_value == FunctionClassEnum.IDENTITY.value:
        return [identity.__name__]
    elif class_value == FunctionClassEnum.SIGMOIDAL_LIKE.value:
        return [sigmoid.__name__, tanh.__name__, softmax.__name__]
    elif class_value == FunctionClassEnum.RELU_LIKE.value:
        return [relu.__name__, leaky_relu.__name__, softplus.__name__]
    else:
        raise ValueError("Invalid activation function class value.")
