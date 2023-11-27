from typing import Callable, List
from enum import Enum
import numpy as np

class FunctionClassEnum(Enum):
    IDENTITY = 1
    SIGMOIDAL_LIKE = 2
    RELU_LIKE = 3

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


def pick_activation(name: str) -> List[Callable[[np.ndarray], np.ndarray]]:
    """
    Pick activation function and its derivative based on the given name.

    Parameters:
    name (str): Name of the activation function.

    Returns:
    List[Callable[[np.ndarray], np.ndarray]]: List containing the activation function and its derivative.
    """
    activation_dict = {
        identity.__name__: [identity, identity_prime],
        relu.__name__: [relu, relu_prime],
        leaky_relu.__name__: [leaky_relu, leaky_relu_prime],
        softplus.__name__: [softplus, softplus_prime],
        tanh.__name__: [tanh, tanh_prime],
        sigmoid.__name__: [sigmoid, sigmoid_prime],
        softmax.__name__: [softmax, softmax_prime]
    }
    
    return activation_dict[name]

def pick_function_class(class_value: int) -> List[str]:
    """
    Pick activation functions based on the given class value.

    Parameters:
    class_value (int): Value representing the activation function class.

    Returns:
    List[str]: List of activation function names.
    """
    activation_class_dict = {
        FunctionClassEnum.IDENTITY.value: [identity.__name__],
        FunctionClassEnum.SIGMOIDAL_LIKE.value: [sigmoid.__name__, tanh.__name__, softmax.__name__],
        FunctionClassEnum.RELU_LIKE.value: [relu.__name__, leaky_relu.__name__, softplus.__name__]
    }
    
    return activation_class_dict[class_value]
