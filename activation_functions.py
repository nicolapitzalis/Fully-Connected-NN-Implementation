import numpy as np

def identity(x):
    """
    Identity activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    return x

def identity_prime(x):
    """
    Derivative of the identity activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    return np.ones_like(x)

def relu(x):
    """
    Rectified Linear Unit (ReLU) activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    return np.maximum(0, x)

def relu_prime(x):
    """
    Derivative of the Rectified Linear Unit (ReLU) activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    return np.where(x > 0, 1, 0)

def leaky_relu(x):
    """
    Leaky ReLU activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    return np.maximum(0.01 * x, x)

def leaky_relu_prime(x):
    """
    Derivative of the Leaky ReLU activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    return np.where(x > 0, 1, 0.01)

def softplus(x):
    """
    Softplus activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    return np.log(1 + np.exp(x))

def softplus_prime(x):
    """
    Derivative of the Softplus activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    return sigmoid(x)

def tanh(x):
    """
    Hyperbolic tangent (tanh) activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    return np.tanh(x)

def tanh_prime(x):
    """
    Derivative of the Hyperbolic tangent (tanh) activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    return 1 - np.tanh(x) ** 2

def sigmoid(x):
    """
    Sigmoid activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    """
    Derivative of the Sigmoid activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    """
    Softmax activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    shifted_exp = np.exp(x - np.max(x))         # Overflow protection
    return shifted_exp / np.sum(shifted_exp)

def softmax_prime(x):
    """
    Derivative of the Softmax activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array.
    """
    f = softmax(x)
    return np.diagflat(f) - np.outer(f, f)    # Jacobian matrix of softmax (s_i - s_iˆ2 in diag, -s_i * s_j in off-diag)


def pick_activation(name: str):
    """
    Returns the activation function and its derivative based on the given name.

    Parameters:
    name (str): The name of the activation function.

    Returns:
    list: A list containing the activation function and its derivative.
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
