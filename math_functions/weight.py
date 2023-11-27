import numpy as np

def zero_init(input_size: int, output_size: int) -> np.ndarray:
    """
    Initializes the weights with zeros.

    Args:
        input_size (int): The size of the input layer.
        output_size (int): The size of the output layer.

    Returns:
        np.ndarray: The initialized weights.
    """
    return np.zeros((output_size, input_size))

def random_init(input_size: int, output_size: int) -> np.ndarray:
    """
    Initializes the weights with random values from a standard normal distribution.

    Args:
        input_size (int): The size of the input layer.
        output_size (int): The size of the output layer.

    Returns:
        np.ndarray: The initialized weights.
    """
    return np.random.randn(output_size, input_size)

def xavier_init(input_size: int, output_size: int) -> np.ndarray:
    """
    Initializes the weights using the Xavier initialization method.

    Args:
        input_size (int): The size of the input layer.
        output_size (int): The size of the output layer.

    Returns:
        np.ndarray: The initialized weights.
    """
    std = np.sqrt(2.0 / (input_size + output_size))
    return np.random.normal(0, std, size=(output_size, input_size))

def he_init(input_size: int, output_size: int) -> np.ndarray:
    """
    Initializes the weights using the He initialization method.

    Args:
        input_size (int): The size of the input layer.
        output_size (int): The size of the output layer.

    Returns:
        np.ndarray: The initialized weights.
    """
    std = np.sqrt(2.0 / input_size)
    return np.random.randn(output_size, input_size) * std

def lecun_init(input_size: int, output_size: int) -> np.ndarray:
    """
    Initializes the weights using the LeCun initialization method.

    Args:
        input_size (int): The size of the input layer.
        output_size (int): The size of the output layer.

    Returns:
        np.ndarray: The initialized weights.
    """
    std = np.sqrt(1.0 / input_size)
    return np.random.normal(0, std, size=(output_size, input_size))
