import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def read_monk(path: str) -> (np.ndarray, np.ndarray):
    """
    Reads the MONK dataset from a CSV file.

    Args:
        path (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing the data and targets as numpy arrays.
    """
    data = pd.read_csv(path, sep=" ", header=None, usecols=range(1, 8))     #excluding first empty col and ids
    targets = data[data.columns[0]].to_numpy()
    data.drop(data.columns[0], axis=1, inplace=True)
    data = data.to_numpy()
    data = OneHotEncoder().fit_transform(data).toarray()
    targets = targets.reshape(targets.shape[0], 1)
    return data, targets

def read_cup(path: str) -> (np.ndarray, np.ndarray):
    """
    Reads the CUP dataset from a CSV file.

    Args:
        path (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing the data and targets as numpy arrays.
    """
    data = pd.read_csv(path, sep=",", header=None, comment='#')
    data.drop(data.columns[0], axis=1, inplace=True)
    targets = data[data.columns[-3:]].to_numpy()
    data.drop(data.columns[-3:], axis=1, inplace=True)
    data = data.to_numpy()
    targets = targets.reshape(targets.shape[0], 3)
    return data, targets

def read_cup_ext_test(path: str) -> np.ndarray:
    """
    Reads the CUP extended test dataset from a CSV file.

    Args:
        path (str): The path to the CSV file.

    Returns:
        np.ndarray: The data as a numpy array.
    """
    data = pd.read_csv(path, sep=",", header=None, comment='#')
    data.drop(data.columns[0], axis=1, inplace=True)
    data = data.to_numpy()
    return data

def read_old_cup(path: str) -> (np.ndarray, np.ndarray):
    """
    Reads the old CUP dataset from a CSV file.

    Args:
        path (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing the data and targets as numpy arrays.
    """
    data = pd.read_csv(path, sep=",", header=None, comment='#')
    data.drop(data.columns[0], axis=1, inplace=True)
    targets = data[data.columns[-2:]].to_numpy()
    data.drop(data.columns[-2:], axis=1, inplace=True)
    data = data.to_numpy()
    targets = targets.reshape(targets.shape[0], 2)
    return data, targets