import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def read_monk(path: str) -> (np.ndarray, np.ndarray):
    data = pd.read_csv(path, sep=" ", header=None, usecols=range(1, 8))     #excluding first empty col and ids
    targets = data[data.columns[0]].to_numpy()
    data.drop(data.columns[0], axis=1, inplace=True)
    data = data.to_numpy()
    data = OneHotEncoder().fit_transform(data).toarray()
    targets = targets.reshape(targets.shape[0], 1)
    return data, targets

def read_cup(path: str) -> (np.ndarray, np.ndarray):
    path = 'datasets/cup/CUP_TR.csv'
    data = pd.read_csv(path, sep=",", header=None, comment='#')
    data.drop(data.columns[0], axis=1, inplace=True)
    targets = data[data.columns[-3:]].to_numpy()
    data.drop(data.columns[-3:], axis=1, inplace=True)
    data = data.to_numpy()
    targets = targets.reshape(targets.shape[0], 3)
    return data, targets
