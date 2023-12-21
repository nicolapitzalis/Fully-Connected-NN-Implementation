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
    # targets = OneHotEncoder().fit_transform(targets).toarray()
    return data, targets
