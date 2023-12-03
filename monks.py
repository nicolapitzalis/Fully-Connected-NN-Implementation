import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def read_monk(path: str) -> (np.ndarray, np.ndarray):
    data = pd.read_csv(path, sep=" ", header=None)
    data.drop(data.columns[0], axis=1, inplace=True)
    data.drop(data.columns[-1], axis=1, inplace=True)
    targets = data[data.columns[0]].to_numpy()
    data.drop(data.columns[0], axis=1, inplace=True)
    data = data.to_numpy() 

    data = OneHotEncoder().fit_transform(data).toarray()
    targets = targets.reshape(targets.shape[0], 1)
    return data, targets
