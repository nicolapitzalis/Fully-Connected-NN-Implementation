from math_functions.function_enums import Metrics
from math_functions.loss import mse, mee
import numpy as np

class Ensemble:

    def __init__(self, models: list):
        self.models = models
        self.predictions = []

    def train(self, data: np.array, target: np.array):
        for model in self.models:
            model.train_net(data, target)

    def predict(self, data: np.array) -> np.array:
        self.predictions = [model.predict(data) for model in self.models]
        return np.mean(self.predictions, axis=0)
    
    def evaluate(self, y_pred: np.array, y_true: np.array, metric: Metrics) -> float:
        if metric == Metrics.MSE.value:
            return mse(y_true, y_pred)
        elif metric == Metrics.MEE.value:
            return mee(y_true, y_pred)
        else:
            raise Exception(f"Metric {metric} not supported.")