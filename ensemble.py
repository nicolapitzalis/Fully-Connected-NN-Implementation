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