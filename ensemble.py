from joblib import Parallel, delayed
import numpy as np

class Ensemble:

    def __init__(self, models: list):
        self.models = models
        self.predictions = []

    def train(self, data: np.array, target: np.array, val_data: np.array = None, val_target: np.array = None, tr_stopping_points: list = None):
        self.models = Parallel(n_jobs=-1)(delayed(model.train_net)(data, target, val_data, val_target, tr_loss_stopping_point=tr_stop) for model, tr_stop in zip(self.models, tr_stopping_points))

    def predict(self, data: np.array) -> np.array:
        self.predictions = [model.predict(data) for model in self.models]
        return np.mean(self.predictions, axis=0)