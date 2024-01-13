from joblib import Parallel, delayed
import numpy as np

class Ensemble:
    """
    A class representing an ensemble of models.

    Attributes:
        models (list): List of models in the ensemble.
        predictions (list): List of predictions made by the ensemble.

    Methods:
        train(data, target, val_data=None, val_target=None, tr_stopping_points=None):
            Trains the models in the ensemble using the given data and target.
            Optionally, validation data, validation target, and training stopping points can be provided.
        
        predict(data):
            Makes predictions using the ensemble on the given data.
            Returns the average prediction across all models in the ensemble.
    """

    def __init__(self, models: list):
        self.models = models
        self.predictions = []

    def train(self, data: np.array, target: np.array, val_data: np.array = None, val_target: np.array = None, tr_stopping_points: list = None):
        """
        Trains the models in the ensemble using the given data and target.

        Args:
            data (np.array): The training data.
            target (np.array): The target values.
            val_data (np.array, optional): The validation data. Defaults to None.
            val_target (np.array, optional): The validation target values. Defaults to None.
            tr_stopping_points (list, optional): List of training stopping points. Defaults to None.
        """
        self.models = Parallel(n_jobs=-1)(delayed(model.train_net)(data, target, val_data, val_target, tr_loss_stopping_point=tr_stop) for model, tr_stop in zip(self.models, tr_stopping_points))

    def predict(self, data: np.array) -> np.array:
        """
        Makes predictions using the ensemble on the given data.

        Args:
            data (np.array): The input data.

        Returns:
            np.array: The average prediction across all models in the ensemble.
        """
        self.predictions = [model.predict(data) for model in self.models]
        return np.mean(self.predictions, axis=0)