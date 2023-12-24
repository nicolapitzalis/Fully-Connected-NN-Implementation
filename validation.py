import numpy as np
from sklearn.utils import shuffle
from typing import Any, Dict, List
from function_enums import get_metric_name
from neural_network import NeuralNetwork

INTERNAL_VAL_SPLIT_PERCENTAGE = 0.66

def splitter(data: np.array, target: np.array, percentage: float) -> (np.array, np.array, np.array, np.array):
    split_index = np.ceil(int(data.shape[0] * percentage)).astype(int)
    first_half_data, second_half_data = data[:split_index], data[split_index:]
    first_half_target, second_half_target = target[:split_index], target[split_index:]

    return first_half_data, second_half_data, first_half_target, second_half_target

def kfold_cv(k: int, data: np.array, target: np.array, metrics: List[int], cv_verbose : bool = False, **config: Dict[str, Any]) -> (NeuralNetwork, Dict[str, float]):
    data, target = shuffle(data, target)

    # split data and target in k folds
    folds_data = np.array_split(data, k)
    folds_target = np.array_split(target, k)
    metrics_values = {metric: [] for metric in metrics}

    # iterate through every fold
    for i, fold in enumerate(folds_data):
        train_data = []
        train_target = []
        validation_data = []
        validation_target = []

        # concatenate all folds except the i-th one
        train_data = np.concatenate([folds_data[j] for j in range(len(folds_data)) if j != i])
        train_target = np.concatenate([folds_target[j] for j in range(len(folds_target)) if j != i])
        
        # keeping the ith fold for validation
        validation_data = fold
        validation_target = folds_target[i]

        # split to early stopping
        train_data, internal_val_data, train_target, internal_val_target = splitter(percentage=INTERNAL_VAL_SPLIT_PERCENTAGE, data=train_data, target=train_target)
        shuffle(train_data, train_target)
        shuffle(internal_val_data, internal_val_target)

        # NN training
        nn = NeuralNetwork(**config).train_net(train_data, train_target, internal_val_data, internal_val_target)

        # NN evaluation on validation set
        for metric in metrics:
            metrics_values[metric].append(nn.predict_and_evaluate(validation_data, validation_target, metric))
        
        if cv_verbose:
            print(f"\nFold {i+1}:")
            for metric in metrics:
                print(f"{get_metric_name(metric)}: {metrics_values[metric][-1]}")

    # return mean metrics
    return {get_metric_name(key): np.mean(value) for key, value in metrics_values.items()}
