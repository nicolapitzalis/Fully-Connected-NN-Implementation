import numpy as np
from sklearn.utils import shuffle
from typing import Dict, List
from math_functions.function_enums import get_metric_name
from neural_network import NeuralNetwork

INTERNAL_VAL_SPLIT_PERCENTAGE = 0.66

def splitter(data: np.array, target: np.array, percentage: float) -> (np.array, np.array, np.array, np.array):
    split_index = np.ceil(data.shape[0] * percentage).astype(int)
    first_half_data, second_half_data = data[:split_index], data[split_index:]
    first_half_target, second_half_target = target[:split_index], target[split_index:]

    return first_half_data, second_half_data, first_half_target, second_half_target

def kfold_cv(k: int, data: np.array, target: np.array, metrics: List[int], net: NeuralNetwork, verbose : bool = False) -> Dict[str, float]:
    
    tr_losses = []
    internal_val_losses = []
    tr_evals = []
    internal_val_evals = []
    
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

        if net.early_stopping:
            # split to early stopping
            train_data, internal_val_data, train_target, internal_val_target = splitter(percentage=INTERNAL_VAL_SPLIT_PERCENTAGE, data=train_data, target=train_target)
            train_data, train_target = shuffle(train_data, train_target)
            internal_val_data, internal_val_target = shuffle(internal_val_data, internal_val_target)
        else:
            internal_val_data = None
            internal_val_target = None
        
        # NN training
        net.train_net(train_data, train_target, internal_val_data, internal_val_target)

        tr_losses.append(net.training_losses[-1])
        tr_evals.append(net.training_evaluations[-1])
        if net.early_stopping:
            internal_val_losses.append(net.validation_losses[-1])
            internal_val_evals.append(net.validation_evaluations[-1])

        # NN evaluation on validation set
        for metric in metrics:
            metrics_values[metric].append(net.predict_and_evaluate(validation_data, validation_target, metric))
        
        if verbose:
            print(f"\nFold {i+1}:")
            for metric in metrics:
                print(f"Fold validation {get_metric_name(metric)}: {metrics_values[metric][-1]}")
            print(f"Training loss: {tr_losses[-1]}")
            print(f"Training evaluation {get_metric_name(net.evaluation_metric_type_value)}: {tr_evals[-1]}")
            if net.early_stopping:
                print(f"Internal validation loss: {internal_val_losses[-1]}")
                print(f"Internal validation evaluation {get_metric_name(net.evaluation_metric_type_value)}: {internal_val_evals[-1]}")

    # build result dictionary
    if net.early_stopping:
        result_dict = {
            "tr_losses_mean": np.mean(tr_losses),
            "tr_losses_std": np.std(tr_losses),
            "internal_val_losses_mean": np.mean(internal_val_losses),
            "internal_val_losses_std": np.std(internal_val_losses),
            f"tr_evals_{get_metric_name(net.evaluation_metric_type_value)}_mean": np.mean(tr_evals),
            f"tr_evals_{get_metric_name(net.evaluation_metric_type_value)}_std": np.std(tr_evals),
            f"internal_val_evals_{get_metric_name(net.evaluation_metric_type_value)}_mean": np.mean(internal_val_evals),
            f"internal_val_evals_{get_metric_name(net.evaluation_metric_type_value)}_std": np.std(internal_val_evals)
        }
    else:
        result_dict = {
            "tr_losses_mean": np.mean(tr_losses),
            "tr_losses_std": np.std(tr_losses),
            f"tr_evals_{get_metric_name(net.evaluation_metric_type_value)}_mean": np.mean(tr_evals),
            f"tr_evals_{get_metric_name(net.evaluation_metric_type_value)}_std": np.std(tr_evals)
        }

    result_dict.update({"validation_" + get_metric_name(key) + "_mean": np.mean(value) for key, value in metrics_values.items()})
    result_dict.update({"validation_" + get_metric_name(key) + "_std": np.std(value) for key, value in metrics_values.items()})
    
    # return mean metrics
    return result_dict
