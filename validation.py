import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from typing import Dict, List, Tuple
from math_functions.function_enums import get_metric_name
from neural_network import NeuralNetwork
from joblib import Parallel, delayed

INTERNAL_VAL_SPLIT_PERCENTAGE = 0.8
PLOTS_PATH = 'plots/'

def holdout(data: np.array, target: np.array, holdout_percentage: float, shuffle_set: bool = True) -> (np.array, np.array, np.array, np.array):
    if shuffle_set:
        data, target = shuffle(data, target)
    
    split_index = np.ceil(data.shape[0] * holdout_percentage).astype(int)
    first_half_data, second_half_data = data[:split_index], data[split_index:]
    first_half_target, second_half_target = target[:split_index], target[split_index:]

    return first_half_data, second_half_data, first_half_target, second_half_target

def process_fold(i: int, folds_data: np.array, folds_target: np.array, net: NeuralNetwork, metrics: List[int], verbose: bool = False) -> (int, np.array, np.array, np.array, np.array, Dict[str, float]):
    train_data = []
    train_target = []
    validation_data = []
    validation_target = []
    internal_val_loss = None
    internal_val_eval = None
    metrics_values = {metric: [] for metric in metrics}
    
    # concatenate all folds except the i-th one
    train_data = np.concatenate([folds_data[j] for j in range(len(folds_data)) if j != i])
    train_target = np.concatenate([folds_target[j] for j in range(len(folds_target)) if j != i])
    
    # keeping the ith fold for validation
    validation_data = folds_data[i]
    validation_target = folds_target[i]
    
    if net.early_stopping:
        # split to early stopping
        train_data, internal_val_data, train_target, internal_val_target = holdout(holdout_percentage=INTERNAL_VAL_SPLIT_PERCENTAGE, data=train_data, target=train_target, shuffle_set=True)
    else:
        internal_val_data = None
        internal_val_target = None
        
    # NN training
    net.train_net(train_data, train_target, internal_val_data, internal_val_target)

    # NN training loss and evaluation
    tr_loss = net.training_losses[-1]
    tr_eval = net.training_evaluations[-1]
    if net.early_stopping:
        internal_val_loss = net.validation_losses[-1]
        internal_val_eval = net.validation_evaluations[-1]

    # NN evaluation on validation set
    for metric in metrics:
        metrics_values[metric].append(net.predict_and_evaluate(validation_data, validation_target, metric))

    if verbose:
        print(f"\nFold {i+1}:")
        for metric in metrics:
            print(f"Fold validation {get_metric_name(metric)}: {metrics_values[metric][-1]}")
        print(f"Training loss: {tr_loss}")
        print(f"Training evaluation {get_metric_name(net.evaluation_metric_type_value)}: {tr_eval}")
        if net.early_stopping:
            print(f"Internal validation loss: {internal_val_loss}")
            print(f"Internal validation evaluation {get_metric_name(net.evaluation_metric_type_value)}: {internal_val_eval}")

    return i, net.training_losses[:net.best_epoch], net.training_evaluations[:net.best_epoch], net.validation_losses[:net.best_epoch], net.validation_evaluations[:net.best_epoch], metrics_values

def kfold_cv(k: int, data: np.array, target: np.array, metrics: List[int], net: NeuralNetwork, config_name: str = 'default', verbose: bool = False, plot: bool = False, log_scale: bool = False) -> Dict[str, float]:
   
    tr_losses = []
    internal_val_losses = []
    tr_evals = []
    internal_val_evals = []

    data, target = shuffle(data, target)

    # split data and target in k folds
    folds_data = np.array_split(data, k)
    folds_target = np.array_split(target, k)
    metrics_values = {metric: [] for metric in metrics}
    
    results = Parallel(n_jobs=-1)(delayed(process_fold)(i, folds_data, folds_target, net, metrics, verbose) for i in range(k))
    fold_indexes, tr_losses, tr_evals, internal_val_losses, internal_val_evals, metrics_values = zip(*results)
    
    last_tr_loss = [loss_value[-1] for loss_value in tr_losses]
    last_tr_eval = [eval[-1] for eval in tr_evals]
    if net.early_stopping:
        last_internal_val_loss = [loss_value[-1] for loss_value in internal_val_losses]
        last_internal_val_eval = [eval[-1] for eval in internal_val_evals]

    grouped_metrics = {metric: [] for metric in metrics}
    for metric_dict in metrics_values:
        for key, value in metric_dict.items():
            grouped_metrics[key].append(value)
   
    # build result dictionary
    if net.early_stopping:
        result_dict = {
            "tr_losses_mean": np.mean(last_tr_loss),
            "tr_losses_std": np.std(last_tr_loss),
            "internal_val_losses_mean": np.mean(last_internal_val_loss),
            "internal_val_losses_std": np.std(last_internal_val_loss),
            f"tr_evals_{get_metric_name(net.evaluation_metric_type_value)}_mean": np.mean(last_tr_eval),
            f"tr_evals_{get_metric_name(net.evaluation_metric_type_value)}_std": np.std(last_tr_eval),
            f"internal_val_evals_{get_metric_name(net.evaluation_metric_type_value)}_mean": np.mean(last_internal_val_eval),
            f"internal_val_evals_{get_metric_name(net.evaluation_metric_type_value)}_std": np.std(last_internal_val_eval)
        }
    else:
        result_dict = {
            "tr_losses_mean": np.mean(last_tr_loss),
            "tr_losses_std": np.std(last_tr_loss),
            f"tr_evals_{get_metric_name(net.evaluation_metric_type_value)}_mean": np.mean(last_tr_eval),
            f"tr_evals_{get_metric_name(net.evaluation_metric_type_value)}_std": np.std(last_tr_eval)
        }

    result_dict.update({"validation_" + get_metric_name(key) + "_mean": np.mean(value) for key, value in grouped_metrics.items()})
    result_dict.update({"validation_" + get_metric_name(key) + "_std": np.std(value) for key, value in grouped_metrics.items()})

    if plot:
        os.makedirs(PLOTS_PATH + config_name, exist_ok=True)
        plot_cv_curves(fold_indexes, tr_losses, "Training loss", "Training loss", config_name, "training_loss.png", log_scale=log_scale)
        plot_cv_curves(fold_indexes, tr_evals, "Training evaluation", "Training evaluation", config_name, "training_evaluation.png", log_scale=log_scale)
        if net.early_stopping:
            plot_cv_curves(fold_indexes, internal_val_losses, "Internal validation loss", "Internal validation loss", config_name, "internal_validation_loss.png", log_scale=log_scale)
            plot_cv_curves(fold_indexes, internal_val_evals, "Internal validation evaluation", "Internal validation evaluation", config_name, "internal_validation_evaluation.png", log_scale=log_scale)

    # return mean metrics
    return result_dict

def plot_cv_curves(fold_indexes: Tuple[int], data: np.array, y_label: str, title: str, config_name: str, filename: str, log_scale: bool = False):
    plt.figure(figsize=(8, 6))

    # Find the maximum length of training loss arrays
    max_length = max(len(loss_array) for loss_array in data)

    # Pad shorter arrays
    padded_data = [np.pad(loss_array, (0, max_length - len(loss_array)), 'edge') for loss_array in data]

    # Plot all the curves
    for index in fold_indexes:
        plt.plot(data[index], label=f"Fold {index+1}", alpha=0.3)
    plt.plot(np.mean(padded_data, axis=0), label="Mean", color='black', linestyle='--')

    if log_scale:
        plt.yscale('log')

    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig('plots/' + config_name + '/' + filename)
    plt.close()
