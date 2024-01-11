import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from typing import Dict, List, Tuple
from ensemble import Ensemble
from math_functions.function_enums import Metrics, get_metric_name
from neural_network_utility import evaluate
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

def process_fold(i: int, folds_data: np.array, folds_target: np.array, net: NeuralNetwork, metrics: List[int], verbose: bool = False, parallel_grid: bool = False) -> (int, np.array, np.array, np.array, np.array, Dict[str, float]):
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
    
    if parallel_grid:
        net = copy.deepcopy(net)
    
    if net.early_stopping:
        # split to early stopping
        train_data, internal_val_data, train_target, internal_val_target = holdout(holdout_percentage=INTERNAL_VAL_SPLIT_PERCENTAGE, data=train_data, target=train_target, shuffle_set=True)
    else:
        internal_val_data = None
        internal_val_target = None
        
    # NN training
    net.train_net(train_data, train_target, internal_val_data, internal_val_target)

    # NN training loss and evaluation
    if net.early_stopping:
        tr_loss = net.training_losses[net.best_epoch]
        tr_eval = net.training_evaluations[net.best_epoch]
        internal_val_loss = net.validation_losses[net.best_epoch]
        internal_val_eval = net.validation_evaluations[net.best_epoch]
    else:
        tr_loss = net.training_losses[-1]
        tr_eval = net.training_evaluations[-1]

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

    if net.early_stopping:
        return i, net.training_losses[:net.best_epoch], net.training_evaluations[:net.best_epoch], net.validation_losses[:net.best_epoch], net.validation_evaluations[:net.best_epoch], metrics_values
    else:
        return i, net.training_losses, net.training_evaluations, None, None, metrics_values

def kfold_cv(k: int, data: np.array, target: np.array, metrics: List[int], net: NeuralNetwork, config_name: str = 'default', verbose: bool = False, plot: bool = False, log_scale: bool = False, parallel_grid: bool = False) -> Dict[str, float]:
   
    tr_losses = []
    internal_val_losses = []
    tr_evals = []
    internal_val_evals = []

    data, target = shuffle(data, target)

    # split data and target in k folds
    folds_data = np.array_split(data, k)
    folds_target = np.array_split(target, k)
    metrics_values = {metric: [] for metric in metrics}
    
    results = Parallel(n_jobs=-1)(delayed(process_fold)(i, folds_data, folds_target, net, metrics, verbose, parallel_grid) for i in range(k))
    fold_indexes, tr_losses, tr_evals, internal_val_losses, internal_val_evals, metrics_values = zip(*results)
    
    # last training loss is equiavalent to the best training loss
    last_tr_loss = [loss_value[-1] for loss_value in tr_losses]
    last_tr_eval = [eval[-1] for eval in tr_evals]
    best_epochs = [len(loss_value) for loss_value in tr_losses]
    if net.early_stopping:
        # last internal validation loss is equiavalent to the best internal validation loss
        last_internal_val_loss = [loss_value[-1] for loss_value in internal_val_losses]
        last_internal_val_eval = [eval[-1] for eval in internal_val_evals]

    grouped_metrics = {metric: [] for metric in metrics}
    for metric_dict in metrics_values:
        for key, value in metric_dict.items():
            grouped_metrics[key].append(value)
   
    # build result dictionary
    if net.early_stopping:
        result_dict = {
            "best_epoch_mean": int(np.mean(best_epochs)),
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
        if net.evaluation_metric_type_value == Metrics.ACCURACY.value:
            plot_cv_curves(fold_indexes, tr_evals, "Training evaluation", "Training evaluation", config_name, "training_evaluation.png", log_scale=False)
        else:
            plot_cv_curves(fold_indexes, tr_evals, "Training evaluation", "Training evaluation", config_name, "training_evaluation.png", log_scale=log_scale)
            
        if net.early_stopping:
            plot_cv_curves(fold_indexes, internal_val_losses, "Internal validation loss", "Internal validation loss", config_name, "internal_validation_loss.png", log_scale=log_scale)
            if net.evaluation_metric_type_value == Metrics.ACCURACY.value:
                plot_cv_curves(fold_indexes, internal_val_evals, "Internal validation evaluation", "Internal validation evaluation", config_name, "internal_validation_evaluation.png", log_scale=False)
            else:
                plot_cv_curves(fold_indexes, internal_val_evals, "Internal validation evaluation", "Internal validation evaluation", config_name, "internal_validation_evaluation.png", log_scale=log_scale)

    # return mean metrics
    return result_dict

def process_fold_ensemble(i: int, folds_data: np.array, folds_target: np.array, ensemble: Ensemble, metrics: List[int], verbose: bool = False) -> (int, List[Dict[str, float]], Dict[str, float]):
    train_data = []
    train_target = []
    validation_data = []
    validation_target = []

    ensemble_results = {}
    models_results = []

    # concatenate all folds except the i-th one
    train_data = np.concatenate([folds_data[j] for j in range(len(folds_data)) if j != i])
    train_target = np.concatenate([folds_target[j] for j in range(len(folds_target)) if j != i])
    
    # keeping the ith fold for validation
    validation_data = folds_data[i]
    validation_target = folds_target[i]

    for model_index, model in enumerate(ensemble.models):
        model_results = {}
        if model.early_stopping:
            # split to early stopping
            internal_train_data, internal_val_data, internal_train_target, internal_val_target = holdout(holdout_percentage=INTERNAL_VAL_SPLIT_PERCENTAGE, data=train_data, target=train_target, shuffle_set=True)
        else:
            internal_val_data = None
            internal_val_target = None
            
        # NN training
        model.train_net(internal_train_data, internal_train_target, internal_val_data, internal_val_target)

        # NN training/internal_validation loss and evaluation
        if model.early_stopping:
            model_results['tr_loss'] = model.training_losses[model.best_epoch]
            model_results['tr_eval'] = model.training_evaluations[model.best_epoch]
            model_results['internal_val_loss'] = model.validation_losses[model.best_epoch]
            model_results['internal_val_eval'] = model.validation_evaluations[model.best_epoch]
        else:
            model_results['tr_loss'] = model.training_losses[-1]
            model_results['tr_eval'] = model.training_evaluations[-1]

        # NN evaluation on validation set
        for metric in metrics:
            model_results[get_metric_name(metric)] = model.predict_and_evaluate(validation_data, validation_target, metric)
            model_results[f'{get_metric_name(metric)}_prediction'] = model.predict(validation_data)
        
        if verbose:
            print(f"\nFold {i+1}, model_{model_index+1}:")
            print("Model results:")
            for metric in metrics:
                print(f"Fold validation {get_metric_name(metric)}: {model_results[get_metric_name(metric)]}")
            print(f"Training loss: {model_results['tr_loss']}")
            print(f"Training evaluation {get_metric_name(model.evaluation_metric_type_value)}: {model_results['tr_eval']}")
            if model.early_stopping:
                print(f"Internal validation loss: {model_results['internal_val_loss']}")
                print(f"Internal validation evaluation {get_metric_name(model.evaluation_metric_type_value)}: {model_results['internal_val_eval']}")

        # appending model results
        models_results.append(copy.deepcopy(model_results))

    # ensemble results for the current fold
    for metric in metrics:
        ensemble_prediction = np.mean([model_result[f'{get_metric_name(metric)}_prediction'] for model_result in models_results], axis=0)
        ensemble_results[get_metric_name(metric)] = evaluate(ensemble_prediction, validation_target, metric)

    if verbose:
        print(f"\nFold {i+1}:")
        print("Ensemble results:")
        for metric in metrics:
            print(f"Fold validation {get_metric_name(metric)}: {ensemble_results[get_metric_name(metric)]}")
        print(f"Training loss: {ensemble_results['tr_loss']}")
        print(f"Training evaluation {get_metric_name(ensemble.models[0].evaluation_metric_type_value)}: {ensemble_results['tr_eval']}")
        if ensemble.models[0].early_stopping:
            print(f"Internal validation loss: {ensemble_results['internal_val_loss']}")
            print(f"Internal validation evaluation {get_metric_name(ensemble.models[0].evaluation_metric_type_value)}: {ensemble_results['internal_val_eval']}")

    return i, models_results, ensemble_results

def kfold_cv_ensemble(k: int, data: np.array, target: np.array, metrics: List[int], ensemble: Ensemble, verbose: bool = False) -> Dict[str, float]:
    
    # -----------------------------------------------------------------------------------------------------------------------
    # we are working under the assumption that every model has the same early_stopping value and evaluation_metric_type_value
    # -----------------------------------------------------------------------------------------------------------------------
    models_results = []
    ensembles_results = []
    models_aggregated_results = {
        f"model_{model_idx}": {
            'tr_losses': [], 
            'tr_evals': [], 
            **({'internal_val_losses': [], 'internal_val_evals': []} if ensemble.models[0].early_stopping else {}),
            **{get_metric_name(metric): [] for metric in metrics}
        } for model_idx in range(1, len(ensemble.models)+1)
    }

    # Initialize dictionary to store aggregated ensemble results
    ensemble_aggregated_results = {
        **{get_metric_name(metric): [] for metric in metrics}
    }
    
    kfold_ensemble_result = {}
    kfold_model_result = {}
    
    # shuffle data and target
    data, target = shuffle(data, target)
    
    # split data and target in k folds
    folds_data = np.array_split(data, k)
    folds_target = np.array_split(target, k)

    # Run process_fold_ensemble in parallel and collect results
    results = Parallel(n_jobs=-1)(delayed(process_fold_ensemble)(i, folds_data, folds_target, ensemble, metrics, verbose) for i in range(k))
    _, models_results, ensembles_results = zip(*results)

    # Aggregate results for each model
    for fold_models_results in models_results:
        for model_idx, model_result in enumerate(fold_models_results):
            model_key = f"model_{model_idx + 1}"
            models_aggregated_results[model_key]['tr_losses'].append(model_result['tr_loss'])
            models_aggregated_results[model_key]['tr_evals'].append(model_result['tr_eval'])
            if ensemble.models[0].early_stopping:
                models_aggregated_results[model_key]['internal_val_losses'].append(model_result['internal_val_loss'])
                models_aggregated_results[model_key]['internal_val_evals'].append(model_result['internal_val_eval'])
            for metric in metrics:
                models_aggregated_results[model_key][get_metric_name(metric)].append(model_result[get_metric_name(metric)])

    # Compute mean and standard deviation for each model
    for model_key, model_results in models_aggregated_results.items():
        kfold_model_result[model_key] = {}
        for result, values in model_results.items():
            if result in [get_metric_name(metric) for metric in metrics]:
                result = 'validation_' + result
            kfold_model_result[model_key][result + '_mean'] = np.mean(values)
            kfold_model_result[model_key][result + '_std'] = np.std(values)

    # Aggregate ensemble results across all folds
    for ensemble_result in ensembles_results:
        for metric in metrics:
            ensemble_aggregated_results[get_metric_name(metric)].append(ensemble_result[get_metric_name(metric)])

    # Compute mean and standard deviation for ensemble results
    for result, values in ensemble_aggregated_results.items():
        if result in [get_metric_name(metric) for metric in metrics]:
            result = 'validation_' + result

        kfold_ensemble_result[result + '_mean'] = np.mean(values)
        kfold_ensemble_result[result + '_std'] = np.std(values)

    
    return kfold_model_result, kfold_ensemble_result

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
