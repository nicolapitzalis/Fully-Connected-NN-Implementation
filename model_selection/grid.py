import itertools
import numpy as np
import json
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Any, Dict, List, Tuple
from network.neural_network import NeuralNetwork
from model_selection.validation import kfold_cv

JSON_PATH = 'json_results/'

def grid_step(k_folds: int, data: np.array, target: np.array, combination: Tuple[Any], metrics: List[int], fixed_param: Dict[str, Any], grid_param: Dict[str, List[Any]], file_name_results: str, verbose: bool = False,  plot: bool = False, log_scale: bool = False) -> Tuple[str, Dict[str, float]]:
    """
    Executes a single step of the grid search algorithm.

    Args:
        k_folds (int): The number of folds for cross-validation.
        data (np.array): The input data.
        target (np.array): The target values.
        combination (Tuple[Any]): The combination of parameter values to be tested.
        metrics (List[int]): The list of metrics to evaluate the model.
        fixed_param (Dict[str, Any]): The fixed parameters for the model.
        grid_param (Dict[str, List[Any]]): The grid of parameter values to be tested.
        file_name_results (str): The file name for saving the results.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        plot (bool, optional): Whether to plot the results. Defaults to False.
        log_scale (bool, optional): Whether to use a logarithmic scale for plotting. Defaults to False.

    Returns:
        Tuple[str, Dict[str, float]]: A tuple containing the configuration name and the evaluation results.
    """
    
    parameters_value = dict(zip(grid_param.keys(), combination))
    params = {**fixed_param, **parameters_value}
    
    net = NeuralNetwork(**params)
    
    config_name = '; '.join([f"{key}: {value}" for key, value in parameters_value.items()])
    result = kfold_cv(k_folds, data, target, metrics, net, f"{file_name_results}/{str(combination)}", verbose=False, plot=plot, log_scale=log_scale, parallel_grid=True)
        
    if verbose:
        print(f"\nConfiguration: \n{config_name}")
        for key, value in result.items():
            print(f"{key}: {value}")
        print("------------------------------------------------------")
        print("------------------------------------------------------")
    
    return config_name, result

def grid_search(k_folds: int, data: np.array, target: np.array, metrics: List[int], fixed_param: Dict[str, Any], grid_param: Dict[str, List[Any]], file_name_results: str, verbose: bool = False,  plot: bool = False, log_scale: bool = False) -> List[Dict[str, float]]:
    """
    Executes a grid search algorithm to find the best combination of hyperparameters.

    Args:
        k_folds (int): The number of folds for cross-validation.
        data (np.array): The input data.
        target (np.array): The target values.
        metrics (List[int]): The list of metrics to evaluate the model.
        fixed_param (Dict[str, Any]): The fixed parameters for the model.
        grid_param (Dict[str, List[Any]]): The grid of parameter values to be tested.
        file_name_results (str): The file name for saving the results.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        plot (bool, optional): Whether to plot the results. Defaults to False.
        log_scale (bool, optional): Whether to use a logarithmic scale for plotting. Defaults to False.

    Returns:
        List[Dict[str, float]]: A list of dictionaries containing the configuration name and the evaluation results for each combination of hyperparameters.
    """
    os.makedirs(JSON_PATH, exist_ok=True)
    
    results = {}
    all_combinations = list(itertools.product(*grid_param.values()))

    if verbose:
        print(f"Grid over n_configurations: {len(all_combinations)}")
    
    tasks = (delayed(grid_step)(k_folds, data, target, combination, metrics, fixed_param, grid_param, file_name_results, verbose, plot, log_scale=log_scale) for combination in all_combinations)
    results = Parallel(n_jobs=-1)(tqdm(tasks, total=len(all_combinations)))
            
    config_names, results = zip(*results)
    results = dict(zip(config_names, results))
        
    with open(JSON_PATH + file_name_results + '.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    return results

def get_all_results(results_json: str) -> List[Tuple[str, Dict[str, float]]]:
    """
    Retrieves all the results from a JSON file.

    Args:
        results_json (str): The name of the JSON file containing the results.

    Returns:
        List[Tuple[str, Dict[str, float]]]: A list of tuples containing the configuration name and the evaluation results for each combination of hyperparameters.
    """
    with open(JSON_PATH + results_json, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results.items()

def get_top_n_results(results_json: str, n: int, metric: str, ascending: bool = True) -> List[Tuple[str, Dict[str, float]]]:
    """
    Retrieves the top N results from a JSON file based on a specified metric.

    Args:
        results_json (str): The name of the JSON file containing the results.
        n (int): The number of top results to retrieve.
        metric (str): The metric to use for ranking the results.
        ascending (bool, optional): Whether to sort the results in ascending order. Defaults to True.

    Returns:
        List[Tuple[str, Dict[str, float]]]: A list of tuples containing the configuration name and the evaluation results for each combination of hyperparameters.
    """
    with open(JSON_PATH + results_json, 'r', encoding='utf-8') as f:
        results = json.load(f)
    if ascending:
        sorted_results = sorted(results.items(), key=lambda x: x[1][metric])
    else: 
        sorted_results = sorted(results.items(), key=lambda x: x[1][metric], reverse=True)
    return sorted_results[:n]