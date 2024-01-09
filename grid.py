import itertools
from joblib import Parallel, delayed
import numpy as np
import json
import os
import copy
from typing import Any, Dict, List, Tuple
from neural_network import NeuralNetwork
from validation import kfold_cv

JSON_PATH = 'json_results/'

def grid_step(k_folds: int, data: np.array, target: np.array, combination: Tuple[Any], metrics: List[int], params: Dict[str, Any], file_name_results: str, verbose: bool = False,  plot: bool = False) -> List[Dict[str, float]]:    
    net = NeuralNetwork(**params)
    
    config_name = '; '.join([f"{key}: {value}" for key, value in params.items()])
    result = kfold_cv(k_folds, copy.deepcopy(data), copy.deepcopy(target), metrics, copy.deepcopy(net), f"{file_name_results}/{str(combination)}", verbose=False, plot=plot, parallel_grid=True)
        
    if verbose:
        print(f"\nConfiguration: \n{config_name}")
        for key, value in result.items():
            print(f"{key}: {value}")
        print("------------------------------------------------------")
        print("------------------------------------------------------")
    
    return config_name, result

def grid_search(k_folds: int, data: np.array, target: np.array, metrics: List[int], fixed_param: Dict[str, Any], grid_param: Dict[str, List[Any]], file_name_results: str, verbose: bool = False,  plot: bool = False) -> List[Dict[str, float]]:
    os.makedirs(JSON_PATH, exist_ok=True)
    
    results = {}
    all_combinations = list(itertools.product(*grid_param.values()))

    if verbose:
        print(f"Grid over n_configurations: {len(all_combinations)}")
    
    parameters_value = [copy.deepcopy(dict(zip(grid_param.keys(), combination))) for combination in all_combinations]
    params = [copy.deepcopy({**fixed_param, **param_values}) for param_values in parameters_value]
    results = Parallel(n_jobs=-1)(delayed(grid_step)(k_folds, copy.deepcopy(data), copy.deepcopy(target), combination, metrics, param, file_name_results, verbose, plot) for combination, param in zip(all_combinations, params))
    config_names, results = zip(*results)
    results = dict(zip(config_names, results))
        
    with open(JSON_PATH + file_name_results + '.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    return results

def get_all_results(results_json: str) -> List[Tuple[str, Dict[str, float]]]:
    with open(JSON_PATH + results_json, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results.items()

def get_top_n_results(results_json: str, n: int, metric: str, ascending: bool = True) -> List[Tuple[str, Dict[str, float]]]:
    with open(JSON_PATH + results_json, 'r', encoding='utf-8') as f:
        results = json.load(f)
    if ascending:
        sorted_results = sorted(results.items(), key=lambda x: x[1][metric])
    else: 
        sorted_results = sorted(results.items(), key=lambda x: x[1][metric], reverse=True)
    return sorted_results[:n]