import itertools
import numpy as np
import json
import os
from typing import Any, Dict, List, Tuple
from neural_network import NeuralNetwork
from validation import kfold_cv

JSON_PATH = 'json_results/'

def grid_search(k_folds: int, data: np.array, target: np.array, metrics: List[int], fixed_param: Dict[str, Any], grid_param: Dict[str, List[Any]], file_name_results: str, verbose: bool = False,  plot: bool = False) -> List[Dict[str, float]]:
    os.makedirs(JSON_PATH, exist_ok=True)
    
    results = {}
    all_combinations = list(itertools.product(*grid_param.values()))

    if verbose:
        print(f"Grid over n_configurations: {len(all_combinations)}")
    for combination in all_combinations:
        parameters_value = dict(zip(grid_param.keys(), combination))
        params = {**fixed_param, **parameters_value}
        
        net = NeuralNetwork(**params)
        
        config_name = '; '.join([f"{key}: {value}" for key, value in parameters_value.items()])
        
        result = kfold_cv(k_folds, data, target, metrics, net, f"{file_name_results}/{str(combination)}", verbose=False, plot=plot)
        results[config_name] = result
        
        with open(JSON_PATH + file_name_results + '.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        
        if verbose:
            print(f"\nConfiguration: \n{config_name}")
            for key, value in result.items():
                print(f"{key}: {value}")
            print("------------------------------------------------------")
            print("------------------------------------------------------")


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
