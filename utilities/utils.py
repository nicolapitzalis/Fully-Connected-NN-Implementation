import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

plot_source_folder = 'plots/'
plot_destination_folder = 'top_plots/'

def parse_config(config_str):
    """
    Parses a configuration string and returns a dictionary of key-value pairs.

    Args:
        config_str (str): The configuration string in the format "key1: value1; key2: value2; ..."

    Returns:
        dict: A dictionary containing the parsed key-value pairs.
    """
    config_dict = {}
    for param in config_str.split('; '):
        key, value = param.split(': ')
        config_dict[key] = value
    return config_dict

def count_configs(resulting_models: List[Tuple[str, float]]):
    """
    Counts the occurrences of each parameter value in the resulting models.

    Args:
        resulting_models (List[Tuple[str, float]]): A list of tuples containing the model configuration string and its corresponding score.

    Prints:
        The counts for each parameter value.
    """
    param_counts = {}
    for config, _ in resulting_models:
        config_dict = parse_config(config)
        for param, value in config_dict.items():
            if param not in param_counts:
                param_counts[param] = {}
            if value not in param_counts[param]:
                param_counts[param][value] = 0
            param_counts[param][value] += 1

    # Printing the counts for each parameter value
    for param, values in param_counts.items():
        print(f"{param}:")
        for value, count in values.items():
            print(f"  {value}: {count}")

def format_config(config_str: str):
    """
    Formats a configuration string into a more readable format.

    Args:
        config_str (str): The configuration string in the format "key1: value1; key2: value2; ..."

    Returns:
        str: The formatted configuration string in the format "(value1, value2, ...)"
    """
    config_dict = {}
    for param in config_str.split('; '):
        key, value = param.split(': ')
        config_dict[key] = value

    formatted_str = '('
    formatted_str += ', '.join(str(value) for value in config_dict.values())
    formatted_str += ')'
    return formatted_str

def save_top_plots(models: List[Tuple[str, float]], subfolder: str, destination_sub_folder: str):
    """
    Saves the top plots from the models to a destination folder.

    Args:
        models (List[Tuple[str, float]]): A list of tuples containing the model configuration string and its corresponding score.
        subfolder (str): The subfolder name where the plots are located.
        destination_sub_folder (str): The subfolder name where the top plots will be saved.
    """
    formatted_config_strings = [format_config(config) for config, _ in models]
    for config in formatted_config_strings:
        if not os.path.exists(plot_destination_folder):
            os.makedirs(plot_destination_folder)

        shutil.copytree(plot_source_folder + subfolder + '/' + config, plot_destination_folder + destination_sub_folder + '/' + config, dirs_exist_ok=True)

def parse_value(value):
    """
    Parses a value and returns its evaluated form if possible.

    Args:
        value: The value to be parsed.

    Returns:
        The evaluated form of the value if possible, otherwise the value itself.
    """
    try:
        # Attempt to evaluate as a Python literal (works for numbers, lists, tuples, etc.)
        return eval(value)
    except Exception:
        # If eval fails, return the string as it is
        return value

def get_list_models(models: List[Tuple[str, float]]):
    """
    Converts a list of models represented as configuration strings to a list of dictionaries.

    Args:
        models (List[Tuple[str, float]]): A list of tuples containing the model configuration string and its corresponding score.

    Returns:
        List[Dict]: A list of dictionaries representing the models.
    """
    list_models = []
    model = {}
    for model_config, _ in models:
        for param in model_config.split('; '):
            key, value = param.split(': ')
            model[key] = parse_value(value)
        list_models.append(model.copy())
    return list_models

def get_ensemble_models(models: List[Tuple[str, Dict[str, float]]], n: int):
    """
    Retrieves the top N ensemble models from a list of models.

    Args:
        models (List[Tuple[str, Dict[str, float]]]): A list of tuples containing the model configuration string and its corresponding dictionary of scores.
        n (int): The number of top ensemble models to retrieve.

    Returns:
        List[Tuple[str, Dict[str, float]]]: A list of the top N ensemble models.
    """
    models_list = []
    for i, model in enumerate(models):
        if i < n:
            models_list.append(model)
    return models_list

def plot_over_epochs(y_values: list, title: str, y_label: str, y_legend: str, y_prime_values: list = None, y_prime_legend: str = None, yscale: str = None,):
    """
    Plots the values over epochs.

    Args:
        y_values (list): The values to be plotted.
        title (str): The title of the plot.
        y_label (str): The label for the y-axis.
        y_legend (str): The legend for the y-values.
        y_prime_values (list, optional): The values to be plotted as a secondary line. Defaults to None.
        y_prime_legend (str, optional): The legend for the y-prime values. Defaults to None.
        yscale (str, optional): The scale for the y-axis. Defaults to None.
    """
    plt.figure(figsize=(4, 4))
    if yscale == 'log':
        plt.yscale('log')
    plt.plot(list(range(len(y_values))), y_values, label=y_legend, color='blue')
    if y_prime_values is not None:
        plt.plot(list(range(len(y_prime_values))), y_prime_values, label=y_prime_legend, linestyle='--', color='red')
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

def save_array_with_comments(array, file_path, comments=None):
    """
    Saves a numpy array to a CSV file with specified comment lines.

    :param array: numpy.ndarray to be saved.
    :param file_path: Path of the file where the array should be saved.
    :param comments: List of comment lines to be included at the beginning of the file.
    """
    # Ensure comments is a list of strings
    if comments is None:
        comments = []

    # Convert the array to a Pandas DataFrame for easy CSV writing
    df = pd.DataFrame(array)
    
    # Add row enumeration starting from 1
    df = df.reset_index()
    df['index'] += 1  # Increment the index by 1
    df = df.rename(columns={'index': 'Row'})
    
    # Save the DataFrame to a CSV file with comments
    with open(file_path, 'w') as file:
        for comment in comments:
            file.write(f"# {comment}\n")
        df.to_csv(file, index=False, header=False)
