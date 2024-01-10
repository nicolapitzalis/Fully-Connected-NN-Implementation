import os
import shutil
import matplotlib.pyplot as plt
from typing import List, Tuple

plot_source_folder = 'plots/'
plot_destination_folder = 'top_plots/'

def parse_config(config_str):
    config_dict = {}
    for param in config_str.split('; '):
        key, value = param.split(': ')
        config_dict[key] = value
    return config_dict

def count_configs(resulting_models: List[Tuple[str, float]]):
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
    config_dict = {}
    for param in config_str.split('; '):
        key, value = param.split(': ')
        config_dict[key] = value

    formatted_str = '('
    formatted_str += ', '.join(str(value) for value in config_dict.values())
    formatted_str += ')'
    return formatted_str

def save_top_plots(models: List[Tuple[str, float]], subfolder: str, destination_sub_folder: str):
    formatted_config_strings = [format_config(config) for config, _ in models]
    for config in formatted_config_strings:
        if not os.path.exists(plot_destination_folder):
            os.makedirs(plot_destination_folder)

        shutil.copytree(plot_source_folder + subfolder + '/' + config, plot_destination_folder + destination_sub_folder + '/' + config, dirs_exist_ok=True)

def parse_value(value):
    try:
        # Attempt to evaluate as a Python literal (works for numbers, lists, tuples, etc.)
        return eval(value)
    except Exception:
        # If eval fails, return the string as it is
        return value

def get_list_models(models: List[Tuple[str, float]]):
    list_models = []
    model = {}
    for model_config, _ in models:
        for param in model_config.split('; '):
            key, value = param.split(': ')
            model[key] = parse_value(value)
        list_models.append(model.copy())
    return list_models

def plot_over_epochs(y_values: list, title: str, y_label: str, legend: str, yscale: str = None):
    plt.figure(figsize=(4, 4))
    if yscale == 'log':
        plt.yscale('log')
    plt.plot(list(range(len(y_values))), y_values, label=legend)
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel(y_label)
    plt.legend()
    plt.show()
