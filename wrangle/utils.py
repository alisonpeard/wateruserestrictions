import os
import json
import pickle

import numpy as np
import matplotlib.pyplot as plt


def load_config(dirname=None):
    if dirname is None:
        dirname = os.getcwd()
    config_path = os.path.join(dirname, "config.json")
    with open(config_path, "r") as config_fh:
        config = json.load(config_fh)
    return config


def save_list(list, path):
    with open(path, "wb") as file:
        pickle.dump(list, file)


def load_list(path):
    with open(path, "rb") as file:
        new_list = pickle.load(file)
    return new_list


def first_file_is_newer(file1:str, file2:str):
    """Return True if file1 was created after file2 and False otherwise."""
    if os.path.exists(file2): 
        return os.path.getctime(file1) > os.path.getctime(file2)
    else:
        return False


def plot_geodataframe(gdf, reset_index=False, sort_columns=None, value_column=None):
    # For testing only

    gdf1 = gdf

    if reset_index:
        gdf1 = gdf1.reset_index()

    if sort_columns is not None:
        gdf1 = gdf1.sort_values(sort_columns, ascending=False)

    if value_column is not None:
        _ = gdf1.plot(value_column, categorical=True)
    else:
        _ = gdf1.plot()

    plt.show()
    plt.close()


def quantile(q):
    def _quantile(x):
        return np.quantile(x, q)
    _quantile.__name__ = "q{:d}".format(int(q * 100))  # assumes only whole number percentiles are used
    return _quantile


def probability_equal_to(threshold):
    def probability(x):
        return np.mean(x == threshold)
    probability.__name__ = "probability_eq_{:d}".format(int(threshold * 100))
    return probability
