import os
import json

def load_config(dirname=None):
    if dirname is None:
        dirname = os.getcwd()
    config_path = os.path.join(dirname, "config.json")
    with open(config_path, "r") as config_fh:
        config = json.load(config_fh)
    return config