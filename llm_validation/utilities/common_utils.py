import os
import yaml
import json
import time
from logging import config as logging_config
from typing import Dict, Any

import numpy as np
import wandb


def initialize_logging():
    with open("logging_config.yaml") as f:
        config_description = yaml.safe_load(f)
        logging_config.dictConfig(config_description)


def wandb_save_file(data: Any, results_path: str, file_name: str):
    with open(f"{results_path}/{file_name}.json", "w") as f:
        json.dump(data, f, indent=2)
    wandb.save(file_name)


def set_random_state():
    np.random.seed(42)


def flatten_dict(d: Dict, parent_key: str = "", sep: str = "_") -> Dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
