import re
import yaml
import json
from logging import config as logging_config
from typing import Dict, Any

import numpy as np
import wandb


def set_random_state():
    np.random.seed(42)


def initialize_logging():
    with open("logging_config.yaml") as f:
        config_description = yaml.safe_load(f)
        logging_config.dictConfig(config_description)


def wandb_save_file(data: Any, results_path: str, file_name: str):
    with open(f"{results_path}/{file_name}.json", "w+") as f:
        json.dump(data, f, indent=2)
    wandb.save(file_name)


def parse_score_content(content: str) -> Dict:
    content = content.strip()
    pattern = r"<score>(.*?)</score>"
    match = re.search(pattern, content)
    if match:
        score = int(match.group(1))
    else:
        score = 0

    pattern = r"<reason>(.*?)</reason>"
    match = re.search(pattern, content)
    if match:
        reason = match.group(1)
    else:
        reason = "parsing error, check <reason> tag"

    return {"reason": reason, "score": score}


def flatten_dict(d: Dict, parent_key: str = "", sep: str = "_") -> Dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
