import json
import logging
import re
from typing import Dict, List

import pandas as pd
import numpy as np
import yaml

from llm_benchmark.aspects.aspect import Aspect
from llm_benchmark.aspects.registry import create_aspect
from llm_benchmark.data.data_model import (
    AspectConfig,
    ChatMessage,
    DatasetRecord,
)
from llm_benchmark.utilities import open_file

np.random.seed(42)

logger = logging.getLogger(__name__)


def load_dataset(path: str) -> List[DatasetRecord]:
    with open_file(path=path, mode="r") as f:
        return [DatasetRecord(**record) for record in json.load(f)]


def load_dataset_from_df(
    path: str, prompt: Dict, label_col: str
) -> List[DatasetRecord]:
    df = pd.read_csv(path, index_col=0)[:]
    df = df.rename(columns={label_col: "response"})

    # df = df[-300:]
    # if len(df) > 3:
    #     df = df.sample(2, replace=False)
    prompts = load_prompt(prompt)
    return [adapt_to_record(row, prompts) for _, row in df.iterrows()]


def transform_prompt(raw_prompt: str) -> str:
    """_transform raw prompt in yaml
    (directly copied from db) to python-compatible version
    """
    # replace single curly with double curly
    prompt_str = raw_prompt.replace("{", "{{").replace("}", "}}")

    # replace $var pattern with {var} pattern
    idx_pattern = r'"\$([a-z_A-Z]+)"'
    idx_pattern = re.compile(idx_pattern)
    return idx_pattern.sub(r'"{\1}"', prompt_str)


def load_prompt(prompt: Dict) -> List[Dict]:
    with open(prompt.path) as f:
        prompts = yaml.load(f, Loader=yaml.SafeLoader)
    for curr in prompts:
        if curr["name"] == prompt.name:
            prompt = [
                {
                    "role": "system",
                    "content": transform_prompt(curr["system"]["value"]),
                }
            ]
            if curr["user"]["value"]:
                prompt.append(
                    {"role": "user", "content": transform_prompt(curr["user"]["value"])}
                )
            return prompt
    return []


def adapt_to_record(row: any, prompts: List[Dict]) -> List[DatasetRecord]:
    info = row.to_dict()

    if "query" in info:
        info["OrigUserRequest"] = info["query"]
        info["ConvHistory"] = ""  # f"user: {info['query']}"

    record = DatasetRecord(
        inputs=info,
        messages=[
            ChatMessage(role=prompt["role"], content=prompt["content"].format(**info))
            for prompt in prompts
        ],
        expected_response=str(info["response"]) if "response" in info else "",
    )
    return record


def initialize_aspects(config: List[AspectConfig]) -> List[Aspect]:
    return [
        create_aspect(aspect_id=aspect_config.id, params=aspect_config.params)
        for aspect_config in config
    ]
