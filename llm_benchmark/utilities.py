import json
import os
from collections import namedtuple
from logging import config as logging_config
from typing import List, Tuple

import boto3
import smart_open
import yaml

from llm_benchmark.data.data_model import BenchMarkConfig


def initialize_logging() -> None:
    with open("logging_config.yaml") as yaml_fh:
        config_description = yaml.safe_load(yaml_fh)
        logging_config.dictConfig(config_description)


def load_config(path: str) -> BenchMarkConfig:
    with open_file(path=path, mode="r") as f:
        return BenchMarkConfig(**json.load(f))


def get_s3_client(role_arn: str) -> boto3.client:
    sts_client = boto3.client("sts")

    assumedRoleObject = sts_client.assume_role(
        RoleArn=role_arn,
        RoleSessionName="LLMBenchmarkSession",
        DurationSeconds=3600,
    )

    credentials = assumedRoleObject["Credentials"]

    return boto3.client(
        "s3",
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )


def open_file(path: str, mode: str = "r"):
    s3_client = None

    if path.startswith("s3://"):
        role = os.getenv("s3_arn")

        if role is None:
            raise Exception(
                "When using S3 paths you must also set the environment variable "
                "s3_arn to the ARN of the role to assume."
            )

        s3_client = get_s3_client(role_arn=os.environ["s3_arn"])

    return smart_open.open(path, mode=mode, transport_params={"client": s3_client})


def json_part(completion: str):
    """_Summary_
    Return the largest json payload found in the "completion" argument.
    """

    # None completions require special handling, so convert the input
    # to sth meaningful here
    completion = "" if not completion else completion
    Delimiters = namedtuple("Delimiters", ["open", "close"])

    BRACES = Delimiters(open="{", close="}")
    BRACKETS = Delimiters(open="[", close="]")

    def pair_delimiters(delimiters: Delimiters, text) -> List[Tuple[int, int]]:
        """_Summary_
        Return a list of (open, close) delimiter pairs
        sorted descending
        """
        pairs, opos = [], text.find(delimiters.open)

        while opos != -1:
            epos = len(text)
            while (pos := text.rfind(delimiters.close, opos + 1, epos)) != -1:
                pairs.append((opos, (epos := pos) + 1))
            opos = text.find(delimiters.open, opos + 1)
        return pairs

    braces = pair_delimiters(BRACES, text=completion)
    brackets = pair_delimiters(BRACKETS, text=completion)

    delimiters = sorted(
        braces + brackets,
        key=lambda t: (t[1] - t[0], len(completion) - t[0]),
        reverse=True,
    )
    for pair in delimiters:
        try:
            _ = json.loads(
                json_completion := completion[pair[0] : pair[1]], strict=False
            )
            return json_completion
        except json.JSONDecodeError:
            continue
    return ""
