import json

from llm_validation.app.configs import ValidationConfig


def load_validation_config(path: str) -> ValidationConfig:
    with open(path=path, mode="r") as f:
        return ValidationConfig(**json.load(f))
