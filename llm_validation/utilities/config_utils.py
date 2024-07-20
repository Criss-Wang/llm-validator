import json

from llm_validation.app.configs import ValidationConfig


def load_validation_config(filename: str) -> ValidationConfig:
    with open(file=filename, mode="r") as f:
        return ValidationConfig(**json.load(f))
