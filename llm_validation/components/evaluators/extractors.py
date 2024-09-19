import json
from typing import Any, Callable, Dict, List
from llm_validation.app.configs import ExtractionConfig


def json_key_extractor(key: str) -> Callable[[str], str]:
    def extract(response: str) -> str:
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return data
        return str(data.get(key, "")).strip()

    return extract


def nested_key_extractor(keys: List[str]) -> Callable[[Dict[str, Any]], str]:
    def extract(data: str) -> str:
        try:
            data = json.loads(data)

            for key in keys:
                if isinstance(data, dict) and key in data:
                    data = data[key]
                else:
                    return ""
            return str(data).strip()
        except json.JSONDecodeError:
            return data

    return extract


def split_extractor(delimiter: str, index: int) -> Callable[[str], str]:
    def extract(text: str) -> str:
        parts = text.split(delimiter)
        return parts[index].strip() if 0 <= index < len(parts) else ""

    return extract


def custom_answer_extractor(
    primary_key: str, fallback_key: str
) -> Callable[[Dict[str, Any]], str]:
    def extract(data: Dict[str, Any]) -> str:
        return str(data.get(primary_key, data.get(fallback_key, ""))).strip()

    return extract


EXTRACTORS = {
    "json_key": json_key_extractor,
    "nested_key": nested_key_extractor,
    "split": split_extractor,
    "custom_answer": custom_answer_extractor,
}


def create_extractor(config: ExtractionConfig) -> Callable[[str], str]:
    if not config:
        return lambda x: x

    extractor_type = config.type
    extractor_args = config.args

    if extractor_type not in EXTRACTORS:
        raise ValueError(f"Unknown extractor type: {extractor_type}")

    extractor = EXTRACTORS[extractor_type](**extractor_args)

    def extract(response: str) -> str:
        return extractor(response)

    return extract
