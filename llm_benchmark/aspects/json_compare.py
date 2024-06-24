import json
from typing import Any, Dict, List, Literal, Optional

from llm_benchmark.aspects.aspect import Aspect
from llm_benchmark.data.data_model import LLMResultRecord
from llm_benchmark.utilities import json_part

MODE = Literal["any", "all"]


class MissingKey:
    pass


class JsonCompare(Aspect):
    counts: Dict[Optional[bool], int] = {}
    _fields: List[str] = []
    _mode: MODE = "all"
    _strict: bool = True

    def __init__(
        self, fields: List[str], mode: MODE = "all", strict: bool = True, **kwargs
    ):
        super().__init__(**kwargs)

        if not fields:
            raise ValueError("fields param is required and must not be empty")

        self._fields = fields
        self._mode = mode
        self._strict = strict

    def check(self, response: Dict, expected_response: Dict) -> Optional[bool]:
        if self._mode == "all":
            for field in self._fields:
                actual_value = self._get_value(response, field)
                expected_value = self._get_value(expected_response, field)

                if actual_value is MissingKey and expected_value is MissingKey:
                    return None

                if actual_value != expected_value:
                    return False

            return True
        else:
            for field in self._fields:
                actual_value = self._get_value(response, field)
                expected_value = self._get_value(expected_response, field)

                if actual_value is MissingKey and expected_value is MissingKey:
                    return None

                if actual_value == expected_value:
                    return True

            return False

    def process_record(self, record: LLMResultRecord) -> Dict[str, Any]:
        try:
            if self._strict is False:
                response = json.loads(json_part(record.response))
            else:
                response = json.loads(record.response)

            expected_response = json.loads(record.expected_response)
            result = self.check(response, expected_response)
        except Exception:
            result = None

        self._record_result(result)
        return {
            self.get_id(): result,
        }

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        return {
            f"{self.get_id()}_true": self.counts.get(True, 0),
            f"{self.get_id()}_false": self.counts.get(False, 0),
            f"{self.get_id()}_none": self.counts.get(None, 0),
        }

    def _record_result(self, result: Optional[bool]):
        self.counts[result] = self.counts.get(result, 0) + 1

    def _get_value(self, collection: dict, key: str):
        keys = key.split(".")

        for k in keys:
            if isinstance(collection, dict) and k in collection:
                collection = collection[k]
            else:
                return MissingKey

        return collection
