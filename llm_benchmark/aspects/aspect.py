from abc import ABC
from typing import Any, Dict, Optional

from pydantic import BaseModel

from llm_benchmark.data.data_model import LLMResultRecord


class Aspect(ABC, BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def get_id(cls) -> str:
        return cls.__name__

    def process_record(self, record: LLMResultRecord) -> Dict[str, Any]:
        raise NotImplementedError

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        raise NotImplementedError


class BooleanAspect(Aspect):
    counts: Dict[Optional[bool], int] = {}

    def check(self, value) -> Optional[bool]:
        raise NotImplementedError

    def process_record(self, record: LLMResultRecord) -> Dict[str, Any]:
        result = self.check(record)
        self.counts[result] = self.counts.get(result, 0) + 1

        return {
            self.get_id(): result,
        }

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        return {
            f"{self.get_id()}_true": self.counts.get(True, 0),
            f"{self.get_id()}_false": self.counts.get(False, 0),
            f"{self.get_id()}_none": self.counts.get(None, 0),
        }
