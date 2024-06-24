import logging
import re
from typing import List, Literal, Optional, Union

from llm_benchmark.aspects.aspect import BooleanAspect

logger = logging.getLogger(__name__)

MODE = Literal["any", "all"]


class Classification(BooleanAspect):
    def __init__(
        self,
        labels: Union[List[str], List[bool]],
        expected_bool: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self._labels = labels
        self._expected_bool = expected_bool

    def check(self, value) -> Optional[bool]:
        # Try to match expected reponses (from gpt-4) with current label
        try:
            idx = 0
            # bools = ["True", "False"]
            response = value.response
            for label in self._labels:
                if label in response.lower():
                    # if self._expected_bool:
                    # label = bools[idx]
                    break
                idx += 1

            return (
                value.expected_response.lower() == label.lower()
                if idx < len(self._labels)
                else None
            )
        except Exception as exc:
            logger.exception(exc)

        return None

    # def _matches(self, value):
    #     return [bool(re.search(regex, value)) for regex in self._regex]
