import logging
import re
from typing import List, Literal, Optional, Union

from llm_benchmark.aspects.aspect import BooleanAspect

logger = logging.getLogger(__name__)

MODE = Literal["any", "all"]


class RegexMatch(BooleanAspect):
    _regex: List[str] = []
    _mode: MODE = "any"

    def __init__(self, regex: Union[str, List[str]], mode: MODE = "any", **kwargs):
        super().__init__(**kwargs)

        regex = regex if isinstance(regex, list) else [regex]
        regex = [re.compile(r) for r in regex]

        self._regex = regex
        self._mode = mode

    def check(self, value) -> Optional[bool]:
        try:
            value = value.response
            if self._mode == "any":
                return any(self._matches(value))
            elif self._mode == "all":
                return all(self._matches(value))
        except Exception as exc:
            logger.exception(exc)

        return None

    def _matches(self, value):
        return [bool(re.search(regex, value)) for regex in self._regex]
