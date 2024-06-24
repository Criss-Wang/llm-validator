import json
from typing import Optional

from llm_benchmark.aspects.aspect import BooleanAspect
from llm_benchmark.utilities import json_part


class ValidJson(BooleanAspect):
    _strict: bool = True

    def __init__(self, strict: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._strict = strict

    def check(self, value) -> Optional[bool]:
        try:
            value = value.response
            if self._strict is False:
                value = json_part(value)
            json.loads(value)
            return True
        except json.JSONDecodeError:
            return False
