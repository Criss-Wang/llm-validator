import json
from typing import List, Optional

from llm_benchmark.aspects.aspect import BooleanAspect
from llm_benchmark.utilities import json_part


class JsonStructure(BooleanAspect):
    fields: List[str] = []
    strict: bool = True

    def __init__(self, fields: List[str], strict: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.strict = strict

        if not fields:
            raise ValueError("fields param is required and must not be empty")

        self.fields = fields
        self.strict = strict

    def check(self, value) -> Optional[bool]:
        try:
            value = value.response
            if self.strict is False:
                value = json_part(value)
            value = json.loads(value)

            for field in self.fields:
                if not self._key_exists(value, field):
                    return False

            return True
        except json.JSONDecodeError:
            return False
        except Exception:
            return None

    def _key_exists(self, collection, key):
        keys = key.split(".")

        for k in keys:
            if isinstance(collection, dict) and k in collection:
                collection = collection[k]
            else:
                return False

        return True
