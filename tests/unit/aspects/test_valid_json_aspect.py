import pytest

from llm_benchmark.aspects.registry import create_aspect
from llm_benchmark.aspects.valid_json import ValidJson


@pytest.mark.parametrize(
    "value, strict, is_valid",
    [
        ("test", True, False),
        ("another test {}", False, True),
        ('{"key": "value"}', True, True),
        ('{"key": "value", "nested": {}}', True, True),
        ("[]", True, True),
        ('[{"key": "value"}]', True, True),
        ('\n\n[{"key": "value"}]', True, True),
        ('\n\n   [{"key": "value"\n\n}  ]\n\n', False, True),
    ],
)
def test_valid_json_aspect(value, strict, is_valid):
    aspect = create_aspect(aspect_id=ValidJson.get_id(), params={"strict": strict})
    assert aspect.check(value) == is_valid
