import pytest

from llm_benchmark.aspects.json_structure import JsonStructure
from llm_benchmark.aspects.registry import create_aspect


def test_fields_param_is_required():
    with pytest.raises(TypeError):
        create_aspect(aspect_id=JsonStructure.get_id(), params={})

    with pytest.raises(ValueError):
        create_aspect(aspect_id=JsonStructure.get_id(), params={"fields": None})

    with pytest.raises(ValueError):
        create_aspect(aspect_id=JsonStructure.get_id(), params={"fields": []})

    with pytest.raises(TypeError):
        create_aspect(aspect_id=JsonStructure.get_id(), params={"other": "param"})


@pytest.mark.parametrize(
    "expected_fields, strict, value, is_valid",
    [
        (["key"], True, "{key}", False),
        (["key"], False, '"key"', False),
        (["key"], True, "{}", False),
        (["key"], False, 'Some {"key": "value"}', True),
        (["key"], True, '{"other_key": "value"}', False),
        (["key", "other_key"], True, '{"key": "value"}', False),
        (["key"], True, '{"key": "value", "other_key": {}}', True),
        (["key", "other_key"], True, '{"key": "value", "other_key": {}}', True),
        (["key"], True, "[]", False),
        (["key"], True, '[{"key": ""}]', False),
        (["nested.key"], True, '[{"key": ""}]', False),
        (["nested.key"], True, '[{"nested": ""}]', False),
        (["nested.key"], True, '[{"nested": "", "key": ""}]', False),
        (["nested.key"], True, '[{"nested": {}}]', False),
        (["nested.key"], True, '[{"nested": {"other": ""}}]', False),
    ],
)
def test_json_structure_aspect(expected_fields, strict, value, is_valid):
    aspect = create_aspect(
        aspect_id=JsonStructure.get_id(),
        params={"fields": expected_fields, "strict": strict},
    )

    assert aspect.check(value) == is_valid
