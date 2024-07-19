import json
import os

import pytest

from llm_validation.aspects.json_compare import JsonCompare
from llm_validation.aspects.registry import create_aspect
from llm_validation.data.data_model import LLMResultRecord


def _get_record_object(strict):
    response = json.dumps(
        {
            "key": 1,
            "nested": {
                "key": 2,
            },
            "other": None,
        }
    )
    if strict is False:
        response = f"Heading text {response}{os.linesep} trailing text."

    expected_response = json.dumps(
        {
            "key": 1,
            "nested": {
                "key": 2,
            },
            "other": "field",
            "missing": "value",
        }
    )
    return LLMResultRecord(
        response=response,
        expected_response=expected_response,
        end_time=0,
        start_time=0,
        number_of_tokens=1,
        prompt="",
        request={},
        success=True,
        time_to_first_token=1,
        total_time=1,
        tokens_per_second=1,
    )


@pytest.mark.parametrize(
    "fields, strict, expected_result",
    [
        (["key"], True, True),
        (["nested.key"], False, True),
        (["other"], True, False),
        (["key", "nested.key"], True, True),
        (["key", "missing"], False, False),
        (["random", "other"], True, None),
        (["missing"], False, False),
        (["other-key"], False, None),
    ],
)
def test_all_mode(fields, strict, expected_result):
    aspect = create_aspect(
        aspect_id=JsonCompare.get_id(),
        params={
            "fields": fields,
            "strict": strict,
            "mode": "all",
        },
    )

    value = _get_record_object(strict)
    result = aspect.process_record(value)

    assert result[JsonCompare.get_id()] is expected_result


@pytest.mark.parametrize(
    "fields, strict, expected_result",
    [
        (["key"], False, True),
        (["nested.key"], True, True),
        (["other"], False, False),
        (["key", "nested.key"], True, True),
        (["key", "missing"], False, True),
        (["random", "other"], True, None),
        (["missing"], False, False),
        (["other-key"], True, None),
    ],
)
def test_any_mode(fields, strict, expected_result):
    aspect = create_aspect(
        aspect_id=JsonCompare.get_id(),
        params={
            "fields": fields,
            "strict": strict,
            "mode": "any",
        },
    )

    value = _get_record_object(strict)
    result = aspect.process_record(value)

    assert result[JsonCompare.get_id()] is expected_result
