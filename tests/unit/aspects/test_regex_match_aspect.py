import pytest

from llm_benchmark.aspects.regex_match import RegexMatch
from llm_benchmark.aspects.registry import create_aspect


@pytest.mark.parametrize(
    "regex, value, expected_result",
    [
        (r"\d+", "123", True),
        (r"\d+", "abc", False),
        (r"\d+$", "123abc", False),
        (r"^\d+", "abc123", False),
        (r"\w+\d+", "abc123", True),
        (r"\d+", "abc123abc", True),
    ],
)
def test_single_regex_matches(regex, value, expected_result):
    aspect = create_aspect(aspect_id=RegexMatch.get_id(), params={"regex": regex})

    assert aspect.check(value) is expected_result


@pytest.mark.parametrize(
    "regex, value, expected_result",
    [
        ("[a-z]+", "ABC", False),
        ("(?i)[a-z]+", "ABC", True),
        (r"\w+", "άβγ", True),
        (r"(?a)\w+", "άβγ", False),
    ],
)
def test_supports_regex_flags(regex, value, expected_result):
    aspect = create_aspect(aspect_id=RegexMatch.get_id(), params={"regex": regex})

    assert aspect.check(value) is expected_result


@pytest.mark.parametrize(
    "regex, value, expected_result",
    [
        ([r"\w+"], "abc", True),
        ([r"\d+"], "abc", False),
        ([r"\d+", r"\w+"], "abc", True),
    ],
)
def test_any_match_mode(regex, value, expected_result):
    aspect = create_aspect(
        aspect_id=RegexMatch.get_id(), params={"regex": regex, "mode": "any"}
    )

    assert aspect.check(value) is expected_result


@pytest.mark.parametrize(
    "regex, value, expected_result",
    [
        (r"\w+", "abc", True),
        ([r"\w+"], "abc", True),
        ([r"\d+"], "abc", False),
        ([r"\d+", r"\w+"], "abc", False),
        ([r"\d+", r"\w+"], "123", True),
    ],
)
def test_all_match_mode(regex, value, expected_result):
    aspect = create_aspect(
        aspect_id=RegexMatch.get_id(), params={"regex": regex, "mode": "all"}
    )

    assert aspect.check(value) is expected_result


@pytest.mark.parametrize(
    "regex",
    [
        "[a-",
        r"\2",
    ],
)
def test_invalid_regex(regex):
    with pytest.raises(Exception):
        create_aspect(aspect_id=RegexMatch.get_id(), params={"regex": regex})
