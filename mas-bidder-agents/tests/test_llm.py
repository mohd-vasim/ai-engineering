import pytest

from mas_bidder_agents.llm import clean_json


def test_clean_simple():
    assert clean_json('{"a": 1}') == {"a": 1}


def test_clean_markdown_json():
    assert clean_json("```json\n{\"a\": 1}\n```") == {"a": 1}


def test_clean_markdown_no_lang():
    assert clean_json("```\n{\"a\": 1}\n```") == {"a": 1}


def test_clean_with_prefix():
    assert clean_json('Here is: {"a": 1}') == {"a": 1}


def test_clean_concatenated():
    assert clean_json('{"a": 1}\n{"b": 2}') == {"a": 1}


def test_clean_garbage_prefix():
    result = clean_json('Some text before\n{"key": "value"}\nmore text')
    assert result == {"key": "value"}


def test_clean_nested():
    result = clean_json('{"outer": {"inner": [1, 2, 3]}}')
    assert result == {"outer": {"inner": [1, 2, 3]}}


def test_clean_raises_on_invalid():
    with pytest.raises(ValueError):
        clean_json("not json at all")
