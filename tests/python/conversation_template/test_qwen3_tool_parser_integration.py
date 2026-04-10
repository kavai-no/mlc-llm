"""
Integration test suite for Qwen3 Tool Parser.

Tests:
1. Direct parsing of valid XML-style tool calls.
2. Handling malformed or unclosed tags gracefully.
3. Type conversion for parameter values (JSON, booleans, numbers).
"""

import json
from mlc_llm.serve.tool_parser import Qwen3CoderToolCallParser


def test_parse_valid_tool_call():
    """Test parsing a valid XML-style tool call."""
    parser = Qwen3CoderToolCallParser()
    input_text = (
        "<tool_call>\n"
        "  <function=search_web>\n"
        "    <parameter=query>latest AI advancements</parameter>\n"
        "    <parameter=max_results>5</parameter>\n"
        "  </function>\n"
        "</tool_call>"
    )
    content, tool_calls = parser.parse(input_text)

    assert len(tool_calls) == 1
    assert tool_calls[0].type == "function"
    assert tool_calls[0].function.name == "search_web"
    assert tool_calls[0].function.arguments == {"query": "latest AI advancements", "max_results": 5}


def test_parse_malformed_tool_call():
    """Test parsing a malformed tool call."""
    parser = Qwen3CoderToolCallParser()
    input_text = "<tool_call><function=search_web></function></tool_call>"
    content, tool_calls = parser.parse(input_text)

    assert len(tool_calls) == 1


def test_parse_with_type_conversion():
    """Test type conversion for parameter values."""
    parser = Qwen3CoderToolCallParser()
    input_text = (
        "<tool_call>\n"
        "  <function=analyze_data>\n"
        "    <parameter=is_valid>true</parameter>\n"
        "    <parameter=data>[1, 2, 3]</parameter>\n"
        "    <parameter=null_value>null</parameter>\n"
        "  </function>\n"
        "</tool_call>"
    )
    content, tool_calls = parser.parse(input_text)

    assert len(tool_calls) == 1
    arguments = tool_calls[0].function.arguments
    assert arguments["is_valid"] is True
    assert arguments["data"] == [1, 2, 3]
    assert arguments["null_value"] is None


def test_parse_empty_input():
    """Test parsing empty input."""
    parser = Qwen3CoderToolCallParser()
    content, tool_calls = parser.parse("")

    assert content == ""
    assert len(tool_calls) == 0


def test_parse_unclosed_tags():
    """Test parsing unclosed tags."""
    parser = Qwen3CoderToolCallParser()
    input_text = "<tool_call><function=search_web>"
    content, tool_calls = parser.parse(input_text)

    assert len(tool_calls) == 1
