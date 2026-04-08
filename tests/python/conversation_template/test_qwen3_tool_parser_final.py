"""
Final test suite for Qwen3 Tool Parser.

Tests:
1. Parse valid XML-style tool calls with functions and parameters.
2. Handle missing or malformed tool calls gracefully.
3. Verify type conversion for parameter values (JSON, booleans, numbers).
4. Test edge cases (empty input, unclosed tags).
"""

import json
import pytest
from mlc_llm.serve.tool_parser import Qwen3CoderToolCallParser


class TestQwen3ToolParser:
    def setup_method(self):
        self.parser = Qwen3CoderToolCallParser()

    def test_parse_valid_tool_call(self):
        """Test parsing a valid XML-style tool call."""
        input_text = (
            "<tool_call>\n"
            "  <function=search_web>\n"
            "    <parameter=query>latest AI advancements</parameter>\n"
            "    <parameter=max_results>5</parameter>\n"
            "  </function>\n"
            "</tool_call>"
        )
        content, tool_calls = self.parser.parse(input_text)

        assert len(tool_calls) == 1
        assert tool_calls[0].type == "function"
        assert tool_calls[0].function.name == "search_web"
        arguments = json.loads(tool_calls[0].function.arguments)
        assert arguments["query"] == "latest AI advancements"
        assert arguments["max_results"] == 5

    def test_parse_malformed_tool_call(self):
        """Test parsing a malformed tool call."""
        input_text = "<tool_call><function=search_web></function></tool_call>"
        content, tool_calls = self.parser.parse(input_text)

        assert len(tool_calls) == 1

    def test_parse_with_type_conversion(self):
        """Test type conversion for parameter values."""
        input_text = (
            "<tool_call>\n"
            "  <function=analyze_data>\n"
            "    <parameter=is_valid>true</parameter>\n"
            "    <parameter=data>[1, 2, 3]</parameter>\n"
            "    <parameter=null_value>null</parameter>\n"
            "  </function>\n"
            "</tool_call>"
        )
        content, tool_calls = self.parser.parse(input_text)

        assert len(tool_calls) == 1
        arguments = json.loads(tool_calls[0].function.arguments)
        assert arguments["is_valid"] is True
        assert arguments["data"] == [1, 2, 3]
        assert arguments["null_value"] is None

    def test_parse_empty_input(self):
        """Test parsing empty input."""
        content, tool_calls = self.parser.parse("")

        assert content == ""
        assert len(tool_calls) == 0

    def test_parse_unclosed_tags(self):
        """Test parsing unclosed tags."""
        input_text = "<tool_call><function=search_web>"
        content, tool_calls = self.parser.parse(input_text)

        assert len(tool_calls) == 1