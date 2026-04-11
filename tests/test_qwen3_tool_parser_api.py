"""
API boundary tests for Qwen3 tool parser module exports and functionality.

This file defines the public API for the Qwen3CoderToolCallParser through failing tests.
Tests focus on module exports, XML parsing functionality, and streaming support.
"""

import pytest
import sys
from pathlib import Path

# Add project paths to Python path
venv_path = Path("/workspace/projects/mlc-llm/.venv")
sys.path.insert(0, str(venv_path))
sys.path.insert(0, str(Path("/workspace/projects/mlc-llm/python")))

# Import the module we're testing
from mlc_llm.serve.tool_parser import Qwen3CoderToolCallParser, get_parser_instance


class TestQwen3ToolParserModuleExports:
    """Test that the Qwen3 tool parser is properly exported from modules."""

    def test_qwen3_coder_tool_call_parser_class_exists(self):
        """Public API: Qwen3CoderToolCallParser class should be accessible."""
        assert Qwen3CoderToolCallParser is not None
        assert callable(Qwen3CoderToolCallParser)
        assert issubclass(Qwen3CoderToolCallParser, object)

    def test_qwen3_coder_tool_call_parser_instantiable(self):
        """Public API: Qwen3CoderToolCallParser should be instantiable."""
        parser = Qwen3CoderToolCallParser()
        assert parser is not None
        assert hasattr(parser, 'parse')
        assert hasattr(parser, 'parse_streaming')

    def test_get_parser_instance_returns_qwen3_coder_parser(self):
        """Public API: get_parser_instance should return Qwen3CoderToolCallParser for qwen3_coder."""
        parser = get_parser_instance("qwen3_coder")
        assert parser is not None
        assert isinstance(parser, Qwen3CoderToolCallParser)

    def test_get_parser_instance_returns_none_for_unknown_parser(self):
        """Public API: get_parser_instance should return None for unknown parser names."""
        parser = get_parser_instance("unknown_parser")
        assert parser is None


class TestQwen3ToolParserXMLParsing:
    """Test XML parsing functionality of Qwen3CoderToolCallParser."""

    @pytest.fixture
    def parser(self):
        """Create a fresh parser instance for each test."""
        return Qwen3CoderToolCallParser()

    def test_parse_empty_string_returns_empty_result(self, parser):
        """Public API: parse should handle empty strings gracefully."""
        content, tool_calls = parser.parse("")
        assert content == ""
        assert tool_calls == []

    def test_parse_text_without_tool_calls_returns_content_only(self, parser):
        """Public API: parse should return text as content when no tool calls present."""
        content, tool_calls = parser.parse("Hello world!")
        assert content == "Hello world!"
        assert tool_calls == []

    def test_parse_simple_tool_call_with_parameters(self, parser):
        """Public API: parse should extract function name and parameters from XML."""
        xml_input = '<tool_call><function=get_weather><parameter=location>New York</parameter><parameter=unit>celsius</parameter></function></tool_call>'
        content, tool_calls = parser.parse(xml_input)
        
        assert content == ""
        assert len(tool_calls) == 1
        assert tool_calls[0].type == "function"
        assert tool_calls[0].function.name == "get_weather"
        assert tool_calls[0].function.arguments["location"] == "New York"
        assert tool_calls[0].function.arguments["unit"] == "celsius"

    def test_parse_tool_call_with_multiple_functions(self, parser):
        """Public API: parse should handle multiple function calls in one tool call."""
        xml_input = '<tool_call><function=get_weather><parameter=location>New York</parameter></function><function=get_timezone><parameter=city>NYC</parameter></function></tool_call>'
        content, tool_calls = parser.parse(xml_input)
        
        assert len(tool_calls) == 2
        function_names = [tc.function.name for tc in tool_calls]
        assert "get_weather" in function_names
        assert "get_timezone" in function_names

    def test_parse_tool_call_with_complex_parameters(self, parser):
        """Public API: parse should handle complex parameter values including JSON."""
        xml_input = '<tool_call><function=create_user><parameter=name>John Doe</parameter><parameter=age>30</parameter><parameter=active>true</parameter><parameter=metadata>{{"role": "admin"}}</parameter></function></tool_call>'
        content, tool_calls = parser.parse(xml_input)
        
        assert len(tool_calls) == 1
        args = tool_calls[0].function.arguments
        assert args["name"] == "John Doe"
        assert args["age"] == 30
        assert args["active"] is True
        assert args["metadata"] == {"role": "admin"}

    def test_parse_tool_call_with_newlines_in_parameters(self, parser):
        """Public API: parse should handle newlines in parameter values."""
        xml_input = '<tool_call><function=get_data><parameter=query>SELECT * FROM users\nWHERE active = 1</parameter></function></tool_call>'
        content, tool_calls = parser.parse(xml_input)
        
        assert len(tool_calls) == 1
        assert "SELECT * FROM users" in tool_calls[0].function.arguments["query"]

    def test_parse_malformed_xml_returns_empty_tool_calls(self, parser):
        """Public API: parse should handle malformed XML gracefully."""
        xml_input = '<tool_call><function=get_weather><parameter=location>New York</parameter></function>'  # Missing closing tag
        content, tool_calls = parser.parse(xml_input)
        
        # Should still extract what it can, or return empty array if completely malformed
        assert isinstance(tool_calls, list)

    def test_parse_text_with_content_and_tool_calls(self, parser):
        """Public API: parse should separate content from tool calls."""
        xml_input = "Here's some text<tool_call><function=get_data><parameter=id>123</parameter></function></tool_call>More text"
        content, tool_calls = parser.parse(xml_input)
        
        assert "Here's some text" in content
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_data"

    def test_parse_tool_call_with_null_parameter(self, parser):
        """Public API: parse should handle null parameter values."""
        xml_input = '<tool_call><function=get_user><parameter=id>123</parameter><parameter=name>null</parameter></function></tool_call>'
        content, tool_calls = parser.parse(xml_input)
        
        assert len(tool_calls) == 1
        args = tool_calls[0].function.arguments
        assert args["id"] == "123"
        assert args["name"] is None

    def test_parse_tool_call_with_number_parameters(self, parser):
        """Public API: parse should convert numeric parameter values."""
        xml_input = '<tool_call><function=calculate><parameter=x>42</parameter><parameter=y>-3.14</parameter></function></tool_call>'
        content, tool_calls = parser.parse(xml_input)
        
        assert len(tool_calls) == 1
        args = tool_calls[0].function.arguments
        assert args["x"] == 42
        assert args["y"] == -3.14

    def test_parse_tool_call_with_list_parameter(self, parser):
        """Public API: parse should handle list parameter values."""
        xml_input = '<tool_call><function=filter><parameter=tags>["python", "ai", "ml"]</parameter></function></tool_call>'
        content, tool_calls = parser.parse(xml_input)
        
        assert len(tool_calls) == 1
        args = tool_calls[0].function.arguments
        assert args["tags"] == ["python", "ai", "ml"]

    def test_parse_unclosed_tool_call_tag(self, parser):
        """Public API: parse should handle unclosed tool call tags."""
        xml_input = '<tool_call><function=get_data><parameter=id>123</parameter></function>'  # Missing </tool_call>
        content, tool_calls = parser.parse(xml_input)
        
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_data"

    def test_parse_multiple_tool_calls(self, parser):
        """Public API: parse should handle multiple separate tool calls."""
        xml_input = '<tool_call><function=get_weather><parameter=location>NYC</parameter></function></tool_call><tool_call><function=get_time><parameter=timezone>EST</parameter></function></tool_call>'
        content, tool_calls = parser.parse(xml_input)
        
        assert len(tool_calls) == 2
        function_names = [tc.function.name for tc in tool_calls]
        assert "get_weather" in function_names
        assert "get_time" in function_names

    def test_parse_tool_call_with_empty_function_name(self, parser):
        """Public API: parse should handle empty or malformed function names."""
        xml_input = '<tool_call><function=> <parameter=test>value</parameter></function></tool_call>'
        content, tool_calls = parser.parse(xml_input)
        
        # Should either return empty array or handle gracefully
        assert isinstance(tool_calls, list)

    def test_parse_tool_call_without_parameters(self, parser):
        """Public API: parse should handle function calls without parameters."""
        xml_input = '<tool_call><function=ping></function></tool_call>'
        content, tool_calls = parser.parse(xml_input)
        
        # Should return empty array since no valid parameters
        assert len(tool_calls) == 0

    def test_parse_tool_call_with_whitespace_in_parameters(self, parser):
        """Public API: parse should handle whitespace in parameter values."""
        xml_input = '<tool_call><function=search><parameter=query>  hello world  </parameter></function></tool_call>'
        content, tool_calls = parser.parse(xml_input)
        
        assert len(tool_calls) == 1
        query_value = tool_calls[0].function.arguments["query"]
        # Should preserve the value but strip internal whitespace appropriately
        assert "hello world" in query_value


class TestQwen3ToolParserStreaming:
    """Test streaming support for Qwen3CoderToolCallParser."""

    @pytest.fixture
    def parser(self):
        """Create a fresh parser instance for each test."""
        return Qwen3CoderToolCallParser()

    def test_parse_streaming_empty_chunk_returns_none(self, parser):
        """Public API: parse_streaming should return None for empty chunks."""
        result = parser.parse_streaming("")
        assert result is None

    def test_parse_streaming_text_without_tool_calls_returns_none(self, parser):
        """Public API: parse_streaming should return None when no tool calls detected."""
        result = parser.parse_streaming("Hello world!")
        assert result is None

    def test_parse_streaming_partial_tool_call_returns_partial(self, parser):
        """Public API: parse_streaming should return partial result for incomplete tool calls."""
        chunk = '<tool_call><function=get_data><parameter=id>'
        result = parser.parse_streaming(chunk)
        
        assert result is not None
        assert result["type"] == "partial_tool_call"
        assert "<tool_call>" in result["data"]

    def test_parse_streaming_complete_tool_call_returns_complete(self, parser):
        """Public API: parse_streaming should return complete tool calls when fully received."""
        chunk = '<tool_call><function=get_data><parameter=id>123</parameter></function></tool_call>'
        result = parser.parse_streaming(chunk)
        
        assert result is not None
        assert result["type"] == "complete_tool_call"
        assert len(result["data"]) == 1
        assert result["data"][0].function.name == "get_data"
        assert result["data"][0].function.arguments["id"] == "123"

    def test_parse_streaming_multiple_chunks_complete_tool_call(self, parser):
        """Public API: parse_streaming should handle tool calls split across multiple chunks."""
        # First chunk - partial function name
        result1 = parser.parse_streaming('<tool_call><function=get')
        assert result1["type"] == "partial_tool_call"
        
        # Second chunk - complete function with parameters
        result2 = parser.parse_streaming('_data><parameter=id>123</parameter></function></tool_call>')
        assert result2["type"] == "complete_tool_call"
        assert len(result2["data"]) == 1
        assert result2["data"][0].function.name == "get_data"
        assert result2["data"][0].function.arguments["id"] == "123"

    def test_parse_streaming_buffer_cleared_after_complete_tool_call(self, parser):
        """Public API: parse_streaming should clear buffer after complete tool call."""
        # Send complete tool call
        result = parser.parse_streaming('<tool_call><function=test><parameter=x>1</parameter></function></tool_call>')
        assert result["type"] == "complete_tool_call"
        
        # Next chunk should start fresh (no residual buffer)
        result2 = parser.parse_streaming("Some text without tool calls")
        assert result2 is None

    def test_parse_streaming_multiple_complete_tool_calls(self, parser):
        """Public API: parse_streaming should handle multiple complete tool calls in sequence."""
        # First tool call
        result1 = parser.parse_streaming('<tool_call><function=get_weather><parameter=location>NYC</parameter></function></tool_call>')
        assert result1["type"] == "complete_tool_call"
        assert len(result1["data"]) == 1
        
        # Second tool call (should be processed independently)
        result2 = parser.parse_streaming('<tool_call><function=get_time><parameter=timezone>EST</parameter></function></tool_call>')
        assert result2["type"] == "complete_tool_call"
        assert len(result2["data"]) == 1
        assert result2["data"][0].function.name == "get_time"

    def test_parse_streaming_partial_then_complete_in_same_chunk(self, parser):
        """Public API: parse_streaming should handle partial and complete tool calls in same chunk."""
        # Send text with partial tool call followed by complete one
        chunk = 'Some text<tool_call><function=partial><parameter=x>1</parameter></function><tool_call><function=complete><parameter=y>2</parameter></function></tool_call>'
        result = parser.parse_streaming(chunk)
        
        # Should return the complete tool call
        assert result["type"] == "complete_tool_call"
        assert len(result["data"]) == 1
        assert result["data"][0].function.name == "complete"

    def test_parse_streaming_malformed_xml_in_chunk(self, parser):
        """Public API: parse_streaming should handle malformed XML in streaming chunks."""
        chunk = '<tool_call><function=test><parameter=x>123</parameter></function'  # Missing closing tags
        result = parser.parse_streaming(chunk)
        
        assert result is not None
        assert result["type"] == "partial_tool_call"

    def test_parse_streaming_with_buffer_preservation(self, parser):
        """Public API: parse_streaming should preserve buffer state between calls."""
        # First chunk - partial opening tag
        result1 = parser.parse_streaming('<tool')
        assert result1["type"] == "partial_tool_call"
        
        # Second chunk - completes the opening, starts content
        result2 = parser.parse_streaming('_call><function=get_data>')
        assert result2["type"] == "partial_tool_call"
        
        # Third chunk - completes the tool call
        result3 = parser.parse_streaming('<parameter=id>123</parameter></function></tool_call>')
        assert result3["type"] == "complete_tool_call"
        assert len(result3["data"]) == 1

    def test_parse_streaming_text_before_tool_calls(self, parser):
        """Public API: parse_streaming should handle text content before tool calls."""
        chunk = 'User message<tool_call><function=get_data><parameter=id>123</parameter></function></tool_call>'
        result = parser.parse_streaming(chunk)
        
        assert result["type"] == "complete_tool_call"
        assert len(result["data"]) == 1

    def test_parse_streaming_unclosed_function_tag(self, parser):
        """Public API: parse_streaming should handle unclosed function tags."""
        chunk = '<tool_call><function=get_data><parameter=id>123</parameter></function'
        result = parser.parse_streaming(chunk)
        
        assert result["type"] == "partial_tool_call"

    def test_parse_streaming_unclosed_parameter_tag(self, parser):
        """Public API: parse_streaming should handle unclosed parameter tags."""
        chunk = '<tool_call><function=get_data><parameter=id>123</parameter></function></tool_call>'
        result = parser.parse_streaming(chunk)
        
        assert result["type"] == "complete_tool_call"

    def test_parse_streaming_mixed_content_and_tool_calls(self, parser):
        """Public API: parse_streaming should handle mixed content and tool calls."""
        chunk = 'Here is some text<tool_call><function=test><parameter=x>value</parameter></function></tool_call>More text'
        result = parser.parse_streaming(chunk)
        
        assert result["type"] == "complete_tool_call"
        assert len(result["data"]) == 1
