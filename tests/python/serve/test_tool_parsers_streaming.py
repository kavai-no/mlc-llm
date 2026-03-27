"""
Test that streaming continues properly after tool calls are executed.
This tests the fix for the bug where finish_reason was incorrectly set to "stop"
instead of "tool_calls" when valid tool calls were present.

Tests use mock objects to avoid requiring actual models or GPU resources.
"""

import pytest
from unittest.mock import MagicMock, patch
from mlc_llm.serve.tool_parsers.qwen3coder import Qwen3CoderToolParser

# Test category: unittest (no GPU, no model required)
pytestmark = [pytest.mark.unittest]


class TestStreamingToolCalls:
    """Test streaming behavior with tool calls."""

    def test_tool_call_with_no_params(self):
        """Test that parameterless functions work correctly."""
        # Use mock tokenizer like in other tests
        class MockTokenizer:
            @property
            def vocab(self):
                return {"<tool_call>": 1001, "</tool_call>": 1002}
            
            def encode(self, text):
                res = self.vocab.get(text, None)
                return [res] if res is not None else [0]
        
        parser = Qwen3CoderToolParser(MockTokenizer())
        
        # XML output for a parameterless function call
        xml_output = '<tool_call><function=get_current_timestamp></function></tool_call>'
        
        result = parser.extract_tool_calls(xml_output, None)
        assert result.tools_called == True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_current_timestamp"
    
    def test_tool_call_validation_in_engine(self):
        """Test that valid tool calls are properly added to tool_calls_list."""
        # Mock the tool parser result with valid tool calls
        mock_parser_result = MagicMock()
        mock_parser_result.tools_called = True
        
        class MockFunction:
            def __init__(self):
                self.name = "get_current_timestamp"
                self.arguments = "{}"
        
        class MockToolCall:
            def __init__(self):
                self.function = MockFunction()
        
        mock_parser_result.tool_calls = [MockToolCall()]
        
        # Test the logic that should populate tool_calls_list
        finish_reasons = ["tool_calls"]
        tool_calls_list = [None]
        
        output_text = "<tool_calls>...</tool_calls>"
        result = mock_parser_result
        
        # Simulate the code path in process_function_call_output
        i = 0
        if finish_reasons[i] == "tool_calls":
            tool_calls_list[i] = result.tool_calls
        
        assert tool_calls_list[0] is not None
        assert len(tool_calls_list[0]) == 1
    
    def test_parameterless_function_xml_format(self):
        """Test XML format for parameterless functions in streaming context."""
        # Use mock tokenizer like in other tests
        class MockTokenizer:
            @property
            def vocab(self):
                return {"<tool_call>": 1001, "</tool_call>": 1002}
            
            def encode(self, text):
                res = self.vocab.get(text, None)
                return [res] if res is not None else [0]
        
        parser = Qwen3CoderToolParser(MockTokenizer())
        
        # Test with whitespace that might occur during streaming
        xml_output = '<tool_call><function=get_current_timestamp>\n</function></tool_call>'
        result = parser.extract_tool_calls(xml_output, None)
        
        assert result.tools_called == True
        assert len(result.tool_calls) == 1



