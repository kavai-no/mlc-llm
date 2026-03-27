"""
Refactored pytest tests for Qwen3CoderToolParser.
Tests verify the fix for parameterless tool calls while maintaining validation.
"""

import json
import pytest
from mlc_llm.serve.tool_parsers.qwen3coder import Qwen3CoderToolParser
from mlc_llm.protocol.openai_api_protocol import (
    ChatTool,
    ChatFunction,
    ChatCompletionRequest
)

# Test category: unittest (no GPU, no model required)
pytestmark = [pytest.mark.unittest]


class MockTokenizer:
    """Mock tokenizer for testing without loading actual model."""
    
    @property
    def vocab(self):
        return {"<tool_call>": 1001, "</tool_call>": 1002}
    
    def encode(self, text):
        res = self.vocab.get(text, None)
        return [res] if res is not None else [0]


@pytest.fixture
def parser():
    """Create a Qwen3CoderToolParser instance for testing."""
    return Qwen3CoderToolParser(MockTokenizer())


class TestParameterlessToolCalls:
    """Test cases for tool calls without parameters (like get_current_timestamp)."""
    
    def test_parameterless_function_no_tool_context(self, parser):
        """Test parameterless function when no tool context is provided."""
        xml_input = '<tool_call><function=get_current_timestamp></function></tool_call>'
        result = parser.extract_tool_calls(xml_input, None)
        
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_current_timestamp"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {}
    
    def test_parameterless_function_with_whitespace(self, parser):
        """Test parameterless function with whitespace in XML."""
        xml_input = '<tool_call><function=get_current_timestamp>\n </function></tool_call>'
        result = parser.extract_tool_calls(xml_input, None)
        
        assert result.tools_called
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {}
    
    def test_parameterless_function_with_tool_context(self, parser):
        """Test parameterless function when tool context is provided."""
        tools = [
            ChatTool(
                type="function",
                function=ChatFunction(
                    name="get_timestamp",
                    description="Get timestamp.",
                    parameters={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                )
            )
        ]
        request = ChatCompletionRequest(model="test", messages=[], tools=tools)
        
        xml_input = '<tool_call><function=get_timestamp></function></tool_call>'
        result = parser.extract_tool_calls(xml_input, request)
        
        assert result.tools_called
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {}
    
    @pytest.mark.bug_report
    def test_bug_report_scenario(self, parser):
        """Test the exact scenario from bug report."""
        xml_input = '<tool_call><function=get_current_timestamp>\n</function></tool_call>'
        result = parser.extract_tool_calls(xml_input, None)
        
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_current_timestamp"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {}, "Bug: parameterless functions should be accepted"


class TestToolValidation:
    """Test cases for validation of required parameters."""
    
    @pytest.fixture
    def tools_with_required_param(self):
        """Create tools with a required parameter."""
        return [
            ChatTool(
                type="function",
                function=ChatFunction(
                    name="search_web",
                    description="Search the web.",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                )
            )
        ]
    
    def test_required_param_provided(self, parser, tools_with_required_param):
        """Test that function with required parameter is accepted."""
        request = ChatCompletionRequest(model="test", messages=[], tools=tools_with_required_param)
        xml_input = ('<tool_call><function=search_web>'
                    '<parameter=query>test query</parameter>'
                    '</function></tool_call>')
        
        result = parser.extract_tool_calls(xml_input, request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0].function.arguments)
        assert "query" in args
    
    def test_required_param_missing(self, parser, tools_with_required_param):
        """Test that function missing required parameter is rejected."""
        request = ChatCompletionRequest(model="test", messages=[], tools=tools_with_required_param)
        xml_input = '<tool_call><function=search_web></function></tool_call>'
        
        result = parser.extract_tool_calls(xml_input, request)
        assert not result.tools_called
        assert len(result.tool_calls) == 0
    
    def test_optional_param_not_required(self, parser):
        """Test that function with optional params can omit them."""
        tools = [
            ChatTool(
                type="function",
                function=ChatFunction(
                    name="search_web2",
                    description="Search the web.",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": []  # No required params
                    }
                )
            )
        ]
        request = ChatCompletionRequest(model="test", messages=[], tools=tools)
        xml_input = '<tool_call><function=search_web2></function></tool_call>'
        
        result = parser.extract_tool_calls(xml_input, request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {}


class TestFunctionCallsWithParameters:
    """Test cases for function calls with actual parameters."""
    
    def test_function_with_single_param(self, parser):
        """Test function call with single parameter."""
        xml_input = '<tool_call><function=search_web><parameter=query>test</parameter></function></tool_call>'
        result = parser.extract_tool_calls(xml_input, None)
        
        assert result.tools_called
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0].function.arguments)
        assert "query" in args
    
    def test_function_with_multiple_params(self, parser):
        """Test function call with multiple parameters."""
        xml_input = ('<tool_call><function=search_web>'
                    '<parameter=query>test</parameter>'
                    '<parameter=count>10</parameter>'
                    '</function></tool_call>')
        result = parser.extract_tool_calls(xml_input, None)
        
        assert result.tools_called
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0].function.arguments)
        assert "query" in args
        assert "count" in args
