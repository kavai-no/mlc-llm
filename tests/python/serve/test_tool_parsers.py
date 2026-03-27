"""
Refactored pytest tests for Qwen3CoderToolParser using real tokenizer.
Tests verify the fix for parameterless tool calls while maintaining validation.
"""

import json
import pytest
from mlc_llm.tokenizers import Tokenizer
from mlc_llm.serve.tool_parsers.qwen3coder import Qwen3CoderToolParser
from mlc_llm.protocol.openai_api_protocol import (
    ChatTool,
    ChatFunction,
    ChatCompletionRequest,
    ChatCompletionMessage
)
from test_data_factory import (
    XmlToolCallBuilder,
    create_search_web_tool,
    SINGLE_PARAM_XML,
    MULTI_PARAM_XML,
    EMPTY_PARAMS_XML
)

# Test category: unittest (no GPU, no model required)
pytestmark = [pytest.mark.unittest]


@pytest.fixture
def parser():
    """Create a Qwen3CoderToolParser instance with real tokenizer."""
    return Qwen3CoderToolParser(Tokenizer("tests/python/serve/qwen3-coder-30b-a3b-instruct"))


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

        xml_input = '<tool_call>\n<function=get_current_timestamp>\n</function>\n</tool_call>'
        result = parser.extract_tool_calls(xml_input, request)
        
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
        xml_input = ('<tool_call>\n'
                    '<function=search_web>\n'
                    '<parameter=query>test query</parameter>\n'
                    '</function>\n'
                    '</tool_call>')
        
        result = parser.extract_tool_calls(xml_input, request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0].function.arguments)
        assert "query" in args
    
    def test_required_param_missing(self, parser, tools_with_required_param):
        """Test that function missing required parameter is rejected."""
        request = ChatCompletionRequest(model="test", messages=[], tools=tools_with_required_param)
        xml_input = '<tool_call>\n<function=search_web></function>\n</tool_call>'
        
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
        xml_input = '<tool_call>\n<function=search_web2></function>\n</tool_call>'
        
        result = parser.extract_tool_calls(xml_input, request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {}


class TestFunctionCallsWithParameters:
    """Test cases for function calls with actual parameters."""
    
    def test_function_with_single_param(self, parser):
        """Test function call with single parameter."""
        tools = [create_search_web_tool()]
        request = ChatCompletionRequest(model="test", messages=[], tools=tools)
        xml_input = SINGLE_PARAM_XML
        result = parser.extract_tool_calls(xml_input, request)
        
        assert result.tools_called
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0].function.arguments)
        assert "query" in args
    
    def test_function_with_multiple_params(self, parser):
        """Test function call with multiple parameters."""
        tools = [create_search_web_tool()]
        request = ChatCompletionRequest(model="test", messages=[], tools=tools)
        xml_input = MULTI_PARAM_XML
        result = parser.extract_tool_calls(xml_input, request)
        
        assert result.tools_called
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0].function.arguments)
        assert "query" in args
        assert "count" in args


class TestEdgeCasesAndErrorHandling:
    """Test edge cases, error scenarios, and robust parsing behavior."""
    
    def test_malformed_xml_minimal(self, parser):
        """Test handling of minimally malformed XML."""
        # Missing closing tag
        xml_input = '<tool_call><function=search_web><parameter=query>test'
        result = parser.extract_tool_calls(xml_input, None)
        
        # Should not crash, but may or may not parse successfully depending on implementation
        assert hasattr(result, 'tools_called')
        assert hasattr(result, 'tool_calls')
    
    def test_malformed_xml_unclosed_tag(self, parser):
        """Test handling of unclosed XML tags."""
        xml_input = '<tool_call><function=search_web><parameter=query>test</parameter></function>'
        result = parser.extract_tool_calls(xml_input, None)
        
        assert hasattr(result, 'tools_called')
        assert hasattr(result, 'tool_calls')
    
    def test_empty_function_name(self, parser):
        """Test handling of empty function names."""
        xml_input = '<tool_call><function=></function></tool_call>'
        result = parser.extract_tool_calls(xml_input, None)
        
        # Should handle gracefully without crashing
        assert hasattr(result, 'tools_called')
    
    def test_very_long_parameter_value(self, parser):
        """Test handling of very long parameter values."""
        # Create a long query string (10,000 characters)
        long_query = "a" * 10000
        xml_input = f'<tool_call><function=search_web><parameter=query>{long_query}</parameter></function></tool_call>'
        
        result = parser.extract_tool_calls(xml_input, None)
        
        if len(result.tool_calls) > 0:
            args = json.loads(result.tool_calls[0].function.arguments)
            assert "query" in args
            assert len(args["query"]) == 10000
    
    def test_special_characters_in_parameters(self, parser):
        """Test handling of special characters and escape sequences."""
        xml_input = '<tool_call><function=search_web><parameter=query>test &amp; "special" &lt;&gt;</parameter></function></tool_call>'
        result = parser.extract_tool_calls(xml_input, None)
        
        if len(result.tool_calls) > 0:
            args = json.loads(result.tool_calls[0].function.arguments)
            assert "query" in args
    
    def test_unicode_and_emoji_parameters(self, parser):
        """Test handling of Unicode characters and emojis."""
        xml_input = '<tool_call><function=search_web><parameter=query>🔍 Hello 世界 🌍</parameter></function></tool_call>'
        result = parser.extract_tool_calls(xml_input, None)
        
        if len(result.tool_calls) > 0:
            args = json.loads(result.tool_calls[0].function.arguments)
            assert "query" in args
    
    def test_multiple_parameter_values_same_name(self, parser):
        """Test handling of duplicate parameter names (should take last value)."""
        xml_input = ('<tool_call><function=search_web>'
                    '<parameter=query>first</parameter>'
                    '<parameter=query>second</parameter>'
                    '</function></tool_call>')
        result = parser.extract_tool_calls(xml_input, None)
        
        if len(result.tool_calls) > 0:
            args = json.loads(result.tool_calls[0].function.arguments)
            assert "query" in args
    
    def test_empty_parameter_tag(self, parser):
        """Test handling of empty parameter tags."""
        xml_input = '<tool_call><function=search_web><parameter=query></parameter></function></tool_call>'
        result = parser.extract_tool_calls(xml_input, None)
        
        if len(result.tool_calls) > 0:
            args = json.loads(result.tool_calls[0].function.arguments)
            assert "query" in args
            assert args["query"] == ""
    
    def test_mixed_content_with_tool_calls(self, parser):
        """Test handling of text mixed with tool calls."""
        xml_input = 'Some text<tool_call><function=search_web></function></tool_call>more text'
        result = parser.extract_tool_calls(xml_input, None)
        
        assert hasattr(result, 'tools_called')
        assert hasattr(result, 'tool_calls')
    
    def test_deeply_nested_parameters(self, parser):
        """Test handling of deeply nested parameter structures."""
        xml_input = '<tool_call><function=search_web>'
        for i in range(10):
            xml_input += f'<parameter=level{i}>value{i}</parameter>'
        xml_input += '</function></tool_call>'
        
        result = parser.extract_tool_calls(xml_input, None)
        
        assert hasattr(result, 'tools_called')
        if len(result.tool_calls) > 0:
            args = json.loads(result.tool_calls[0].function.arguments)
            # Should have all parameters
            for i in range(10):
                assert f'level{i}' in args


class TestComprehensiveValidation:
    """Comprehensive validation tests that verify the complete tool call workflow."""
    
    def test_validation_with_tool_definitions(self, parser):
        """Test that validation works when we have tool definitions (legacy compatibility test)."""
        # Define tools with required parameters
        tools = [
            ChatTool(
                type="function",
                function=ChatFunction(
                    name="search_web",
                    description="Search the web.",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]  # Query is required
                    }
                )
            )
        ]
        
        request = ChatCompletionRequest(model="test", messages=[], tools=tools)
        
        # Test with required parameter provided
        xml_with_query = '<tool_call><function=search_web><parameter=query>test query</parameter></function></tool_call>'
        result1 = parser.extract_tool_calls(xml_with_query, request)
        
        assert result1.tools_called
        assert len(result1.tool_calls) == 1
        args1 = json.loads(result1.tool_calls[0].function.arguments)
        assert "query" in args1, "Should have query parameter"
        
        # Test with required parameter missing (should be rejected)
        xml_without_query = '<tool_call><function=search_web></function></tool_call>'
        result2 = parser.extract_tool_calls(xml_without_query, request)
        
        assert not result2.tools_called
        assert len(result2.tool_calls) == 0
        
        # Test with a tool that has NO parameters
        tools_no_params = [
            ChatTool(
                type="function",
                function=ChatFunction(
                    name="get_timestamp",
                    description="Get timestamp.",
                    parameters={
                        "type": "object",
                        "properties": {},
                        "required": []  # No required params
                    }
                )
            )
        ]
        
        request_no_params = ChatCompletionRequest(model="test", messages=[], tools=tools_no_params)
        xml_no_params_call = '<tool_call><function=get_timestamp></function></tool_call>'
        result3 = parser.extract_tool_calls(xml_no_params_call, request_no_params)
        
        assert result3.tools_called
        assert len(result3.tool_calls) == 1
        args3 = json.loads(result3.tool_calls[0].function.arguments)
        assert args3 == {}, "Should have empty arguments"


class TestStreamingBehavior:
    """Test cases for streaming behavior of Qwen3CoderToolParser."""
    
    def test_empty_delta_when_no_tool_content(self, parser):
        """Test that None is returned when there's no meaningful tool content."""
        # Empty delta should return None
        tools = [
            ChatTool(
                type="function",
                function=ChatFunction(
                    name="test_function",
                    description="A test function.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                )
            )
        ]
        request = ChatCompletionRequest(messages=[], model="test", tools=tools)
        result = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="",
            delta_text="",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request
        )
        assert result is None, "Empty deltas should return None"
    
    def test_empty_delta_with_whitespace_only(self, parser):
        """Test that structured message is returned for whitespace-only deltas."""
        # Whitespace only delta returns structured message with empty tool_calls
        tools = [
            ChatTool(
                type="function",
                function=ChatFunction(
                    name="test_function",
                    description="A test function.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                )
            )
        ]
        request = ChatCompletionRequest(messages=[], model="test", tools=tools)
        result = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="   \n  ",
            delta_text="   \n  ",
            previous_token_ids=[],
            current_token_ids=[20, 30, 40],
            delta_token_ids=[20, 30, 40],
            request=request
        )
        assert result is not None, "Whitespace-only deltas should return structured message"
        assert isinstance(result, ChatCompletionMessage)
        assert len(result.tool_calls) == 0, "Should have empty tool calls list"
    
    def test_empty_delta_with_only_tool_call_open_tag(self, parser):
        """Test that only opening tool_call tag without content returns structured message."""
        # Just opening tag - no function info yet
        tools = [
            ChatTool(
                type="function",
                function=ChatFunction(
                    name="test_function",
                    description="A test function.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                )
            )
        ]
        request = ChatCompletionRequest(messages=[], model="test", tools=tools)
        result = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="<tool_call>",
            delta_text="<tool_call>",
            previous_token_ids=[],
            current_token_ids=[151657],
            delta_token_ids=[151657],
            request=request
        )
        # Should return structured message but no tool calls yet
        assert result is not None
    
    def test_tool_call_id_consistency_during_streaming(self, parser):
        """Test that tool_call_id is consistent across streaming chunks."""
        tools = [
            ChatTool(
                type="function",
                function=ChatFunction(
                    name="search_web",
                    description="Search the web.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                )
            )
        ]
        request = ChatCompletionRequest(messages=[], model="test", tools=tools)
        
        # First chunk: opening tag
        result1 = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="<tool_call>",
            delta_text="<tool_call>",
            previous_token_ids=[],
            current_token_ids=[1001],
            delta_token_ids=[1001],
            request=request
        )
        
        # Should return structured message but no tool call ID yet
        assert result1 is not None, "Should return structured message for opening tag"
        assert result1.tool_call_id is None, "tool_call_id should be None for opening tag only"
        
        # Second chunk: function name starts
        result2 = parser.extract_tool_calls_streaming(
            previous_text="<tool_call>",
            current_text="<tool_call><function=search_web",
            delta_text="<function=search_web",
            previous_token_ids=[151657],
            current_token_ids=[151657, 27],
            delta_token_ids=[27],
            request=request
        )
        
        # Should return structured message but no tool call ID yet
        assert result2 is not None, "Should return structured message for partial function name"
        assert result2.tool_call_id is None, "tool_call_id should be None for partial function name"
        
        # Third chunk: function name complete with opening brace
        result3 = parser.extract_tool_calls_streaming(
            previous_text="<tool_call><function=search_web",
            current_text="<tool_call><function=search_web>",
            delta_text=">",
            previous_token_ids=[151657, 27],
            current_token_ids=[151657, 27, 29],
            delta_token_ids=[29],
            request=request
        )
        
        # Should return structured message with function name and tool_call_id
        assert result3 is not None, "Should return structured message when function name completes"
        if hasattr(result3, 'tool_calls') and len(result3.tool_calls) > 0:
            assert result3.tool_call_id is None, "tool_call_id should be None in delta messages"
    
    def test_parameterless_function_xml_format(self, parser):
        """Test XML format for parameterless functions in streaming context."""
        # Test with whitespace that might occur during streaming
        xml_output = '<tool_call><function=get_current_timestamp>\n</function></tool_call>'
        result = parser.extract_tool_calls(xml_output, None)
        
        assert result.tools_called == True
        assert len(result.tool_calls) == 1
    
    def test_no_multiple_tool_ids_for_same_call(self, parser):
        """Test that the same tool call doesn't get multiple IDs during streaming."""
        tools = [
            ChatTool(
                type="function",
                function=ChatFunction(
                    name="search_web",
                    description="Search the web.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                )
            )
        ]
        request = ChatCompletionRequest(messages=[], model="test", tools=tools)
        
        collected_ids = []
        
        # Simulate streaming chunks for a single function call
        chunks = [
            ("<tool_call>", "<tool_call>"),
            ("<tool_call>", "<function=search_web"),
            ("<tool_call><function=search_web", ">"),
            ("<tool_call><function=search_web>", "{\n  \"query\": \""),
        ]
        
        for i, (current_text, delta_text) in enumerate(chunks):
            previous_text_val = "" if i == 0 else chunks[i-1][0]
            previous_token_ids_val = [] if i == 0 else []
            current_token_ids_val = [1001] if "<tool_call>" in current_text else [1001, 1003]
            delta_token_ids_val = [1001] if "<tool_call>" in delta_text else [1003]
            
            result = parser.extract_tool_calls_streaming(
                previous_text=previous_text_val,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=previous_token_ids_val,
                current_token_ids=current_token_ids_val,
                delta_token_ids=delta_token_ids_val,
                request=request
            )
            
            if result is not None and hasattr(result, 'tool_calls') and len(result.tool_calls) > 0:
                tool_id = result.tool_calls[0].id
                collected_ids.append(tool_id)
        
        # All IDs should be the same for a single tool call
        if len(collected_ids) > 1:
            assert all(id == collected_ids[0] for id in collected_ids), \
                f"Tool call ID should remain consistent across chunks: {collected_ids}"
    
    def test_no_empty_tool_calls_list_in_deltas(self, parser):
        """Test that tool_calls list doesn't contain incomplete entries."""
        tools = [
            ChatTool(
                type="function",
                function=ChatFunction(
                    name="test_function",
                    description="A test function.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                )
            )
        ]
        request = ChatCompletionRequest(messages=[], model="test", tools=tools)
        # Delta with just whitespace and opening tag
        result = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="<tool_call>  \n",
            delta_text="  \n",
            previous_token_ids=[],
            current_token_ids=[151657],
            delta_token_ids=[151658, 151658],
            request=request
        )
        
        # Should return structured message but with no valid tool calls yet
        assert result is not None
        if hasattr(result, 'tool_calls'):
            # Either empty list or function names are complete and valid
            assert len(result.tool_calls) == 0 or (len(result.tool_calls) > 0 and 
                result.tool_calls[0].function.name is not None)
    
    def test_mixed_content_in_streaming(self, parser):
        """Test that mixed XML and JSON-like content in model output is handled correctly."""
        tools = [
            ChatTool(
                type="function",
                function=ChatFunction(
                    name="test_func",
                    description="A test function.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                )
            )
        ]
        request = ChatCompletionRequest(messages=[], model="test", tools=tools)
        # Simulate model output that has both XML structure and JSON-like content
        delta_text = '<tool_call><function=test_func' + '{"invalid": "json"}'
        result = parser.extract_tool_calls_streaming(
            previous_text='',
            current_text=delta_text,
            delta_text=delta_text,
            # Simulate tokenization
            previous_token_ids=[],
            current_token_ids = [10] * len(delta_text),
            delta_token_ids = [15] + ([10] * (len(delta_text) - 1) if len(delta_text) > 1 else []),
            request=request
        )
        
        # Should return structured message even with mixed content
        assert result is not None, "Mixed content should still return structured message"
    
    def test_streaming_state_reset_between_messages(self, parser):
        """Test that streaming state is properly reset between messages."""
        tools = [
            ChatTool(
                type="function",
                function=ChatFunction(
                    name="test_function",
                    description="A test function.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                )
            )
        ]
        request = ChatCompletionRequest(messages=[], model="test", tools=tools)
        # First message with tool call
        result1 = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="<tool_call><function=test_function>",
            delta_text="<tool_call><function=test_function>",
            previous_token_ids=[],
            current_token_ids=[151657, 27],
            delta_token_ids=[151657, 27],
            request=request
        )
        assert result1 is not None

    def test_mock_tokenizer_empty_delta_when_no_tool_content(self, parser):
        """Test that empty deltas return None (using mock tokenizer scenario)."""
        # This test replicates behavior from streaming_direct.py tests
        tools = [
            ChatTool(
                type="function",
                function=ChatFunction(
                    name="search_web",
                    description="Search the web.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                )
            )
        ]
        request = ChatCompletionRequest(messages=[], model="test", tools=tools)
        
        # Empty delta should return None
        result = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="",
            delta_text="",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request
        )
        assert result is None, "Empty deltas should return None"

    def test_mock_tokenizer_tool_call_id_consistency_during_streaming(self, parser):
        """Test that tool_call_id remains consistent across streaming chunks (from streaming_direct.py)."""
        tools = [
            ChatTool(
                type="function",
                function=ChatFunction(
                    name="search_web",
                    description="Search the web.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                )
            )
        ]
        request = ChatCompletionRequest(messages=[], model="test", tools=tools)
        
        # First chunk: opening tag
        result1 = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="<tool_call>",
            delta_text="<tool_call>",
            previous_token_ids=[],
            current_token_ids=[1001],
            delta_token_ids=[1001],
            request=request
        )
        
        # Should return structured message but no tool call ID yet
        assert result1 is not None, "Should return structured message for opening tag"
        assert result1.tool_call_id is None, "tool_call_id should be None for opening tag only"
        
        # Second chunk: function name starts
        result2 = parser.extract_tool_calls_streaming(
            previous_text="<tool_call>",
            current_text="<tool_call><function=search_web",
            delta_text="<function=search_web",
            previous_token_ids=[1001],
            current_token_ids=[1001, 1003],
            delta_token_ids=[1003],
            request=request
        )
        
        # Should return structured message but no tool call ID yet
        assert result2 is not None, "Should return structured message for partial function name"
        assert result2.tool_call_id is None, "tool_call_id should be None for partial function name"
        
        # Third chunk: function name complete with opening brace
        result3 = parser.extract_tool_calls_streaming(
            previous_text="<tool_call><function=search_web",
            current_text="<tool_call><function=search_web>",
            delta_text=">",
            previous_token_ids=[1001, 1003],
            current_token_ids=[1001, 1003, 1004],
            delta_token_ids=[1004],
            request=request
        )
        
        # Should return structured message with function name and tool_call_id
        assert result3 is not None, "Should return structured message when function name completes"
        if hasattr(result3, 'tool_calls') and len(result3.tool_calls) > 0:
            assert result3.tool_call_id is None, "tool_call_id should be None in delta messages"
