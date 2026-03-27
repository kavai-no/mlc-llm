"""
Test demonstrating the use of XmlToolCallBuilder for creating test data.
This shows how to use the fluent interface to generate well-formed XML tool calls.
"""

import pytest
from mlc_llm.tokenizers import Tokenizer
from mlc_llm.serve.tool_parsers.qwen3coder import Qwen3CoderToolParser
from mlc_llm.protocol.openai_api_protocol import ChatCompletionRequest
from test_data_factory import XmlToolCallBuilder, create_search_web_tool

# Test category: unittest (no GPU, no model required)
pytestmark = [pytest.mark.unittest]


@pytest.fixture
def parser():
    """Create a Qwen3CoderToolParser instance with real tokenizer."""
    return Qwen3CoderToolParser(Tokenizer("tests/python/serve/qwen3-coder-30b-a3b-instruct"))


class TestFluentBuilderExamples:
    """Demonstrate the fluent builder usage."""
    
    def test_fluent_builder_single_param(self, parser):
        """Test using fluent builder to create a tool call with one parameter."""
        tools = [create_search_web_tool()]
        request = ChatCompletionRequest(model="test", messages=[], tools=tools)
        
        # Create XML using the fluent builder
        xml_input = (XmlToolCallBuilder()
                     .with_function("search_web")
                     .add_parameter("query", "fluent builder test")
                     .build())
        
        result = parser.extract_tool_calls(xml_input, request)
        
        assert result.tools_called
        assert len(result.tool_calls) == 1
        
        # Verify the query parameter was extracted correctly
        import json
        args = json.loads(result.tool_calls[0].function.arguments)
        assert "query" in args
        assert args["query"] == "fluent builder test"
    
    def test_fluent_builder_multiple_params(self, parser):
        """Test using fluent builder to create a tool call with multiple parameters."""
        tools = [create_search_web_tool()]
        request = ChatCompletionRequest(model="test", messages=[], tools=tools)
        
        # Create XML using the fluent builder (fluent interface)
        xml_input = (XmlToolCallBuilder()
                     .with_function("search_web")
                     .add_parameter("query", "multi-param test")
                     .add_parameter("count", "5")
                     .build())
        
        result = parser.extract_tool_calls(xml_input, request)
        
        assert result.tools_called
        assert len(result.tool_calls) == 1
        
        # Verify both parameters were extracted correctly
        import json
        args = json.loads(result.tool_calls[0].function.arguments)
        assert "query" in args
        assert "count" in args
        assert args["query"] == "multi-param test"
        assert args["count"] == "5"
    
    def test_fluent_builder_chained_calls(self, parser):
        """Demonstrate method chaining with the builder."""
        tools = [create_search_web_tool()]
        request = ChatCompletionRequest(model="test", messages=[], tools=tools)
        
        # Method chaining - all in one line
        xml_input = XmlToolCallBuilder() \
            .with_function("search_web") \
            .add_parameter("query", "chained test") \
            .add_parameter("count", "3") \
            .build()
        
        result = parser.extract_tool_calls(xml_input, request)
        
        assert result.tools_called
        import json
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["query"] == "chained test"
    
    def test_fluent_builder_empty_params(self):
        """Test builder creates proper empty XML when no parameters added."""
        xml_input = (XmlToolCallBuilder()
                     .with_function("get_current_timestamp")
                     .build())
        
        # Should create XML with no parameter tags
        assert "<tool_call>" in xml_input
        assert "<function=get_current_timestamp>" in xml_input
        assert "</function>" in xml_input
        assert "</tool_call>" in xml_input
    
    def test_builder_validation(self):
        """Test that builder validates required fields."""
        from test_data_factory import XmlToolCallBuilder
        
        # Should raise error if function name not set
        try:
            XmlToolCallBuilder().add_parameter("query", "test").build()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Function name must be set" in str(e)
