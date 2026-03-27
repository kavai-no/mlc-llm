#!/usr/bin/env python3
"""Test that validation still works when tool definitions are provided."""

import json
import pytest
from mlc_llm.serve.tool_parsers.qwen3coder import Qwen3CoderToolParser
from mlc_llm.protocol.openai_api_protocol import ChatTool, ChatFunction, ChatCompletionRequest

# Test category: unittest (no GPU, no model required)
pytestmark = [pytest.mark.unittest]

def test_validation_with_tool_definitions():
    """Test that validation works when we have tool definitions."""
    
    # Create a mock tokenizer
    class MockTokenizer:
        @property
        def vocab(self):
            return {"<tool_call>": 1001, "</tool_call>": 1002}
        
        def encode(self, text):
            res = self.vocab.get(text, None)
            return [res] if res is not None else [0]
    
    parser = Qwen3CoderToolParser(MockTokenizer())
    
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
    
    print("Test 1: Function call with required parameter provided")
    xml_with_query = '<tool_call><function=search_web><parameter=query>test query</parameter></function></tool_call>'
    result1 = parser.extract_tool_calls(xml_with_query, request)
    print(f"  Tools called: {result1.tools_called}")
    print(f"  Number of tool calls: {len(result1.tool_calls)}")
    
    if len(result1.tool_calls) > 0:
        args = json.loads(result1.tool_calls[0].function.arguments)
        print(f"  Arguments: {args}")
        assert "query" in args, "Should have query parameter"
        print("  ✅ PASS: Function with required parameter accepted\n")
    else:
        print("  ❌ FAIL: Should accept function with required parameter\n")
        return False
    
    print("Test 2: Function call WITHOUT required parameter (should be rejected)")
    xml_without_query = '<tool_call><function=search_web></function></tool_call>'
    result2 = parser.extract_tool_calls(xml_without_query, request)
    print(f"  Tools called: {result2.tools_called}")
    print(f"  Number of tool calls: {len(result2.tool_calls)}")
    
    if len(result2.tool_calls) == 0:
        print("  ✅ PASS: Function with missing required parameter rejected\n")
    else:
        print("  ❌ FAIL: Should reject function without required parameter\n")
        return False
    
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
    
    print("Test 3: Function with no parameters when tool also has no parameters")
    xml_no_params_call = '<tool_call><function=get_timestamp></function></tool_call>'
    result3 = parser.extract_tool_calls(xml_no_params_call, request_no_params)
    print(f"  Tools called: {result3.tools_called}")
    print(f"  Number of tool calls: {len(result3.tool_calls)}")
    
    if len(result3.tool_calls) > 0:
        args = json.loads(result3.tool_calls[0].function.arguments)
        print(f"  Arguments: {args}")
        assert args == {}, "Should have empty arguments"
        print("  ✅ PASS: Function with no parameters accepted\n")
    else:
        print("  ❌ FAIL: Should accept function with no parameters when tool also has none\n")
        return False
    
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    test_validation_with_tool_definitions()