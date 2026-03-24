#!/usr/bin/env python3
"""
Test script to verify streaming tool call fixes.
This simulates the engine behavior with accumulated text tracking.
"""

import sys
sys.path.insert(0, '/home/kristoffer/mlc-llm/python')

# Mock TVM module to avoid import errors
class MockTVM:
    pass

class MockObject:
    def __init__(self):
        pass

mock_tvm = MockTVM()
mock_tvm.runtime = type('Module', (), {'Object': MockObject})()
sys.modules['tvm'] = mock_tvm

from mlc_llm.serve.tool_parsers.qwen3coder import Qwen3CoderToolParser
from mlc_llm.protocol.openai_api_protocol import ChatCompletionRequest, ChatCompletionMessage, ChatFunctionCall
import json

class MockTokenizer:
    def __init__(self):
        self.tokenizer = None
        
    def encode(self, text):
        return list(range(len(text)))

def test_streaming_response_format():
    """Test that tool parser returns proper OpenAI API format"""
    print("Testing streaming response format...")
    
    # Create mock tokenizer
    tokenizer = MockTokenizer()
    
    # Create tool parser instance
    tool_parser = Qwen3CoderToolParser(tokenizer)
    
    # Create a minimal request
    request = ChatCompletionRequest(
        messages=[ChatCompletionMessage(content="test", role="user")],
        model="test-model"
    )
    
    # Simulate streaming chunks that would come from the model
    test_cases = [
        {
            "desc": "Empty delta (no text)",
            "previous_text": "",
            "current_text": "<tool_call>",
            "delta_text": "<tool_call>",
            "delta_token_ids": [1, 2],
        },
        {
            "desc": "Function name detection",
            "previous_text": "<tool_call>",
            "current_text": "<tool_call><function=get_weather>",
            "delta_text": "<function=get_weather>",
            "delta_token_ids": [3, 4, 5],
        },
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i+1}: {test_case['desc']}")
        
        result = tool_parser.extract_tool_calls_streaming(
            previous_text=test_case["previous_text"],
            current_text=test_case["current_text"],
            delta_text=test_case["delta_text"],
            previous_token_ids=[],
            current_token_ids=test_case["delta_token_ids"],
            delta_token_ids=test_case["delta_token_ids"],
            request=request
        )
        
        # Verify result is not None
        assert result is not None, f"Test case {i+1}: Returned None instead of ChatCompletionMessage"
        
        # Verify it's a ChatCompletionMessage
        assert isinstance(result, ChatCompletionMessage), f"Test case {i+1}: Not a ChatCompletionMessage object"
        
        # Verify required fields exist
        assert hasattr(result, 'content'), f"Test case {i+1}: Missing content field"
        assert hasattr(result, 'role'), f"Test case {i+1}: Missing role field"
        assert hasattr(result, 'tool_calls'), f"Test case {i+1}: Missing tool_calls field"
        
        print(f"  ✓ Returns ChatCompletionMessage with proper structure")
        print(f"  ✓ content: {repr(result.content)}")
        print(f"  ✓ role: {repr(result.role)}")
        print(f"  ✓ tool_calls: {result.tool_calls}")
    
    print("\n✅ All streaming format tests passed!")

if __name__ == "__main__":
    test_streaming_response_format()
