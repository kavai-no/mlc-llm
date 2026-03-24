#!/usr/bin/env python3
"""
Quick test for non-streaming tool call handling
"""
import sys
sys.path.insert(0, '/home/kristoffer/mlc-llm/python')

from mlc_llm.protocol.openai_api_protocol import ChatCompletionMessage

# Test that our message format is valid  
delta_message = ChatCompletionMessage(content="", role="assistant", tool_calls=[{"index": 0, "id": "test_id", "type": "function", "function": {"name": "get_weather", "arguments": ""}}])

print(f"Content: {delta_message.content}")
print(f"Role: {delta_message.role}")
print(f"Tool calls: {len(delta_message.tool_calls)}")
assert isinstance(delta_message.content, str), "Content must be a string"
print("✅ Non-streaming format validation passed!")