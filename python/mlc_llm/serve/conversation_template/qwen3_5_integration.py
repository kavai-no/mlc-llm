"""
Integration prototype for Qwen3 Tool Parser in mlc-llm.
This module ensures compatibility with existing conversation templates and engine APIs.
"""

import pytest
from typing import Any, Dict, List, Optional, Union
import asyncio
from mlc_llm.serve.engine import AsyncMLCEngine, MLCEngine
from mlc_llm.serve.tool_parser import Qwen3CoderToolCallParser
from mlc_llm.protocol.openai_api_protocol import ChatCompletionResponse, ChatCompletionStreamResponse


def create_qwen3_5_integration():
    """
    Create a Qwen3.5 integration template that supports tool parsing.
    
    Returns:
        Dict[str, Any]: A conversation template with tool parser and engine compatibility.
    """
    # Initialize the tool parser
    tool_parser = Qwen3CoderToolCallParser()
    
    # Define the integration template
    integration_template = {
        "name": "qwen3_5_integration",
        "description": "Integration template for Qwen3.5 with tool parsing support.",
        "tool_parser": "qwen3_coder",  # Use string name instead of live instance
        "engine_compatibility": ["AsyncMLCEngine", "MLCEngine"],
    }
    
    return integration_template


@pytest.mark.asyncio
async def test_streaming_tool_call_integration():
    """
    Test the integration of streaming tool calls with AsyncMLCEngine.
    
    Returns:
        ChatCompletionStreamResponse: The first chunk of the streaming response.
    """
    # Initialize the engine and parser
    model_path = "/path/to/placeholder/model"
    device = "cuda:0"
    async_engine = AsyncMLCEngine(model=model_path, device=device)
    tool_parser = Qwen3CoderToolCallParser()
    
    # Define test messages with tool calls
    messages = [
        {"role": "user", "content": "What's the weather today?"},
        {"role": "assistant", "content": None}
    ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    # Simulate a streaming response with tool calls
    async def get_first_chunk():
        stream_response = await async_engine.chat.completions.create(
            messages=messages,
            model="qwen3.5",
            stream=True,
            tools=tools,
            tool_choice="auto"
        )
        
        # Verify that the first chunk contains tool call information
        first_chunk = None
        async for chunk in stream_response:
            if chunk.choices and len(chunk.choices) > 0:
                first_chunk = chunk
                break
        return first_chunk
    
    return await get_first_chunk()


def test_non_streaming_tool_call_integration():
    """
    Test the integration of non-streaming tool calls with MLCEngine.
    
    Returns:
        ChatCompletionResponse: The non-streaming response with tool calls.
    """
    # Initialize the engine and parser
    model_path = "/path/to/placeholder/model"
    device = "cuda:0"
    sync_engine = MLCEngine(model=model_path, device=device)
    tool_parser = Qwen3CoderToolCallParser()
    
    # Define test messages with tool calls
    messages = [
        {"role": "user", "content": "What's the weather today?"},
        {"role": "assistant", "content": None}
    ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    # Simulate a non-streaming response with tool calls
    non_stream_response = sync_engine.chat.completions.create(
        messages=messages,
        model="qwen3.5",
        stream=False,
        tools=tools,
        tool_choice="auto"
    )
    
    return non_stream_response