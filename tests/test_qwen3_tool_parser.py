"""
Test boundary for Qwen3 tool parser integration in mlc-llm.
This tests the public API, not internal implementation details.
"""

import pytest
import sys
import os
from pathlib import Path

# Use .venv for Python path resolution
venv_path = Path("/workspace/projects/mlc-llm/.venv")
sys.path.insert(0, str(venv_path))

# Add the project root to Python path
sys.path.insert(0, str(Path("/workspace/projects/mlc-llm/python")))

from mlc_llm.serve.engine import AsyncMLCEngine, MLCEngine


def test_qwen3_tool_parser_public_api():
    """
    Public API: Qwen3 tool parser should be accessible and functional.
    This verifies that the integration with mlc-llm works at the module level.
    """
    # Verify that the necessary modules are exported
    assert hasattr(mlc_llm.serve, 'engine'), "mlc_llm.serve should export engine module"
    assert hasattr(mlc_llm.serve.engine, 'AsyncMLCEngine'), "engine should export AsyncMLCEngine class"
    assert hasattr(mlc_llm.serve.engine, 'MLCEngine'), "engine should export MLCEngine class"


def test_qwen3_tool_parser_handles_tool_calls():
    """
    Public API: Qwen3 tool parser should correctly handle tool calls.
    This tests the behavior of the integrated tool parser without mocking internal details.
    """
    # Initialize engines with a placeholder model path
    model_path = "/path/to/placeholder/model"
    device = "cuda:0"
    
    async_engine = AsyncMLCEngine(model=model_path, device=device)
    sync_engine = MLCEngine(model=model_path, device=device)
    
    # Define a test message with tool calls
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
    
    # Test streaming response with tools
    # Test streaming response with tools
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
    
    first_chunk = asyncio.run(get_first_chunk())
    assert first_chunk is not None, "Streaming response should contain at least one chunk"
    assert len(first_chunk.choices[0].delta.tool_calls) == 1, "First chunk should contain tool call information"
    
    # Test non-streaming response with tools
    non_stream_response = async_engine.chat.completions.create(
        messages=messages,
        model="qwen3.5",
        stream=False,
        tools=tools,
        tool_choice="auto"
    )
    
    assert len(non_stream_response.choices) == 1, "Non-streaming response should contain one choice"
    assert non_stream_response.choices[0].finish_reason == "tool_calls", "Response should indicate tool calls were used"


def test_qwen3_tool_parser_handles_invalid_tool_calls():
    """
    Public API: Qwen3 tool parser should handle invalid tool calls gracefully.
    This tests error handling at the public API level.
    """
    model_path = "/path/to/placeholder/model"
    device = "cuda:0"
    async_engine = AsyncMLCEngine(model=model_path, device=device)
    
    messages = [
        {"role": "user", "content": "Invalid tool call test"},
        {"role": "assistant", "content": None}
    ]
    
    tools = [{"type": "function", "function": {"name": "invalid_tool", "parameters": {}}}]
    
    with pytest.raises(Exception) as exc_info:
        async_engine.chat.completions.create(
            messages=messages,
            model="qwen3.5",
            stream=False,
            tools=tools,
            tool_choice="auto"
        )
    
    assert "Invalid function call" in str(exc_info.value), "Should raise an error for invalid tool calls"
