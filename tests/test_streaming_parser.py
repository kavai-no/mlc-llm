"""
Test streaming parser behavior for Qwen3CoderToolCallParser.

This tests the incremental parsing of XML fragments as they arrive
from a streaming model output.
"""
import pytest
from mlc_llm.serve.tool_parser import Qwen3CoderToolCallParser


def test_streaming_complete_tool_call():
    """Test that a complete tool call is parsed correctly in one chunk."""
    parser = Qwen3CoderToolCallParser()
    
    chunk = '<tool_call><function=weather><parameter=location>London</parameter></function></tool_call>'
    result = parser.parse_streaming(chunk)
    
    assert result is not None
    assert result["type"] == "complete_tool_call"
    assert len(result["data"]) > 0


def test_streaming_partial_function_tag():
    """Test that a partial function tag returns partial data."""
    parser = Qwen3CoderToolCallParser()
    
    chunk = '<tool_call><function=weather'
    result = parser.parse_streaming(chunk)
    
    assert result is not None
    assert result["type"] == "partial_tool_call"
    assert "<function=weather" in result["data"]


def test_streaming_partial_parameter_tag():
    """Test that a partial parameter tag returns partial data."""
    parser = Qwen3CoderToolCallParser()
    
    chunk = '<tool_call><function=weather><parameter=location>Lon'
    result = parser.parse_streaming(chunk)
    
    assert result is not None
    assert result["type"] == "partial_tool_call"
    assert "<parameter=location>Lon" in result["data"]


def test_streaming_non_tool_content():
    """Test that non-tool content returns None."""
    parser = Qwen3CoderToolCallParser()
    
    chunk = "Hello, how can I help you today?"
    result = parser.parse_streaming(chunk)
    
    assert result is None


def test_streaming_multiple_chunks_complete():
    """Test that multiple chunks can be combined to form a complete tool call."""
    parser = Qwen3CoderToolCallParser()
    
    # First chunk: partial function tag
    chunk1 = '<tool_call><function=weather'
    result1 = parser.parse_streaming(chunk1)
    assert result1 is not None
    assert result1["type"] == "partial_tool_call"
    
    # Second chunk: complete the function tag and add parameter
    chunk2 = '><parameter=location>London</parameter></function></tool_call>'
    result2 = parser.parse_streaming(chunk2)
    assert result2 is not None
    assert result2["type"] == "complete_tool_call"


def test_streaming_empty_chunk():
    """Test that an empty chunk returns None."""
    parser = Qwen3CoderToolCallParser()
    
    chunk = ""
    result = parser.parse_streaming(chunk)
    
    assert result is None
