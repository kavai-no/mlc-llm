"""
Test suite for the streaming parser fix.

This ensures that complete tool calls are properly detected when they appear in a single chunk,
even with the regex capture group behavior.
"""
import pytest
from mlc_llm.serve.tool_parser import Qwen3CoderToolCallParser


def test_streaming_complete_tool_call_single_chunk():
    """Test that a complete tool call in a single chunk is detected correctly."""
    parser = Qwen3CoderToolCallParser()
    
    # Complete tool call in one chunk
    chunk = "<tool_call><function=get_weather><parameter=location>New York</parameter></function></tool_call>"
    result = parser.parse_streaming(chunk)
    
    assert result is not None
    assert result['type'] == 'complete_tool_call'
    assert len(result['data']) == 1
    assert result['data'][0].function.name == 'get_weather'
    assert result['data'][0].function.arguments['location'] == 'New York'


def test_streaming_complete_tool_call_multiple_parameters():
    """Test that a complete tool call with multiple parameters is detected correctly."""
    parser = Qwen3CoderToolCallParser()
    
    chunk = "<tool_call><function=get_weather><parameter=location>New York</parameter><parameter=units>celsius</parameter></function></tool_call>"
    result = parser.parse_streaming(chunk)
    
    assert result is not None
    assert result['type'] == 'complete_tool_call'
    assert len(result['data']) == 1
    assert result['data'][0].function.name == 'get_weather'
    assert result['data'][0].function.arguments['location'] == 'New York'
    assert result['data'][0].function.arguments['units'] == 'celsius'


def test_streaming_complete_tool_call_no_buffer_leak():
    """Test that the buffer is properly managed and doesn't leak between calls."""
    parser = Qwen3CoderToolCallParser()
    
    # First call with complete tool call
    chunk1 = "<tool_call><function=get_weather><parameter=location>New York</parameter></function></tool_call>"
    result1 = parser.parse_streaming(chunk1)
    assert result1['type'] == 'complete_tool_call'
    
    # Second call should start fresh (no buffer from previous call)
    chunk2 = "<tool_call><function=get_timezone><parameter=city>London</parameter></function></tool_call>"
    result2 = parser.parse_streaming(chunk2)
    assert result2['type'] == 'complete_tool_call'
    assert len(result2['data']) == 1
    assert result2['data'][0].function.name == 'get_timezone'
    assert result2['data'][0].function.arguments['city'] == 'London'


def test_streaming_partial_vs_complete_detection():
    """Test that partial and complete tool calls are distinguished correctly."""
    parser = Qwen3CoderToolCallParser()
    
    # Partial: missing closing tag
    partial_chunk = "<tool_call><function=get_weather><parameter=location>New York</parameter>"
    result1 = parser.parse_streaming(partial_chunk)
    assert result1['type'] == 'partial_tool_call'
    
    # Complete: has closing tags
    complete_chunk = "<tool_call><function=get_weather><parameter=location>New York</parameter></function></tool_call>"
    result2 = parser.parse_streaming(complete_chunk)
    assert result2['type'] == 'complete_tool_call'
    
    # Verify the complete one has proper data
    assert len(result2['data']) == 1
    assert result2['data'][0].function.name == 'get_weather'


def test_streaming_empty_buffer_initial_state():
    """Test that empty buffer returns None for non-tool content."""
    parser = Qwen3CoderToolCallParser()
    
    # Empty chunk should return None
    result = parser.parse_streaming("")
    assert result is None
    
    # Non-tool content should return None
    result = parser.parse_streaming("Hello world")
    assert result is None