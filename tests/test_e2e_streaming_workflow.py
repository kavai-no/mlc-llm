"""
End-to-end test for the complete streaming workflow.

This test verifies that:
1. The parser can handle fragmented XML tags across chunks
2. Complete tool calls are properly detected and extracted
3. Partial content is buffered correctly
4. Buffer is cleared after complete tool calls
5. Multiple sequential tool calls work correctly
"""
import pytest
from mlc_llm.serve.tool_parser import Qwen3CoderToolCallParser


def test_end_to_end_streaming_workflow():
    """Test a complete streaming workflow with multiple chunks."""
    parser = Qwen3CoderToolCallParser()
    
    # Simulate streaming tokens arriving in chunks
    chunks = [
        "<tool_call>",  # Opening tag
        "<function=get_weather>",  # Function opening
        "<parameter=location>New York</parameter>",  # First parameter
        "<parameter=units>celsius</parameter>",  # Second parameter
        "</function></tool_call>"  # Closing tags
    ]
    
    all_tool_calls = []
    for chunk in chunks:
        result = parser.parse_streaming(chunk)
        if result and result['type'] == 'complete_tool_call':
            all_tool_calls.extend(result['data'])
    
    # Should have exactly one complete tool call
    assert len(all_tool_calls) == 1
    tc = all_tool_calls[0]
    assert tc.function.name == 'get_weather'
    assert tc.function.arguments['location'] == 'New York'
    assert tc.function.arguments['units'] == 'celsius'


def test_multiple_complete_tool_calls_in_sequence():
    """Test multiple complete tool calls processed sequentially."""
    parser = Qwen3CoderToolCallParser()
    
    # First complete tool call
    chunk1 = "<tool_call><function=get_weather><parameter=location>New York</parameter></function></tool_call>"
    result1 = parser.parse_streaming(chunk1)
    assert result1['type'] == 'complete_tool_call'
    assert len(result1['data']) == 1
    assert result1['data'][0].function.name == 'get_weather'
    
    # Second complete tool call (should not include first one)
    chunk2 = "<tool_call><function=get_timezone><parameter=city>London</parameter></function></tool_call>"
    result2 = parser.parse_streaming(chunk2)
    assert result2['type'] == 'complete_tool_call'
    assert len(result2['data']) == 1
    assert result2['data'][0].function.name == 'get_timezone'
    
    # Third complete tool call
    chunk3 = "<tool_call><function=get_temperature><parameter=location>Paris</parameter></function></tool_call>"
    result3 = parser.parse_streaming(chunk3)
    assert result3['type'] == 'complete_tool_call'
    assert len(result3['data']) == 1
    assert result3['data'][0].function.name == 'get_temperature'


def test_partial_chunk_followed_by_complete():
    """Test that partial chunks are buffered and completed when more data arrives."""
    parser = Qwen3CoderToolCallParser()
    
    # First chunk: incomplete (missing closing tags)
    partial1 = "<tool_call><function=get_weather><parameter=location>New York</parameter>"
    result1 = parser.parse_streaming(partial1)
    assert result1['type'] == 'partial_tool_call'
    
    # Second chunk: completes the tool call
    complete2 = "</function></tool_call>"
    result2 = parser.parse_streaming(complete2)
    assert result2['type'] == 'complete_tool_call'
    assert len(result2['data']) == 1
    assert result2['data'][0].function.name == 'get_weather'
    assert result2['data'][0].function.arguments['location'] == 'New York'


def test_mixed_content_and_tool_calls():
    """Test that regular text content doesn't interfere with tool call parsing."""
    parser = Qwen3CoderToolCallParser()
    
    # Text before tool call
    chunk1 = "Here is some text before the tool call: "
    result1 = parser.parse_streaming(chunk1)
    assert result1 is None  # No tool calls detected
    
    # Complete tool call
    chunk2 = "<tool_call><function=get_weather><parameter=location>New York</parameter></function></tool_call>"
    result2 = parser.parse_streaming(chunk2)
    assert result2['type'] == 'complete_tool_call'
    assert len(result2['data']) == 1
    
    # Text after tool call (should be in buffer but not parsed as tool call)
    chunk3 = "And here is text after the tool call."
    result3 = parser.parse_streaming(chunk3)
    assert result3 is None  # No more complete tool calls


def test_empty_and_whitespace_chunks():
    """Test that empty and whitespace-only chunks are handled correctly."""
    parser = Qwen3CoderToolCallParser()
    
    # Empty chunk
    result1 = parser.parse_streaming("")
    assert result1 is None
    
    # Whitespace only
    result2 = parser.parse_streaming("   \n\t  ")
    assert result2 is None
    
    # Complete tool call after whitespace
    chunk3 = "<tool_call><function=get_weather><parameter=location>New York</parameter></function></tool_call>"
    result3 = parser.parse_streaming(chunk3)
    assert result3['type'] == 'complete_tool_call'
    assert len(result3['data']) == 1