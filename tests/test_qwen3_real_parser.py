import pytest
from mlc_llm.serve.tool_parser import Qwen3CoderToolCallParser
from mlc_llm.protocol.openai_api_protocol import ChatToolCall

def test_real_qwen3_parser_complete():
    """Test the actual Qwen3 parser with a complete XML block."""
    parser = Qwen3CoderToolCallParser()
    text = "<tool_call><function=test_func><parameter=p1>val1</parameter></function></tool_call>"
    content, tool_calls = parser.parse(text)
    
    assert content == ""
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "test_func"
    assert tool_calls[0].function.arguments["p1"] == "val1"

def test_real_qwen3_parser_fragmented():
    """Test the actual Qwen3 parser with a fragmented/unclosed XML block."""
    parser = Qwen3CoderToolCallParser()
    # Unclosed function tag
    text = "<tool_call><function=test_func><parameter=p1>val1</parameter>"
    content, tool_calls = parser.parse(text)
    
    # Based on the regex in tool_parser.py: r"<function=(.*?)</function>|<function=(.*)$"
    # It should still be able to extract the function name even if unclosed.
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "test_func"

def test_real_qwen3_parser_no_tools():
    """Test the parser with text that contains no tool calls."""
    parser = Qwen3CoderToolCallParser()
    text = "Hello, how can I help you today?"
    content, tool_calls = parser.parse(text)
    
    assert content == text
    assert len(tool_calls) == 0

def test_real_qwen3_parser_streaming():
    """Verify that the real parser handles streaming chunks."""
    parser = Qwen3CoderToolCallParser()
    
    # Test empty chunk
    assert parser.parse_streaming("") is None
    
    # Test non-tool chunk
    assert parser.parse_streaming("Hello world") is None
    
    # Test partial tool call start
    result = parser.parse_streaming("<tool_call>")
    assert result["type"] == "partial_tool_call"
    assert "<tool_call>" in result["data"]
