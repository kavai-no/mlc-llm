"""Test the Qwen3 tool parser implementation."""

import sys
sys.path.insert(0, '/workspace/projects/mlc-llm/python')

from mlc_llm.serve.qwen3_tool_parser import Qwen3CoderToolCallParser, get_parser_instance


def test_parse_simple_tool_call():
    """Test parsing a simple tool call."""
    parser = Qwen3CoderToolCallParser()
    
    text = """I'll check the weather for you.
<tool_call>
<function=get_weather>
<parameter=city>New York</parameter>
</function>
</tool_call>"""
    
    content, tool_calls = parser.parse(text)
    
    print("Test: Simple tool call")
    print(f"  Content: {repr(content)}")
    print(f"  Tool calls: {len(tool_calls)}")
    
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.arguments == {"city": "New York"}
    assert "I'll check the weather for you." in content
    assert "<tool_call>" not in content
    print("  ✓ PASSED\n")


def test_parse_multiple_parameters():
    """Test parsing a tool call with multiple parameters."""
    parser = Qwen3CoderToolCallParser()
    
    text = """<tool_call>
<function=get_weather>
<parameter=city>San Francisco</parameter>
<parameter=unit>celsius</parameter>
</function>
</tool_call>"""
    
    content, tool_calls = parser.parse(text)
    
    print("Test: Multiple parameters")
    print(f"  Content: {repr(content)}")
    print(f"  Tool calls: {len(tool_calls)}")
    
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.arguments == {
        "city": "San Francisco",
        "unit": "celsius"
    }
    print("  ✓ PASSED\n")


def test_parse_streaming_text_only():
    """Test streaming with text only (no tool call)."""
    parser = Qwen3CoderToolCallParser()
    
    tokens = ["Hello, ", "how ", "can ", "I ", "help ", "you?"]
    
    print("Test: Streaming text only")
    for i, token in enumerate(tokens):
        content_delta, tool_calls = parser.parse_streaming(token)
        print(f"  Token {i+1}: {repr(token)}")
        print(f"    Content delta: {repr(content_delta)}")
        print(f"    Tool calls: {len(tool_calls)}")
        
        if i < len(tokens) - 1:
            # Should buffer and send content
            assert content_delta is not None
            assert len(tool_calls) == 0
        else:
            # Last token should flush
            assert content_delta is not None
            assert len(tool_calls) == 0
    
    print("  ✓ PASSED\n")


def test_parse_streaming_with_tool_call():
    """Test streaming with a tool call."""
    parser = Qwen3CoderToolCallParser()
    
    tokens = [
        "Let me check that for you.",
        "\n<tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call>"
    ]
    
    print("Test: Streaming with tool call")
    for i, token in enumerate(tokens):
        content_delta, tool_calls = parser.parse_streaming(token)
        print(f"  Token {i+1}: {repr(token[:50])}")
        print(f"    Content delta: {repr(content_delta)}")
        print(f"    Tool calls: {len(tool_calls)}")
        
        if i == 0:
            # First token: text before tool call
            assert content_delta is not None
            assert "Let me check that for you." in content_delta
            assert len(tool_calls) == 0
        else:
            # Second token: complete tool call
            assert content_delta == ""
            assert len(tool_calls) == 1
            assert tool_calls[0].function.name == "get_weather"
            assert tool_calls[0].function.arguments == {"city": "Paris"}
    
    print("  ✓ PASSED\n")


def test_parse_streaming_partial_tool_call():
    """Test streaming with partial tool call (buffering)."""
    parser = Qwen3CoderToolCallParser()
    
    tokens = [
        "Checking weather",
        "\n<tool_call>\n<function=get_weather>\n<parameter=city>",
        "London",
        "</parameter>\n</function>\n</tool_call>"
    ]
    
    print("Test: Streaming partial tool call")
    for i, token in enumerate(tokens):
        content_delta, tool_calls = parser.parse_streaming(token)
        print(f"  Token {i+1}: {repr(token)}")
        print(f"    Content delta: {repr(content_delta)}")
        print(f"    Tool calls: {len(tool_calls)}")
        
        if i == 0:
            # First token: text before tool call
            assert content_delta is not None
            assert "Checking weather" in content_delta
            assert len(tool_calls) == 0
        elif i == 1:
            # Second token: start of tool call, buffer it
            assert content_delta is None
            assert len(tool_calls) == 0
        elif i == 2:
            # Third token: still incomplete, buffer
            assert content_delta is None
            assert len(tool_calls) == 0
        else:
            # Fourth token: complete tool call
            assert content_delta == ""
            assert len(tool_calls) == 1
            assert tool_calls[0].function.name == "get_weather"
            assert tool_calls[0].function.arguments == {"city": "London"}
    
    print("  ✓ PASSED\n")


def test_render_tool_call():
    """Test rendering a tool call."""
    parser = Qwen3CoderToolCallParser()
    
    from mlc_llm.protocol.openai_api_protocol import ChatToolCall, ChatFunctionCall
    
    tool_call = ChatToolCall(
        id="call_abc123",
        type="function",
        function=ChatFunctionCall(
            name="get_weather",
            arguments={"city": "Tokyo", "unit": "fahrenheit"}
        )
    )
    
    rendered = parser.render_tool_call(tool_call)
    
    print("Test: Render tool call")
    print(f"  Rendered: {repr(rendered)}")
    
    assert "<tool_call>" in rendered
    assert "<function=get_weather>" in rendered
    assert "<parameter=city>" in rendered
    assert "Tokyo" in rendered
    assert "<parameter=unit>" in rendered
    assert "fahrenheit" in rendered
    assert "</function>" in rendered
    assert "</tool_call>" in rendered
    print("  ✓ PASSED\n")


def test_render_tool_result():
    """Test rendering a tool result."""
    parser = Qwen3CoderToolCallParser()
    
    rendered = parser.render_tool_result("call_abc123", "The weather is sunny, 25°C")
    
    print("Test: Render tool result")
    print(f"  Rendered: {repr(rendered)}")
    
    assert "<|startoftext|>" in rendered
    assert "The weather is sunny, 25°C" in rendered
    assert "<|endofreason|>" in rendered
    print("  ✓ PASSED\n")


def test_parser_registry():
    """Test that the parser is registered correctly."""
    parser = get_parser_instance("qwen3_coder")
    
    print("Test: Parser registry")
    print(f"  Parser instance: {parser}")
    
    assert parser is not None
    assert isinstance(parser, Qwen3CoderToolCallParser)
    print("  ✓ PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Qwen3 Tool Parser")
    print("=" * 60 + "\n")
    
    test_parse_simple_tool_call()
    test_parse_multiple_parameters()
    test_parse_streaming_text_only()
    test_parse_streaming_with_tool_call()
    test_parse_streaming_partial_tool_call()
    test_render_tool_call()
    test_render_tool_result()
    test_parser_registry()
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
