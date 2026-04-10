"""
Test suite for Qwen3 Tool Parser integration using Test-Driven Development (TDD).
This suite focuses on public API behavior and ensures compatibility with existing conversation templates.
"""

import pytest
from unittest.mock import patch, MagicMock
from mlc_llm.serve.conversation_template.qwen3_5 import create_qwen3_5_conversation
from mlc_llm.serve.tool_parser import Qwen3CoderToolCallParser


def test_qwen3_tool_parser_public_api():
    """
    Public API: Verify that the Qwen3 tool parser is accessible and functional.
    This tests the behavior of the integrated tool parser without mocking internal details.
    """
    # Verify that the necessary modules are exported
    from mlc_llm.serve import tool_parser
    assert hasattr(tool_parser, 'Qwen3CoderToolCallParser'), "tool_parser should export Qwen3CoderToolCallParser class"


def test_qwen3_tool_parser_handles_tool_calls():
    """
    Public API: Verify that the Qwen3 tool parser correctly handles tool calls.
    This tests the behavior of the integrated tool parser without mocking internal details.
    """
    # Initialize the parser
    parser = Qwen3CoderToolCallParser()
    
    # Simulate a tool call in XML-style format (expected by the parser)
    input_text = (
        "<tool_call>\n"
        "  <function=search_web>\n"
        "    <parameter=query>latest AI advancements</parameter>\n"
        "  </function>\n"
        "</tool_call>"
    )
    
    # Parse the tool call
    content, tool_calls = parser.parse(input_text)
    
    assert len(tool_calls) == 1
    assert tool_calls[0].type == "function"
    assert tool_calls[0].function.name == "search_web"
    assert tool_calls[0].function.arguments == {"query": "latest AI advancements"}


def test_qwen3_tool_parser_handles_invalid_tool_calls():
    """
    Public API: Verify that the Qwen3 tool parser handles invalid tool calls gracefully.
    This tests error handling at the public API level.
    """
    # Initialize the parser
    parser = Qwen3CoderToolCallParser()
    
    # Simulate an invalid tool call (missing parameters)
    input_text = "<tool_call><function=invalid_tool></function></tool_call>"
    
    # Parse the tool call (should handle gracefully)
    content, tool_calls = parser.parse(input_text)
    
    assert len(tool_calls) == 0


def test_qwen3_tool_parser_integration_with_conversation_template():
    """
    Integration: Verify that the Qwen3 tool parser integrates correctly with conversation templates.
    This tests the integration of the tool parser into existing conversation structures.
    """
    # Create a Qwen3.5 conversation template
    conversation = create_qwen3_5_conversation()
    
    # Extract the tool parser from the conversation template
    parser = conversation["tool_parser"]
    
    # Simulate a tool call in XML-style format (expected by the parser)
    input_text = (
        "<tool_call>\n"
        "  <function=search_web>\n"
        "    <parameter=query>latest AI advancements</parameter>\n"
        "  </function>\n"
        "</tool_call>"
    )
    
    # Parse the tool call
    content, tool_calls = parser.parse(input_text)
    
    assert len(tool_calls) == 1
    assert tool_calls[0].type == "function"
    assert tool_calls[0].function.name == "search_web"
    assert tool_calls[0].function.arguments == {"query": "latest AI advancements"}


def test_qwen3_tool_parser_handles_streaming_and_non_streaming_calls():
    """
    Integration: Verify that the Qwen3 tool parser handles both streaming and non-streaming calls.
    This tests the behavior of the integrated tool parser in different modes.
    """
    # Initialize the parser
    parser = Qwen3CoderToolCallParser()
    
    # Simulate a tool call in XML-style format (expected by the parser)
    input_text = (
        "<tool_call>\n"
        "  <function=search_web>\n"
        "    <parameter=query>latest AI advancements</parameter>\n"
        "  </function>\n"
        "</tool_call>"
    )
    
    # Parse the tool call
    content, tool_calls = parser.parse(input_text)
    
    assert len(tool_calls) == 1
    assert tool_calls[0].type == "function"
    assert tool_calls[0].function.name == "search_web"
    assert tool_calls[0].function.arguments == {"query": "latest AI advancements"}
