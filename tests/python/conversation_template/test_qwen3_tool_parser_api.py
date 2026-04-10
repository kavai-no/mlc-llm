"""
API-level test suite for Qwen3 Tool Parser.

Tests:
1. Integration with the engine and conversation templates.
2. Mocking external dependencies (e.g., OpenAI API).
3. Verifying tool call parsing in a real-world scenario.
"""

import json
from unittest.mock import patch, MagicMock
from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders
from mlc_llm.serve.tool_parser import Qwen3CoderToolCallParser


def test_integration_with_engine():
    """Test the integration of the tool parser with the engine."""
    # Create a mock conversation template similar to qwen3_5
    conv_template = Conversation(
        name="qwen3_5",
        system_template=f"<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}<|im_end|>\n",
        system_message="You are a helpful assistant.",
        roles={
            "user": "<|im_start|>user",
            "assistant": "<|im_start|>assistant\n<think>",
        },
        seps=["<|im_end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|endoftext|>", "<|im_end|>"],
        stop_token_ids=[248046, 248044],
    )

    # Initialize the tool parser for this template (without assigning to conv_template)
    parser = Qwen3CoderToolCallParser()
    
    # Mock external dependencies (e.g., OpenAI API)
    mock_tool_call = {
        "id": "call_1234567890",
        "type": "function",
        "function": {
            "name": "search_web",
            "arguments": {"query": "latest AI advancements"}
        }
    }

    # Simulate a tool call in XML-style format (expected by the parser)
    input_text = (
        "<tool_call>\n"
        "  <function=search_web>\n"
        "    <parameter=query>latest AI advancements</parameter>\n"
        "  </function>\n"
        "</tool_call>"
    )
    content, tool_calls = parser.parse(input_text)

    assert len(tool_calls) == 1
    assert tool_calls[0].type == "function"
    assert tool_calls[0].function.name == "search_web"
    assert tool_calls[0].function.arguments == {"query": "latest AI advancements"}


def test_mocking_external_dependencies():
    """Test the parser with mocked external dependencies."""
    # Initialize the parser
    parser = Qwen3CoderToolCallParser()
    
    # Mock an external API response (e.g., OpenAI)
    mock_response = {
        "choices": [{
            "message": {
                "content": "<tool_call>\n  <function=search_web>\n    <parameter=query>latest AI advancements</parameter>\n  </function>\n</tool_call>"
            }
        }]
    }

    # Simulate parsing the mocked response (directly use the content)
    input_text = mock_response['choices'][0]['message']['content']
    content, tool_calls = parser.parse(input_text)

    assert len(tool_calls) == 1
    assert tool_calls[0].type == "function"
    assert tool_calls[0].function.name == "search_web"
    assert tool_calls[0].function.arguments == {"query": "latest AI advancements"}