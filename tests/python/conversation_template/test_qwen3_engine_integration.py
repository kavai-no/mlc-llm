"""
Test suite for Qwen3 Engine Integration.

Tests:
1. Integration between the Qwen3 engine and tool parser.
2. Mocking external dependencies (e.g., OpenAI API).
3. Verifying tool call processing in a real-world scenario.
"""

import json
from unittest.mock import patch, MagicMock
from mlc_llm.serve.conversation_template.qwen3_5 import create_qwen3_5_conversation
from mlc_llm.serve.tool_parser import Qwen3CoderToolCallParser


def test_engine_integration_with_tool_parser():
    """Test the integration of the Qwen3 engine with the tool parser."""
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