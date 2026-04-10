"""
End-to-end test for Qwen3 tool parser hydration pattern.
This tests the complete flow from conversation template to engine resolution.
"""

import pytest
import sys
from pathlib import Path

# Use .venv for Python path resolution
venv_path = Path("/workspace/projects/mlc-llm/.venv")
sys.path.insert(0, str(venv_path))

# Add the project root to Python path
sys.path.insert(0, str(Path("/workspace/projects/mlc-llm/python")))

from mlc_llm.serve.conversation_template.qwen3_5 import create_qwen3_5_conversation
from mlc_llm.protocol.conversation_protocol import Conversation
from mlc_llm.serve.tool_parser import Qwen3CoderToolCallParser, get_parser_instance


def test_end_to_end_hydration_pattern():
    """
    Test the complete hydration pattern from template to parser instance.
    """
    # Step 1: Create conversation template (returns Conversation object)
    conv_template = create_qwen3_5_conversation()
    
    # Verify it's a proper Conversation object
    assert isinstance(conv_template, Conversation)
    
    # Step 2: Verify tool_parser is stored as string for hydration
    assert conv_template.tool_parser == "qwen3_coder"
    assert isinstance(conv_template.tool_parser, str)
    
    # Step 3: Test JSON serialization (what would be stored in config)
    template_dict = conv_template.model_dump()
    assert "tool_parser" in template_dict
    assert template_dict["tool_parser"] == "qwen3_coder"
    
    # Step 4: Test deserialization and hydration
    hydrated_template = Conversation.model_validate(template_dict)
    assert hasattr(hydrated_template, 'tool_parser_instance')
    assert hydrated_template.tool_parser_instance is not None
    assert isinstance(hydrated_template.tool_parser_instance, Qwen3CoderToolCallParser)
    
    # Step 5: Test parser functionality
    parser = hydrated_template.tool_parser_instance
    xml_content = "<tool_call><function=get_weather><parameter=location>New York</parameter></function></tool_call>"
    content, tool_calls = parser.parse(xml_content)
    
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.arguments["location"] == "New York"


def test_hydration_registry_pattern():
    """
    Test that the parser registry correctly handles different parser types.
    """
    # Test getting a parser by name
    parser = get_parser_instance("qwen3_coder")
    assert parser is not None
    assert isinstance(parser, Qwen3CoderToolCallParser)
    
    # Test with invalid parser name
    invalid_parser = get_parser_instance("nonexistent_parser")
    assert invalid_parser is None


def test_conversation_cloning_preserves_hydration():
    """
    Test that conversation cloning preserves the hydration pattern.
    """
    # Create original conversation template
    conv_template = create_qwen3_5_conversation()
    
    # Clone the conversation (simulating what happens in engine)
    cloned_conv = Conversation.model_validate(conv_template.model_dump())
    
    # Verify both have the same tool_parser string
    assert conv_template.tool_parser == cloned_conv.tool_parser
    
    # Verify both can be hydrated independently
    assert conv_template.tool_parser_instance is not None
    assert cloned_conv.tool_parser_instance is not None
    
    # Verify they are different instances (request isolation)
    assert conv_template.tool_parser_instance is not cloned_conv.tool_parser_instance


def test_xml_parsing_with_hydrated_parser():
    """
    Test XML parsing with a fully hydrated parser from conversation template.
    """
    # Create and hydrate the conversation template
    conv_template = create_qwen3_5_conversation()
    
    # Get the hydrated parser
    parser = conv_template.tool_parser_instance
    assert parser is not None
    
    # Test with various XML formats
    test_cases = [
        {
            "xml": "<tool_call><function=get_weather><parameter=location>New York</parameter></function></tool_call>",
            "expected_name": "get_weather",
            "expected_param": ("location", "New York")
        },
        {
            "xml": "<tool_call><function=get_time><parameter=timezone>America/New_York</parameter></function></tool_call>",
            "expected_name": "get_time",
            "expected_param": ("timezone", "America/New_York")
        },
    ]
    
    for test_case in test_cases:
        content, tool_calls = parser.parse(test_case["xml"])
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == test_case["expected_name"]
        param_name, param_value = test_case["expected_param"]
        assert tool_calls[0].function.arguments[param_name] == param_value


def test_streaming_parsing_with_hydrated_parser():
    """
    Test streaming XML parsing with a fully hydrated parser.
    """
    # Create and hydrate the conversation template
    conv_template = create_qwen3_5_conversation()
    parser = conv_template.tool_parser_instance
    
    # Test partial tool call detection
    partial_chunk = "<tool_call><function=get_weather><parameter=location>New York</parameter></function"
    result = parser.parse_streaming(partial_chunk)
    assert result["type"] == "partial_tool_call"
    assert "<tool_call>" in result["data"]
    
    # Test complete tool call detection
    complete_chunk = "<tool_call><function=get_weather><parameter=location>New York</parameter></function></tool_call>"
    result = parser.parse_streaming(complete_chunk)
    assert result["type"] == "complete_tool_call"
    assert len(result["data"]) == 1
    # The streaming parser should return properly parsed ChatToolCall objects
    tool_call = result["data"][0]
    assert hasattr(tool_call, 'function')
    assert hasattr(tool_call.function, 'name')
    assert tool_call.function.name == "get_weather"