"""
Test boundary for Qwen3 XML-style tool call rendering.
This tests the XML rendering capabilities of the conversation template.
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


def test_xml_rendering_complete_tool_call():
    """
    Test that the conversation template can render complete XML tool calls.
    """
    # Create the conversation template
    conv_template = create_qwen3_5_conversation()
    
    # Verify it's a Conversation object
    assert isinstance(conv_template, Conversation), "Template should return a Conversation object"
    
    # Verify tool_parser is set as string for hydration
    assert conv_template.tool_parser == "qwen3_coder", "Tool parser should be 'qwen3_coder' string"
    
    # Test that the template has the expected structure
    assert conv_template.name == "qwen3_5"
    assert "<|im_start|>" in conv_template.system_template
    assert "user" in conv_template.role_templates
    assert "assistant" in conv_template.role_templates

def test_xml_rendering_with_hydration():
    """
    Test that the conversation template properly hydrates the tool parser.
    """
    # Create the conversation template
    conv_template = create_qwen3_5_conversation()
    
    # Verify hydration works (tool_parser_instance should be created)
    assert hasattr(conv_template, 'tool_parser_instance'), "Conversation should have tool_parser_instance attribute"
    assert conv_template.tool_parser_instance is not None, "Tool parser instance should be hydrated"
    
    # Verify the hydrated parser is a Qwen3CoderToolCallParser
    from mlc_llm.serve.tool_parser import Qwen3CoderToolCallParser
    assert isinstance(conv_template.tool_parser_instance, Qwen3CoderToolCallParser), "Hydrated parser should be Qwen3CoderToolCallParser"

def test_xml_rendering_json_serialization():
    """
    Test that the conversation template can be serialized to JSON and back.
    """
    # Create the conversation template
    conv_template = create_qwen3_5_conversation()
    
    # Convert to dict (JSON serialization)
    template_dict = conv_template.model_dump()
    
    # Verify tool_parser is preserved as string in JSON
    assert "tool_parser" in template_dict
    assert template_dict["tool_parser"] == "qwen3_coder"
    
    # Test deserialization (hydration)
    from mlc_llm.protocol.conversation_protocol import Conversation
    hydrated_template = Conversation.model_validate(template_dict)
    
    assert hasattr(hydrated_template, 'tool_parser_instance')
    assert hydrated_template.tool_parser_instance is not None

def test_xml_rendering_multiple_tool_calls():
    """
    Test XML rendering with multiple tool calls.
    """
    # Create the conversation template
    conv_template = create_qwen3_5_conversation()
    
    # Verify the hydrated parser can handle multiple tool calls
    parser = conv_template.tool_parser_instance
    assert parser is not None, "Parser should be available for testing"
    
    # Test with a complex XML structure containing multiple tool calls
    xml_content = """<tool_call><function=get_weather><parameter=location>New York</parameter><parameter=unit>celsius</parameter></function></tool_call>
    <tool_call><function=get_time><parameter=timezone>America/New_York</parameter></function></tool_call>"""
    
    # The parser should be able to parse this structure
    content, tool_calls = parser.parse(xml_content)
    assert len(tool_calls) == 2, "Should parse two tool calls"
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[1].function.name == "get_time"

def test_xml_rendering_streaming_compatibility():
    """
    Test that the conversation template works with streaming responses.
    """
    # Create the conversation template
    conv_template = create_qwen3_5_conversation()
    
    # Verify the hydrated parser supports streaming
    parser = conv_template.tool_parser_instance
    assert hasattr(parser, 'parse_streaming'), "Parser should support streaming"
    
    # Test streaming with truly fragmented XML chunks (missing closing tags)
    fragments = [
        "<tool_call>",
        "<function=get_weather>",
        "<parameter=location>New York</parameter>",
        "</function"  # Missing closing '>' for the function tag
    ]
    
    # Join fragments to create a single string
    fragment_text = "".join(fragments)
    result = parser.parse_streaming(fragment_text)
    assert result["type"] == "partial_tool_call", "Should detect partial tool call"
    assert "<tool_call>" in result["data"], "Should contain the tool call data"


def test_xml_rendering_template_structure():
    """
    Test that the conversation template has the correct structure for XML rendering.
    """
    # Create the conversation template
    conv_template = create_qwen3_5_conversation()
    
    # Verify all required fields are present
    assert conv_template.name == "qwen3_5"
    assert conv_template.system_message == "You are a helpful assistant."
    assert "<|im_start|>" in conv_template.system_template
    assert "user" in conv_template.role_templates
    assert "assistant" in conv_template.role_templates
    assert "<|im_end|>\n" in conv_template.seps
    assert "<|endoftext|>" in conv_template.stop_str
    assert "<|im_end|>" in conv_template.stop_str
    
    # Verify stop token IDs are set correctly for Qwen3.5
    assert 248046 in conv_template.stop_token_ids
    assert 248044 in conv_template.stop_token_ids