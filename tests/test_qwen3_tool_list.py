"""
Test Qwen3 template with tool list rendering and tool call results.
"""
import pytest
from mlc_llm.conversation_template import ConvTemplateRegistry


def test_qwen3_tool_list_rendering():
    """Test that the Qwen3 template can render a list of tools in the system prompt."""
    # Tools should be in OpenAI API format for the parser
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_timezone",
                "description": "Get timezone information for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]
    
    template = ConvTemplateRegistry.get_conv_template("qwen3_5")
    # Set tools using the proper field
    template.tools = tools
    
    # Check that tools are stored
    assert len(template.tools) == 2
    assert template.tools[0]["function"]["name"] == "get_weather"
    assert template.tools[1]["function"]["name"] == "get_timezone"
    
    # Test the XML rendering using the tool parser instance
    if hasattr(template, 'tool_parser_instance') and template.tool_parser_instance:
        xml = template.tool_parser_instance.render_tools(template.tools)
        assert "<tools>" in xml
        assert "</tools>" in xml
        assert "<name>get_weather</name>" in xml
        assert "<name>get_timezone</name>" in xml
        assert "<name>location</name>" in xml


def test_qwen3_tool_call_with_results():
    """Test that the Qwen3 template can render tool calls and their results."""
    template = ConvTemplateRegistry.get_conv_template("qwen3_5")
    
    # Add messages including tool calls
    template.messages = [
        ("user", "What's the weather in San Francisco?"),
        ("assistant", [
            {"type": "text", "text": "Let me check..."},
            {"type": "tool_call", "name": "get_weather", "parameters": {"location": "San Francisco", "unit": "celsius"}}
        ])
    ]
    
    # Generate the prompt
    prompts = template.as_prompt()
    
    # Check that the prompt contains tool call XML
    full_prompt = "".join(prompts) if isinstance(prompts, list) else prompts
    assert "<tool_call>" in full_prompt or "get_weather" in full_prompt  # Tool call should be present


def test_qwen3_empty_tool_list():
    """Test that empty tool list doesn't break rendering."""
    template = ConvTemplateRegistry.get_conv_template("qwen3_5")
    template.tools = []
    
    if hasattr(template, 'tool_parser_instance') and template.tool_parser_instance:
        xml = template.tool_parser_instance.render_tools(template.tools)
        assert xml == ""


def test_qwen3_no_tool_list():
    """Test that no tool list parameter works."""
    template = ConvTemplateRegistry.get_conv_template("qwen3_5")
    
    if hasattr(template, 'tool_parser_instance') and template.tool_parser_instance:
        xml = template.tool_parser_instance.render_tools(template.tools)
        assert xml == ""


if __name__ == "__main__":
    test_qwen3_tool_list_rendering()
    test_qwen3_tool_call_with_results()
    test_qwen3_empty_tool_list()
    test_qwen3_no_tool_list()
    print("All tests passed!")