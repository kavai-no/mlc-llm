"""
Test Qwen3 template with tool list rendering and tool call results.
"""
import pytest
from mlc_llm.conversation_template.qwen3_5 import Qwen3_5_Template


def test_qwen3_tool_list_rendering():
    """Test that the Qwen3 template can render a list of tools in the system prompt."""
    tools = [
        {
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
        },
        {
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
    ]
    
    template = Qwen3_5_Template(system_message="You are a helpful assistant")
    # Set tools using the proper field
    template.tools = tools
    
    # Check that tools are stored
    assert len(template.tools) == 2
    assert template.tools[0]["name"] == "get_weather"
    assert template.tools[1]["name"] == "get_timezone"
    
    # Test the XML rendering
    xml = template._render_tools_xml()
    assert "<tools>" in xml
    assert "</tools>" in xml
    assert "<function><name>get_weather</name>" in xml
    assert "<function><name>get_timezone</name>" in xml
    assert "<parameter><name>location</name>" in xml
    assert "<required>[location]</required>" in xml


def test_qwen3_tool_call_with_results():
    """Test that the Qwen3 template can render tool calls and their results."""
    template = Qwen3_5_Template(system_message="You are a helpful assistant")
    
    # Add messages including tool calls and results
    template.messages = [
        ("user", "What's the weather in San Francisco?"),
        ("assistant", [
            {"type": "text", "text": "Let me check..."},
            {"type": "tool_call", "name": "get_weather", "parameters": {"location": "San Francisco", "unit": "celsius"}}
        ]),
        ("tool", [
            {"type": "tool_result", "content": "Temperature: 15°C, Conditions: Sunny"}
        ])
    ]
    
    # Generate the prompt
    prompts = template.as_prompt()
    
    # Check that the prompt contains tool call XML
    full_prompt = "".join(prompts) if isinstance(prompts, list) else prompts
    assert "<tool_call>" in full_prompt or "get_weather" in full_prompt  # Tool call should be present
    
    # The prompt should contain the tool result
    assert "Temperature: 15°C" in full_prompt or "Sunny" in full_prompt


def test_qwen3_empty_tool_list():
    """Test that empty tool list doesn't break rendering."""
    template = Qwen3_5_Template(system_message="You are a helpful assistant")
    template.tools = []
    
    xml = template._render_tools_xml()
    assert xml == ""


def test_qwen3_no_tool_list():
    """Test that no tool list parameter works."""
    template = Qwen3_5_Template(system_message="You are a helpful assistant")
    
    xml = template._render_tools_xml()
    assert xml == ""


if __name__ == "__main__":
    test_qwen3_tool_list_rendering()
    test_qwen3_tool_call_with_results()
    test_qwen3_empty_tool_list()
    test_qwen3_no_tool_list()
    print("All tests passed!")