import pytest
from mlc_llm.conversation_template import ConvTemplateRegistry

def test_qwen3_tool_rendering():
    """Test that the Qwen3 template can render actual tool lists."""
    template = ConvTemplateRegistry.get_conv_template("qwen3_5")
    
    # Add a message with tools
    template.messages = [
        ("user", "What's the weather?"),
        ("assistant", [
            {"type": "text", "text": "Let me check..."},
            {"type": "tool_call", "name": "get_weather", "parameters": {"location": "San Francisco"}}
        ])
    ]
    
    # Try to generate the prompt
    try:
        prompts = template.as_prompt()
        print("Generated prompt:", prompts)
        assert len(prompts) > 0, "Should generate at least one prompt"
    except Exception as e:
        pytest.fail(f"Failed to render tools: {e}")

if __name__ == "__main__":
    test_qwen3_tool_rendering()
