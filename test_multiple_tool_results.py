#!/usr/bin/env python3
"""
Test that multiple tool results are each individually wrapped in XML.
"""
from mlc_llm.conversation_template import ConvTemplateRegistry

def test_multiple_tool_results():
    """Test that multiple tool results are each wrapped individually."""
    template = ConvTemplateRegistry.get_conv_template("qwen3_5")
    
    # Add messages with multiple tool calls and results
    template.messages = [
        ("user", "What's the weather in San Francisco and New York?"),
        ("assistant", [
            {"type": "text", "text": "Let me check both locations..."},
            {"type": "tool_call", "name": "get_weather", "parameters": {"location": "San Francisco", "unit": "celsius"}},
            {"type": "tool_call", "name": "get_weather", "parameters": {"location": "New York", "unit": "fahrenheit"}}
        ])
    ]
    
    # Generate the prompt
    prompts = template.as_prompt()
    full_prompt = "".join(prompts) if isinstance(prompts, list) else prompts
    
    print("Generated prompt:")
    print(full_prompt)
    print("\n" + "="*80 + "\n")
    
    # Check that both tool calls are present
    assert "San Francisco" in full_prompt, "First location should be in prompt"
    assert "New York" in full_prompt, "Second location should be in prompt"
    assert full_prompt.count("<tool_call>") == 2, "Should have exactly 2 tool calls"
    
    print("✅ Multiple tool calls rendered correctly")
    
    # Now add tool results - each should be wrapped individually
    template.messages.append(("assistant", [
        {"type": "text", "text": "Here are the results:"},
        {"type": "tool_result", "content": "Temperature: 15°C, Conditions: Sunny"},
        {"type": "tool_result", "content": "Temperature: 72°F, Conditions: Cloudy"}
    ]))
    
    prompts = template.as_prompt()
    full_prompt = "".join(prompts) if isinstance(prompts, list) else prompts
    
    print("Generated prompt with results:")
    print(full_prompt)
    print("\n" + "="*80 + "\n")
    
    # Check that both tool results are wrapped individually
    assert full_prompt.count("<tool_response>") == 2, "Should have exactly 2 tool responses"
    assert "15°C" in full_prompt, "First result should be in prompt"
    assert "72°F" in full_prompt, "Second result should be in prompt"
    
    print("✅ Multiple tool results each wrapped individually")
    print("\n🎉 All tests PASSED!")

if __name__ == "__main__":
    test_multiple_tool_results()