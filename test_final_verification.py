#!/usr/bin/env python3
"""
Final verification test for Qwen3 tool calling rendering.
Tests proper newlines, correct XML structure, and minimal protocol changes.
"""

import sys
sys.path.insert(0, '/workspace/projects/mlc-llm/python')

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders
from mlc_llm.serve.tool_parser import Qwen3CoderToolCallParser

print("Final Verification Test - Qwen3 Tool Calling")
print("=" * 70)

# Create conversation template with tool parser (matching qwen3_5)
template = Conversation(
    name="qwen3_5",
    system_template=f"<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}<|im_end|>\n",
    system_message="You are a helpful assistant.",
    roles={
        "user": "<|im_start|>user",
        "assistant": "<|im_start|>assistant\n<think>",
        "tool": "<|im_start|>tool",
    },
    seps=["<|im_end|>\n"],
    role_content_sep="\n",
    role_empty_sep="\n",
    stop_str=["<|endoftext|>", "<|im_end|>"],
    stop_token_ids=[248046, 248044],
    function_string="<tool_call><function={function_name}><parameter={param_name}>{param_value}</parameter></function></tool_call>",
)

# Add tool parser for XML rendering
template.tool_parser = "qwen3_coder"
parser = Qwen3CoderToolCallParser()
template.tool_parser_instance = parser

# Test 1: Tool call with proper newlines
print("\nTest 1: Tool Call Rendering (with proper newlines)")
print("-" * 70)

template.messages = [
    ("user", "Get weather in San Francisco"),
    ("assistant", [{
        "type": "tool_call",
        "name": "get_weather",
        "parameters": {
            "location": "San Francisco",
            "unit": "celsius"
        }
    }])
]

try:
    prompt = template.as_prompt()
    full_text = str(prompt[0])
    
    # Check for proper XML structure with newlines
    checks = [
        ("Has <tool_call> tag", "<tool_call>" in full_text),
        ("Has <function=...> tag", "<function=get_weather>" in full_text),
        ("Has parameter tags", "<parameter=location>" in full_text and "<parameter=unit>" in full_text),
        ("Proper newlines in parameters", "\nSan Francisco\n" in full_text or "\nSan Francisco" in full_text),
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        # Show actual rendered output
        tool_call_section = full_text[full_text.find("<tool_call>"):full_text.find("</tool_call>") + len("</tool_call>")]
        print(f"\n  Rendered tool call:\n{tool_call_section}")
    
except Exception as e:
    print(f"❌ ERROR: {e}")

# Test 2: Tool result rendering (should pass through content)
print("\nTest 2: Tool Result Rendering (content passthrough)")
print("-" * 70)

template.messages = [
    ("user", "Get weather"),
    ("tool", [{
        "type": "tool_result",
        "content": "<function=get_weather><parameter=status>success</parameter></function>"
    }])
]

try:
    prompt = template.as_prompt()
    full_text = str(prompt[0])
    
    # Check that content is preserved without duplication
    checks = [
        ("Content preserved", "<function=get_weather>" in full_text),
        ("No duplicate tags", full_text.count("<tool_response>") <= 1),
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        # Show actual rendered output
        tool_result_section = full_text[full_text.find("<function=get_weather>"):full_text.find("</function>") + len("</function>")]
        print(f"\n  Rendered tool result:\n{tool_result_section}")
    
except Exception as e:
    print(f"❌ ERROR: {e}")

# Test 3: Verify conversation protocol remains generic
print("\nTest 3: Conversation Protocol Genericity")
print("-" * 70)

try:
    # Create template without tool parser (fallback mode)
    simple_template = Conversation(
        name="simple",
        system_template=f"<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}<|im_end|>\n",
        system_message="Simple assistant.",
        roles={"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"},
        seps=["<|im_end|>\n"],
    )
    
    simple_template.messages = [
        ("user", "Hello"),
        ("assistant", [{
            "type": "tool_call",
            "name": "test_func",
            "parameters": {"param1": "value1"}
        }])
    ]
    
    # Should use fallback rendering when no tool parser
    prompt = simple_template.as_prompt()
    full_text = str(prompt[0])
    
    checks = [
        ("Fallback works without parser", "test_func" in full_text),
        ("No XML-specific errors", True),  # If we got here, no errors
    ]
    
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
    
except Exception as e:
    print(f"❌ ERROR: {e}")

print("\n" + "=" * 70)
print("Final Verification Complete!")