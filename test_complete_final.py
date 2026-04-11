#!/usr/bin/env python3
"""
Complete final test - verifies all Qwen3 tool calling functionality.
"""

import sys
sys.path.insert(0, '/workspace/projects/mlc-llm/python')

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders
from mlc_llm.serve.tool_parser import Qwen3CoderToolCallParser

print("Complete Final Test - Qwen3 Tool Calling")
print("=" * 70)

# Create conversation template matching qwen3_5
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
)

# Add tool parser for XML rendering
template.tool_parser = "qwen3_coder"
parser = Qwen3CoderToolCallParser()
template.tool_parser_instance = parser

# Test complete workflow
template.messages = [
    ("user", "What's the weather in San Francisco?"),
    ("assistant", [
        {"type": "text", "text": "Let me check."},
        {"type": "tool_call", "name": "get_weather", "parameters": {
            "location": "San Francisco",
            "unit": "celsius"
        }}
    ]),
    ("tool", [{
        "type": "tool_result", "content": "<function=get_weather><parameter=status>success</parameter><parameter=temperature>15.5</parameter></function>"
    }]),
]

try:
    prompt = template.as_prompt()
    full_text = str(prompt[0])
    
    print("✅ SUCCESS: Complete workflow rendered!")
    print(f"\nPrompt length: {len(full_text)} characters")
    
    # Verify key elements
    checks = [
        ("System message", "You are a helpful assistant." in full_text),
        ("User message", "What's the weather in San Francisco?" in full_text),
        ("Assistant text", "Let me check." in full_text),
        ("Tool call XML", "<tool_call>" in full_text and "</tool_call>" in full_text),
        ("Function name", "<function=get_weather>" in full_text),
        ("Parameters with newlines", "\nSan Francisco\n" in full_text or "\nSan Francisco" in full_text),
        ("Tool result content", "<parameter=status>success</parameter>" in full_text),
    ]
    
    print("\nVerification:")
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests PASSED!")
        print("\nSample output (tool call section):")
        start = full_text.find("<tool_call>")
        end = full_text.find("</tool_call>") + len("</tool_call>")
        if start != -1 and end != -1:
            print(full_text[start:end])
    else:
        print("\n❌ Some tests failed!")
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Test Complete!")