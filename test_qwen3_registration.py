#!/usr/bin/env python3
"""Test script to verify qwen3_5 template registration."""

# Test 1: Check that the registry file has the qwen3_5 registration
print("Test 1: Checking registry file content...")
with open('python/mlc_llm/conversation_template/registry.py', 'r') as f:
    content = f.read()
    if 'name="qwen3_5"' in content:
        print("✓ qwen3_5 template registration found in registry.py")
    else:
        print("✗ qwen3_5 template NOT found in registry.py")

# Test 2: Check that the file compiles without syntax errors
print("\nTest 2: Checking Python syntax...")
import py_compile
try:
    py_compile.compile('python/mlc_llm/conversation_template/registry.py', doraise=True)
    print("✓ registry.py compiles successfully")
except py_compile.PyCompileError as e:
    print(f"✗ Syntax error in registry.py: {e}")

# Test 3: Verify the template structure matches qwen3_5.py
print("\nTest 3: Checking template structure...")
with open('python/mlc_llm/conversation_template/qwen3_5.py', 'r') as f:
    template_content = f.read()
    
checks = [
    ('XML-style tool calling format', '<tool_call><function=' in template_content),
    ('Qwen3_5_Template class exists', 'class Qwen3_5_Template' in template_content),
    ('Function string with placeholders', '{function_name}' in template_content),
]

for check_name, result in checks:
    if result:
        print(f"✓ {check_name}")
    else:
        print(f"✗ {check_name}")

print("\nAll basic tests completed!")