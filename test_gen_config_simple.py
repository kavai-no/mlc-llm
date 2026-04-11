#!/usr/bin/env python3
"""
Simple test to verify qwen3_5 template registration without requiring full package import.
This tests the registry structure directly.
"""

def test_registry_structure():
    """Test that qwen3_5 is properly registered in the ConvTemplateRegistry."""
    
    # Read and parse the registry file
    with open('python/mlc_llm/conversation_template/registry.py', 'r') as f:
        content = f.read()
    
    print("Testing qwen3_5 template registration...")
    
    # Test 1: Check that qwen3_5 is registered
    if 'name="qwen3_5"' in content:
        print("✓ qwen3_5 template is registered in ConvTemplateRegistry")
    else:
        raise AssertionError("qwen3_5 template registration not found")
    
    # Test 2: Check that the template uses XML-style tool calling
    if 'Qwen3.5 XML-style tool calling' in content:
        print("✓ Template is documented as XML-style tool calling")
    else:
        raise AssertionError("XML-style documentation not found")
    
    # Test 3: Check that the template has proper structure
    checks = [
        ('<|im_start|>system', 'System start marker'),
        ('<|im_end|>', 'End marker'),
        ('"user": "<|im_start|>user"', 'User role definition'),
        ('"assistant": "<|im_start|>assistant"', 'Assistant role definition'),
    ]
    
    for pattern, description in checks:
        if pattern in content:
            print(f"✓ {description} found")
        else:
            raise AssertionError(f"{description} not found: {pattern}")
    
    # Test 4: Check that tool_parser is set to "qwen3_coder"
    if 'tool_parser="qwen3_coder"' in content or 'tool_parser: "qwen3_coder"' in content:
        print("✓ Tool parser is set to qwen3_coder")
    else:
        raise AssertionError("Tool parser not set correctly")
    
    # Test 5: Check that the template file exists and is imported
    with open('python/mlc_llm/conversation_template/qwen3_5.py', 'r') as f:
        template_content = f.read()
    
    if 'Qwen3_5_Template' in template_content:
        print("✓ Qwen3_5_Template class exists in qwen3_5.py")
    else:
        raise AssertionError("Qwen3_5_Template class not found")
    
    # Test 6: Check that the template is imported in __init__.py
    with open('python/mlc_llm/conversation_template/__init__.py', 'r') as f:
        init_content = f.read()
    
    if 'qwen3_5,' in init_content or 'from .qwen3_5 import' in init_content:
        print("✓ qwen3_5 template is imported in __init__.py")
    else:
        raise AssertionError("qwen3_5 import not found in __init__.py")
    
    print("\n" + "="*60)
    print("All registration structure tests passed!")
    print("="*60)

if __name__ == '__main__':
    test_registry_structure()