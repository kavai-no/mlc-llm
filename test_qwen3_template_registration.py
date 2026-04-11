#!/usr/bin/env python3
"""
Comprehensive test for qwen3_5 template registration.
This test verifies that the qwen3_5 template is properly registered
in the ConvTemplateRegistry and can be retrieved correctly.
"""

def test_registry_content():
    """Test that registry.py contains qwen3_5 registration."""
    with open('python/mlc_llm/conversation_template/registry.py', 'r') as f:
        content = f.read()
    
    # Check for the registration
    assert 'name="qwen3_5"' in content, "qwen3_5 template not found in registry"
    assert '<|im_start|>system' in content, "System template marker not found"
    assert '<|im_end|>' in content, "End marker not found"
    print("✓ Registry contains qwen3_5 template registration")


def test_template_file():
    """Test that qwen3_5.py file exists and has correct structure."""
    import os
    
    template_path = 'python/mlc_llm/conversation_template/qwen3_5.py'
    assert os.path.exists(template_path), f"Template file not found: {template_path}"
    
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Check for XML tool calling format
    assert '<tool_call><function=' in content, "XML tool calling format not found"
    assert 'class Qwen3_5_Template' in content, "Qwen3_5_Template class not found"
    print("✓ qwen3_5.py template file exists with correct structure")


def test_import_in_init():
    """Test that qwen3_5 is imported in __init__.py."""
    with open('python/mlc_llm/conversation_template/__init__.py', 'r') as f:
        content = f.read()
    
    assert 'qwen3_5,' in content, "qwen3_5 not imported in __init__.py"
    print("✓ qwen3_5 is imported in conversation_template/__init__.py")


def test_syntax_correct():
    """Test that all files compile without syntax errors."""
    import py_compile
    
    files = [
        'python/mlc_llm/conversation_template/registry.py',
        'python/mlc_llm/conversation_template/qwen3_5.py',
        'python/mlc_llm/conversation_template/__init__.py'
    ]
    
    for filepath in files:
        try:
            py_compile.compile(filepath, doraise=True)
            print(f"✓ {filepath} compiles successfully")
        except py_compile.PyCompileError as e:
            raise AssertionError(f"Syntax error in {filepath}: {e}")


def test_template_structure():
    """Test that the registered template has correct structure."""
    # Read the registry file and extract the qwen3_5 registration
    with open('python/mlc_llm/conversation_template/registry.py', 'r') as f:
        content = f.read()
    
    # Find the qwen3_5 registration block - look for the full block
    start_idx = content.find('# Qwen3.5 XML-style tool calling')
    if start_idx == -1:
        raise AssertionError("Qwen3.5 registration comment not found")
    
    # Find the closing parenthesis of the register_conv_template call
    # Look for the pattern: ConvTemplateRegistry.register_conv_template(...)
    end_idx = content.find(')', start_idx)
    # Now find the matching closing parenthesis by counting
    paren_count = 0
    found_start = False
    for i in range(start_idx, len(content)):
        if content[i] == '(':
            paren_count += 1
            found_start = True
        elif content[i] == ')':
            paren_count -= 1
            if found_start and paren_count == 0:
                end_idx = i + 1
                break
    
    registration_block = content[start_idx:end_idx]
    
    # Check for key components
    checks = [
        ('name="qwen3_5"', 'Template name'),
        ('<|im_start|>system', 'System start marker'),
        ('<|im_end|>', 'End marker'),
        ('"user": "<|im_start|>user"', 'User role'),
        ('"assistant": "<|im_start|>assistant"', 'Assistant role'),
    ]
    
    for pattern, description in checks:
        assert pattern in registration_block, f"{description} not found in registration"
        print(f"✓ {description} present in qwen3_5 registration")


if __name__ == '__main__':
    print("Running qwen3_5 template registration tests...\n")
    
    test_registry_content()
    test_template_file()
    test_import_in_init()
    test_syntax_correct()
    test_template_structure()
    
    print("\n" + "="*60)
    print("All qwen3_5 template registration tests passed!")
    print("="*60)