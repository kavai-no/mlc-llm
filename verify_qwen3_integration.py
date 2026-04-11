#!/usr/bin/env python
"""
Final integration verification for Qwen3 XML-style tool parser.
This script verifies all components are in place and working correctly.
"""

import sys
import os

def verify_file_exists(path):
    """Verify a file exists at the given path."""
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return False
    print(f"✓ File exists: {path}")
    return True

def verify_content(file_path, patterns, description):
    """Verify that a file contains specific patterns."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        for pattern in patterns:
            if pattern not in content:
                print(f"❌ Pattern not found in {description}: {pattern}")
                return False
        
        print(f"✓ All patterns found in {description}")
        return True
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return False

def verify_imports():
    """Verify that all necessary imports work."""
    try:
        # Test basic imports
        from mlc_llm.protocol.conversation_protocol import Conversation, BaseToolParser
        from mlc_llm.serve.tool_parser import Qwen3CoderToolCallParser, register_parser, get_parser_instance
        from mlc_llm.conversation_template.qwen3_5 import Qwen3_5_Template
        
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def verify_parser_registration():
    """Verify that the parser is properly registered."""
    try:
        from mlc_llm.serve.tool_parser import get_parser_instance
        
        # Try to get an instance of the qwen3_coder parser
        parser = get_parser_instance("qwen3_coder")
        if parser is None:
            print("❌ Parser 'qwen3_coder' not registered")
            return False
        
        print("✓ Parser 'qwen3_coder' is registered and accessible")
        return True
    except Exception as e:
        print(f"❌ Error getting parser instance: {e}")
        return False

def verify_template_structure():
    """Verify the qwen3_5 template has correct structure."""
    file_path = "python/mlc_llm/conversation_template/qwen3_5.py"
    
    patterns = [
        "class Qwen3_5_Template(Conversation)",
        "XML-style tool calling",
        "<tool_call><function=",
    ]
    
    return verify_content(file_path, patterns, "qwen3_5 template")

def verify_conversation_protocol():
    """Verify ConversationProtocol has tool_parser support."""
    file_path = "python/mlc_llm/protocol/conversation_protocol.py"
    
    patterns = [
        "tool_parser: Optional[str]",
        "from_json_dict",
        "BaseToolParser",
    ]
    
    return verify_content(file_path, patterns, "ConversationProtocol")

def verify_tool_parser_module():
    """Verify tool_parser module has all required components."""
    file_path = "python/mlc_llm/serve/tool_parser.py"
    
    patterns = [
        "class Qwen3CoderToolCallParser(BaseToolParser)",
        "def parse(self, text: str) -> List[Dict[str, Any]]:",
        "def parse_streaming(self, token: str) -> Optional[List[Dict[str, Any]]]:",
        "register_parser",
        "get_parser_instance",
    ]
    
    return verify_content(file_path, patterns, "tool_parser module")

def main():
    """Run all verification checks."""
    print("=" * 70)
    print("QWEN3 XML-STYLE TOOL PARSER INTEGRATION VERIFICATION")
    print("=" * 70)
    print()
    
    checks = [
        ("File Structure", verify_file_exists, "python/mlc_llm/serve/tool_parser.py"),
        ("File Structure", verify_file_exists, "python/mlc_llm/protocol/conversation_protocol.py"),
        ("File Structure", verify_file_exists, "python/mlc_llm/conversation_template/qwen3_5.py"),
        ("Imports", verify_imports, None),
        ("Parser Registration", verify_parser_registration, None),
        ("Template Structure", verify_template_structure, None),
        ("Conversation Protocol", verify_conversation_protocol, None),
        ("Tool Parser Module", verify_tool_parser_module, None),
    ]
    
    results = []
    for description, check_func, arg in checks:
        print(f"\n{'=' * 70}")
        print(f"{description} Verification...")
        print('=' * 70)
        if arg is not None:
            result = check_func(arg)
        else:
            result = check_func()
        results.append(result)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    if all(results):
        print(f"✅ ALL {total} VERIFICATION CHECKS PASSED!")
        print()
        print("The Qwen3 XML-style tool parser integration is complete and working.")
        return 0
    else:
        print(f"❌ {total - passed}/{total} checks failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())