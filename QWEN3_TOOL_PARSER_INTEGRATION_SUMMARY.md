# Qwen3 XML-Style Tool Parser Integration Summary

## Overview
Successfully integrated Qwen3 XML-style tool parser into mlc-llm engine with support for streaming, request isolation via hydration pattern, and proper JSON serialization.

## Key Components

### 1. Parser Registry (`python/mlc_llm/serve/tool_parser.py`)
- **BaseToolParser**: Abstract base class defining interface
- **Qwen3CoderToolCallParser**: Implementation for Qwen3 XML format
  - Parses: `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`
  - Supports streaming with `parse_streaming` method
  - Handles fragmented XML input
- **Registry Functions**:
  - `register_parser(name, parser_class)`: Register parser classes
  - `get_parser_instance(parser_name)`: Create instances from names

### 2. Conversation Protocol (`python/mlc_llm/protocol/conversation_protocol.py`)
- Extended `ConversationProtocol` with:
  - `tool_parser: Optional[str]` field (string-based reference)
  - `from_json_dict` classmethod for hydration pattern
  - Property accessor `tool_parser_instance` for runtime instantiation

### 3. Qwen3 Template (`python/mlc_llm/conversation_template/qwen3_5.py`)
- `Qwen3_5_Template` class extending `Conversation`
- Defines XML tool call format in `function_string`
- Uses string-based parser reference (no direct instantiation)

### 4. Template Registration (`python/mlc_llm/conversation_template/registry.py`)
- Registered qwen3_5 template in `ConvTemplateRegistry`
- Includes proper structure with name and tool_parser fields

## Design Decisions

### Hydration Pattern
**Problem**: Need to serialize Conversation objects to JSON but parsers are class instances.

**Solution**: Store parser as string reference, hydrate at runtime.
```python
# Serialization
conv_json = conversation.to_json_dict()  # tool_parser: "qwen3_coder"

# Deserialization
conversation = Conversation.from_json_dict(conv_json)  # Creates live instance
```

### Request Isolation
**Problem**: Multiple concurrent requests need isolated parsers.

**Solution**: Parser lives on Conversation object (cloned per request).
```python
# Each request gets its own conversation clone
request_conv = conversation.clone()
parser = request_conv.tool_parser_instance  # Isolated instance
```

### Streaming Support
**Problem**: Model outputs tokens incrementally, need to parse partial XML.

**Solution**: `parse_streaming` method with buffer management.
- Accumulates incomplete tags
- Returns complete tool calls as they become available
- Clears buffer after returning results

## Bug Fixes

### 1. Streaming Parser Function Name Extraction
**Issue**: Regex used `finditer` + `match.group(0)` which captured entire `<function=name>` tag instead of just the name.

**Fix**: Use `findall` to extract capture groups directly:
```python
# Before (incorrect)
matches = list(re.finditer(pattern, buffer))
for match in matches:
    function_name = match.group(0)  # Returns "<function=get_weather>"

# After (correct)
matches = re.findall(pattern, buffer)
for match in matches:
    function_name = match[1]  # Returns "get_weather"
```

### 2. Complete Tool Call Detection
**Issue**: Parser checked for `</function>` in extracted content, but regex only captures content between tags.

**Fix**: Check for `</tool_call>` in buffer:
```python
# Before (incorrect)
content = match.group(2)  # Only the content inside <function>...</function>
if "</function>" in content:  # Never true!
    return complete_tool_calls

# After (correct)
if "</tool_call>" in buffer:  # Check full buffer
    return complete_tool_calls
```

### 3. Buffer Leak Prevention
**Issue**: Buffer accumulated across parser calls, causing memory leaks.

**Fix**: Clear buffer after returning complete tool calls:
```python
if "</tool_call>" in buffer:
    complete_tool_calls = extract_complete_tool_calls()
    buffer = ""  # Clear buffer
    return complete_tool_calls
```

## Test Coverage

### Streaming Parser Fix Tests (5 tests)
- Fragmented function name extraction
- Complete tool call detection
- Buffer clearing behavior
- Multiple concurrent tool calls
- Edge cases with malformed XML

### XML Rendering Tests (6 tests)
- Basic tool call rendering
- Multiple parameters
- Nested parameter structures
- Special characters in values
- Empty parameter handling
- Complex real-world scenarios

### End-to-End Hydration Tests (5 tests)
- Serialization/deserialization cycle
- Parser instance creation from JSON
- Conversation cloning with parsers
- Multiple parser types
- Error handling for unknown parsers

### End-to-End Streaming Workflow Tests (5 tests)
- Incremental token processing
- Partial tag accumulation
- Complete tool call extraction
- Buffer management across calls
- Real-world streaming scenarios

### Fragmented XML Tests (4 tests)
- Partial opening tags
- Mid-tag fragmentation
- Multiple concurrent fragments
- Edge cases with special characters

**Total**: 25 tests, all passing ✅

## Usage Example

```python
from mlc_llm.conversation_template import Qwen3_5_Template
from mlc_llm.serve.tool_parser import get_parser_instance

# Create conversation with parser
conversation = Qwen3_5_Template()
conversation.tool_parser = "qwen3_coder"

# Serialize to JSON (for API responses, storage, etc.)
json_data = conversation.to_json_dict()

# Deserialize and hydrate (on another machine/request)
restored_conv = Conversation.from_json_dict(json_data)
parser = restored_conv.tool_parser_instance

# Parse streaming output
tokens = ["<tool_call>", "<function=get_weather>", "...", "</tool_call>"]
for token in tokens:
    result = parser.parse_streaming(token)
    if result:
        print(f"Complete tool call: {result}")
```

## Files Modified

1. `python/mlc_llm/protocol/conversation_protocol.py`
   - Added tool_parser field and hydration logic

2. `python/mlc_llm/serve/tool_parser.py`
   - Fixed streaming parser bugs
   - Added registry functions

3. `python/mlc_llm/conversation_template/qwen3_5.py`
   - Updated to use string-based parser reference

4. `python/mlc_llm/conversation_template/registry.py`
   - Registered qwen3_5 template

## Verification

Run verification script:
```bash
cd /workspace/projects/mlc-llm
python verify_integration_simple.py
```

All tests pass:
```bash
./run_tests.sh tests/test_streaming_parser_fix.py
./run_tests.sh tests/test_xml_rendering.py  
./run_tests.sh tests/test_e2e_hydration.py
./run_tests.sh tests/test_e2e_streaming_workflow.py
./run_tests.sh tests/test_fragmented_xml.py
```

## Future Work

1. Test with actual Qwen3 model outputs in production scenarios
2. Add more parser implementations for other XML-based formats
3. Optimize streaming performance for high-throughput scenarios
4. Add validation for tool call structure and parameters