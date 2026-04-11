# Qwen3 Tool Parser Implementation Plan

## Overview
This plan outlines the implementation of XML-style tool calling support for Qwen3 models in mlc-llm. The goal is to add proper tool parser integration while maintaining compatibility with existing conversation templates.

## Current State Analysis

### Working Components (from commit d46f65fc)
1. **Qwen3.5 Conversation Templates** (`python/mlc_llm/conversation_template/qwen3_5.py`)
   - `qwen3_5`: Thinking enabled with `<think>` block
   - `qwen3_5_nothink`: Thinking disabled with closed empty `<think>` block
   - Both templates registered in `ConvTemplateRegistry`

### Missing Components (to be implemented)
1. **Tool Parser Registry** - Need to add parser registration system
2. **XML Tool Call Format Support** - Qwen3 uses XML format: `<tool_call><function=name>...</function></tool_call>`
3. **Streaming Parser** - Handle incremental token output from models
4. **Integration with Conversation Protocol** - Add tool_parser field to Conversation class
5. **Hydration Pattern** - Serialize/deserialize parsers for API compatibility

## Implementation Plan

### Phase 1: Tool Parser Infrastructure

#### Task 1.1: Create Base Tool Parser Class
- Location: `python/mlc_llm/serve/tool_parser.py`
- Create abstract base class `BaseToolParser` with interface:
  - `parse(text)` - Parse complete tool call output
  - `parse_streaming(token)` - Parse incremental token output
  - `supports_streaming()` - Return boolean

#### Task 1.2: Implement Qwen3 XML Parser
- Location: `python/mlc_llm/serve/tool_parser.py`
- Create `Qwen3CoderToolCallParser` class extending `BaseToolParser`
- Handle XML format: `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`
- Implement streaming support with buffer management
- Handle fragmented XML input (partial tags)

#### Task 1.3: Create Parser Registry
- Location: `python/mlc_llm/serve/tool_parser.py`
- Add registry functions:
  - `register_parser(name, parser_class)` - Register parser classes
  - `get_parser_instance(parser_name)` - Create instances from names
- Register Qwen3CoderToolCallParser as "qwen3_coder"

### Phase 2: Conversation Protocol Integration

#### Task 2.1: Extend Conversation Protocol
- Location: `python/mlc_llm/protocol/conversation_protocol.py`
- Add `tool_parser: Optional[str]` field to Conversation class
- Implement hydration pattern:
  - `to_json_dict()` - Serialize with string reference
  - `from_json_dict()` - Deserialize and create parser instance
  - Property `tool_parser_instance` - Runtime instantiation

#### Task 2.2: Update Qwen3 Template
- Location: `python/mlc_llm/conversation_template/qwen3_5.py`
- Modify templates to support tool calling:
  - Add function_string for XML format
  - Keep existing thinking blocks intact
  - Ensure backward compatibility

### Phase 3: Testing and Validation

#### Task 3.1: Unit Tests
- Create test file: `tests/test_qwen3_tool_parser.py`
- Test cases:
  - Complete XML parsing
  - Fragmented XML streaming
  - Buffer management
  - Multiple concurrent tool calls
  - Error handling

#### Task 3.2: Integration Tests
- Create test file: `tests/test_qwen3_integration.py`
- Test cases:
  - Conversation serialization/deserialization
  - Parser hydration
  - End-to-end workflow with streaming

#### Task 3.3: Regression Tests
- Verify existing functionality still works
- Run existing test suite
- Check conversation template registration

### Phase 4: Documentation

#### Task 4.1: Update Docstrings
- Add documentation to tool_parser.py
- Document XML format in conversation templates
- Add usage examples

#### Task 4.2: Create Usage Guide
- Location: `docs/tool_calling.md`
- Document Qwen3 tool calling workflow
- Provide examples for different use cases

## Technical Specifications

### XML Format Specification
```xml
<tool_call>
  <function=get_weather>
    <parameter=location>San Francisco</parameter>
    <parameter=unit>celsius</parameter>
  </function>
</tool_call>
```

### Streaming Behavior
1. Parser receives tokens incrementally
2. Buffer accumulates partial XML
3. When complete `<tool_call>` tag detected:
   - Extract tool call from buffer
   - Clear buffer
   - Return parsed result
4. Continue with remaining content

### Hydration Pattern
```python
# Serialization
conversation.tool_parser = "qwen3_coder"
json_data = conversation.to_json_dict()

# Deserialization
restored_conv = Conversation.from_json_dict(json_data)
parser = restored_conv.tool_parser_instance  # Creates live instance
```

## Success Criteria

1. ✅ Qwen3 XML tool calls parsed correctly
2. ✅ Streaming support works with fragmented input
3. ✅ Parser instances isolated per request (hydration pattern)
4. ✅ Backward compatibility maintained
5. ✅ All tests pass (unit + integration + regression)
6. ✅ Documentation complete and accurate
7. ✅ No memory leaks from buffer management

## Risk Mitigation

### Potential Issues
1. **Regex Complexity**: XML parsing with regex can be fragile
   - Solution: Test extensively with real model outputs
2. **Streaming Edge Cases**: Partial tags, special characters
   - Solution: Comprehensive fragmented XML tests
3. **Serialization Compatibility**: API changes breaking existing code
   - Solution: Maintain backward compatibility, deprecate old APIs gracefully

### Contingency Plan
If implementation proves too complex:
1. Fall back to simpler JSON-based tool calling first
2. Add XML support as separate feature flag
3. Document limitations clearly

## Timeline

- **Phase 1**: 2 days (Tool Parser Infrastructure)
- **Phase 2**: 1 day (Conversation Protocol Integration)
- **Phase 3**: 2 days (Testing and Validation)
- **Phase 4**: 0.5 day (Documentation)

**Total**: ~5.5 days with buffer for unexpected issues

## Dependencies

### External
- Qwen3 model outputs in XML format
- Existing mlc-llm conversation protocol

### Internal
- Conversation template registration system
- JSON serialization infrastructure

## Files to Modify

1. `python/mlc_llm/serve/tool_parser.py` - Add parser classes and registry
2. `python/mlc_llm/protocol/conversation_protocol.py` - Extend with tool_parser field
3. `python/mlc_llm/conversation_template/qwen3_5.py` - Update for tool calling
4. `tests/test_qwen3_tool_parser.py` - Unit tests (new)
5. `tests/test_qwen3_integration.py` - Integration tests (new)
6. `docs/tool_calling.md` - Usage documentation (new)

## Verification Steps

After implementation, run:
```bash
# Test parser functionality
python -m pytest tests/test_qwen3_tool_parser.py -v

# Test integration
python -m pytest tests/test_qwen3_integration.py -v

# Verify conversation templates still work
python test_qwen3_registration.py

# End-to-end verification
python verify_integration_simple.py
```

All tests should pass with no regressions in existing functionality.