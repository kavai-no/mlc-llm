# Summary of Changes for Tool Parser Integration

## Overview
Successfully integrated a dynamic tool parser system for Qwen3/Qwen3.5 models into mlc-llm using an XML-style format (`<tool_call><function name="...">...</function></tool_call>`), with model-agnostic parsing and proper hydration pattern.

## Key Changes

### 1. Protocol Extension (conversation_protocol.py)
- Added `BaseToolParser` abstract class with `parse()` and `parse_streaming()` methods
- Extended `Conversation` class to include:
  - `tool_parser: Optional[str]` field for serialization compatibility
  - `tool_parser_instance: Optional[Any]` property for runtime hydration
  - `@model_validator(mode='after')` decorator to automatically hydrate parser instances during deserialization
  - Custom `model_dump_json()` method that excludes the hydrated parser instance from JSON serialization
- Implemented `from_json_dict()` classmethod with automatic module registration and parser hydration

### 2. Parser Implementation (tool_parser.py)
- Created `Qwen3CoderToolCallParser` class inheriting from `BaseToolParser`
- Implemented XML parsing logic for nested tags: `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`
- Added parameter type conversion (null, JSON, Python literals, strings)
- Implemented registry system with `@register_parser()` decorator and `get_parser_instance()` lookup
- Supports both complete XML blocks and fragmented/unclosed tags for streaming scenarios

### 3. Registry System
- Global `PARSER_REGISTRY` dictionary for parser lookup
- Decorator-based registration: `@register_parser("qwen3_coder")`
- Runtime hydration via `get_parser_instance(name)`

### 4. Serialization Fixes
- Added custom `model_dump()` method in `MLCChatConfig` to exclude hydrated parser instances from JSON serialization
- Fixed `gen_config.py` to use conversation objects directly instead of attempting to serialize them with `.model_dump_json()`

## Testing
All tests pass:
- Fragmented XML parsing (complete and unclosed tags)
- Hydration pattern (string name → live instance conversion)
- Registry lookup and parser instantiation
- Streaming placeholder functionality

## Files Modified
1. `/workspace/projects/mlc-llm/python/mlc_llm/protocol/conversation_protocol.py`
2. `/workspace/projects/mlc-llm/python/mlc_llm/serve/tool_parser.py`
3. `/workspace/projects/mlc-llm/python/mlc_llm/protocol/mlc_chat_config.py`
4. `/workspace/projects/mlc-llm/python/mlc_llm/interface/gen_config.py`

## Design Decisions
1. **Hydration Pattern**: Store string identifier in JSON, hydrate to live instance during deserialization for request-level isolation
2. **Model-Agnostic**: No hardcoded model names; parser resolution via registry lookup
3. **Streaming Support**: Parser must handle fragmented XML tags (e.g., `<tool_call><func` as single token)
4. **Environment Isolation**: All tests run within `.venv` to avoid dependency conflicts
5. **Serialization Compatibility**: Only string identifiers are serialized; live instances excluded from JSON output
