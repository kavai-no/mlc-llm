# Qwen3.5 Tool Parser Integration - Changes Summary

## Problem
Qwen3.5 outputs XML format for tool calls:
```xml
<tool_call>
<function=get_weather>
<parameter=city>New York</parameter>
</function>
</tool_call>
```

But mlc-llm expected JSON format and wasn't using the Qwen3CoderToolCallParser.

## Solution
Integrated the existing tool parser into both streaming and non-streaming paths.

## Changes Made

### 1. `process_function_call_output()` in engine_base.py
**File**: `/workspace/projects/mlc-llm/python/mlc_llm/serve/engine_base.py`

**Change**: Added logic to use tool parser when available, fall back to JSON parsing

```python
if conv_template is not None and hasattr(conv_template, 'tool_parser_instance') and conv_template.tool_parser_instance:
    content, tool_calls = conv_template.tool_parser_instance.parse(output_text)
    if tool_calls:
        tool_calls_list[i] = tool_calls  # Use ChatToolCall directly
    else:
        tool_calls_list[i] = []
else:
    # Fall back to original JSON parsing
    fn_json_list = convert_function_str_to_json(output_text)
    # ... existing conversion logic ...
```

**Key**: Returns `ChatToolCall` objects directly from parser, no conversion needed.

### 2. `process_chat_completion_stream_output()` in engine_base.py
**File**: `/workspace/projects/mlc-llm/python/mlc_llm/serve/engine_base.py`

**Changes**:
1. Added `conv_template` parameter (optional, for backward compatibility)
2. Added streaming tool parsing logic:

```python
tool_calls = None
if use_function_calling and conv_template and hasattr(conv_template, 'tool_parser_instance') and conv_template.tool_parser_instance:
    try:
        parse_result = conv_template.tool_parser_instance.parse_streaming(delta_output.delta_text)
        if parse_result and parse_result.get("type") == "complete_tool_call":
            tool_calls = parse_result["data"]
    except Exception:
        pass

# Return in response
delta=openai_api_protocol.ChatCompletionMessage(
    content="" if tool_calls else delta_output.delta_text, 
    role="assistant", 
    tool_calls=tool_calls  # Include tool calls in streaming response
)
```

### 3. Updated call in engine.py
**File**: `/workspace/projects/mlc-llm/python/mlc_llm/serve/engine.py`

**Change**: Pass conversation template to streaming output processor

```python
response = engine_base.process_chat_completion_stream_output(
    delta_outputs,
    request,
    request_id,
    self.state,
    use_function_calling,
    finish_reasons,
    conv_template.model_copy(deep=True),  # Added this parameter
)
```

## Testing
- ✅ Parser works with XML format (verified with test scripts)
- ✅ Tool parser hydration through Conversation protocol works
- ✅ Non-streaming path uses tool parser when available
- ✅ Streaming path parses tool calls incrementally
- ✅ Backward compatibility maintained (fallback to JSON parsing)

## Impact
- **Minimal changes**: Only modified the engine integration points
- **Backward compatible**: Existing JSON-based parsers still work
- **No breaking changes**: All existing functionality preserved
- **Future-proof**: Tool parser system is now actually used by the engine