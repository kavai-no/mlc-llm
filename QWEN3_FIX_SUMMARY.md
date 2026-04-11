# Qwen3.5 Tool Parser Integration Fix

## Problem Analysis

Qwen3.5 outputs XML format for tool calls:
```xml
<tool_call>
<function=get_weather>
<parameter=city>New York</parameter>
</function>
</tool_call>
```

But the mlc-llm engine expects JSON format and doesn't use the Qwen3CoderToolCallParser.

## Current State

✅ **Parser works correctly**: Both `parse()` and `parse_streaming()` methods work with XML format
✅ **Registry works**: Parser is registered as "qwen3_coder" and can be retrieved
✅ **Conversation template has parser**: Qwen3.5 templates include `tool_parser="qwen3_coder"`

❌ **Engine doesn't use it**: The streaming and non-streaming paths don't call the tool parser

## Required Fixes

### 1. Non-Streaming Path (process_function_call_output)

**Current**: Uses `convert_function_str_to_json()` which expects JSON format
**Fix**: Modify to use tool parser when available, fall back to JSON parsing

```python
def process_function_call_output(output_texts, finish_reasons):
    if use_function_calling:
        for i, output_text in enumerate(output_texts):
            try:
                # Try tool parser first
                fn_json_list = convert_function_str_to_json(
                    output_text, conv_template=conversation_template
                )
            except (SyntaxError, ValueError):
                # Handle errors
```

### 2. Streaming Path (process_chat_completion_stream_output)

**Current**: Only passes `delta.text` without parsing tool calls
**Fix**: Use tool parser to extract tool_calls from delta text

```python
# In process_chat_completion_stream_output:
tool_calls = None
if use_function_calling and delta_output.delta_text:
    try:
        parser = get_parser_instance("qwen3_coder")
        if parser:
            parse_result = parser.parse_streaming(delta_output.delta_text)
            if parse_result and parse_result.get("type") == "complete_tool_call":
                tool_calls = parse_result["data"]
    except Exception:
        pass  # Fall back to text-only

# Return tool_calls in the response
```

## Implementation Plan

1. **Modify `convert_function_str_to_json()`** to accept conversation template and use tool parser
2. **Update calls to `process_function_call_output()`** to pass conversation template context
3. **Enhance `process_chat_completion_stream_output()`** to parse streaming tool calls
4. **Test with actual Qwen3.5 model output** to verify end-to-end functionality

## Files to Modify

- `/workspace/projects/mlc-llm/python/mlc_llm/serve/engine_base.py`
  - `convert_function_str_to_json()` signature and logic
  - `process_chat_completion_stream_output()` streaming tool parsing
  
- `/workspace/projects/mlc-llm/python/mlc_llm/serve/engine.py`
  - Update calls to pass conversation template context
