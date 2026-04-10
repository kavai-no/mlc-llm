# Streaming Parser Fix Summary

## Problem Identified
The streaming parser in `Qwen3CoderToolCallParser` was incorrectly detecting complete tool calls due to a misunderstanding of how the regex capture groups work.

### Root Cause
The `FUNCTION_REGEX` pattern uses two capture groups:
```python
r"<function=(.*?)</function>|<function=(.*?)$"
```

When using `findall()`, this returns tuples like `(captured_content, '')`. The code correctly extracts the first non-None value, but the extracted content only includes what's between `<function=` and the closing tag - NOT the closing tag itself.

So for `<function=get_weather>...</function>`, it extracts `get_weather>...` but not `</function>`.

## Fix Applied

### 1. Corrected Complete Tool Call Detection (Line 238-245)
**Before:**
```python
for func_str in function_strs:
    tc = self._parse_function_call(func_str)
    if tc is not None:
        # Only add if the function string contains closing tags
        if "</function>" in func_str and "</tool_call>" in self._streaming_buffer:
            complete_tool_calls.append(tc)
```

**After:**
```python
# Check if we have a complete tool call (with closing tag)
has_complete_tool_call = "</tool_call>" in self._streaming_buffer and raw_blocks

for func_str in function_strs:
    tc = self._parse_function_call(func_str)
    if tc is not None and has_complete_tool_call:
        complete_tool_calls.append(tc)
```

### 2. Added Buffer Clearing (Line 250-251)
**Added:**
```python
if complete_tool_calls:
    # Clear the buffer since we've processed complete tool calls
    self._streaming_buffer = ""
    return {"type": "complete_tool_call", "data": complete_tool_calls}
```

This prevents buffer leaks between sequential calls.

## Test Results
All 26 related tests pass:
- `tests/test_streaming_parser.py` (6 tests)
- `tests/test_fragmented_xml.py` (4 tests)  
- `tests/test_xml_rendering.py` (6 tests)
- `tests/test_streaming_parser_fix.py` (5 tests)
- `tests/test_e2e_streaming_workflow.py` (5 tests)

## Key Behaviors Verified
1. ✅ Complete tool calls in single chunks are detected correctly
2. ✅ Multiple parameters work properly
3. ✅ Buffer is cleared after complete tool calls (no leaks)
4. ✅ Partial vs complete detection works correctly
5. ✅ Empty/whitespace chunks handled properly
6. ✅ Mixed content and tool calls work together
7. ✅ Fragmented XML across multiple chunks works end-to-end

## Files Modified
- `/workspace/projects/mlc-llm/python/mlc_llm/serve/tool_parser.py` (lines 238-251)

## Backward Compatibility
✅ All existing tests continue to pass
✅ No breaking changes to the API
✅ The fix only corrects incorrect behavior