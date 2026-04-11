# Updated Requirements Summary

## Critical Updates to Implementation Plans

### 1. Parallel Tool Calls Support Added ✅

**New Requirement**: The `parallel_tool_calls` request parameter must be supported.

**Behavior**:
- If `false` or not specified: **request ends after the first tool call** (default behavior)
- If `true`: allow multiple tool calls in single request
- Future extension: duplicate/repeating tool call prevention mechanism

**Implementation Strategy**:

#### Option A: Parser-Level Control (Recommended)
```python
class Qwen3CoderToolCallParser(BaseToolParser):
    def __init__(self, allow_parallel_tool_calls=False):
        self._buffer = ""
        self.allow_parallel_tool_calls = allow_parallel_tool_calls  # Default: False
    
    def parse_streaming(self, token: str) -> Optional[List[Dict[str, Any]]]:
        self._buffer += token.strip()
        
        if "</tool_call>" not in self._buffer:
            return None
        
        # Extract all complete tool calls from buffer
        results = []
        while "</tool_call>" in self._buffer:
            try:
                result = self._parse_single_tool_call(self._buffer)
                if result:
                    results.append(result)
                    
                    # Stop after first tool call if not allowing parallel
                    if not self.allow_parallel_tool_calls and results:
                        break
                
                # Remove processed tool call from buffer
                self._buffer = self._buffer.split("</tool_call>", 1)[1]
            except Exception:
                break
        
        return results if results else None
```

#### Option B: Engine-Level Control
```python
# In engine layer, pass parallel_tool_calls to parser
parser = Qwen3CoderToolCallParser(
    allow_parallel_tool_calls=request.get("parallel_tool_calls", False)
)
```

**Test Cases**:
```python
def test_parser_stops_after_first_tool_call_by_default():
    """Default behavior: stop after first tool call."""
    parser = Qwen3CoderToolCallParser()
    
    # Send two complete tool calls in sequence
    result1 = parser.parse_streaming("</tool_call>")  # First tool call
    assert len(result1) == 1
    
    result2 = parser.parse_streaming("</tool_call>")  # Second tool call
    assert result2 is None  # Should not process second call by default

def test_parser_allows_parallel_tool_calls_when_enabled():
    """When enabled, allow multiple tool calls in single request."""
    parser = Qwen3CoderToolCallParser(allow_parallel_tool_calls=True)
    
    # Send two complete tool calls
    result1 = parser.parse_streaming("</tool_call>")
    assert len(result1) == 1
    
    result2 = parser.parse_streaming("</tool_call>")
    assert len(result2) == 1  # Should process second call when enabled
```

### 2. Corrected Test Runner Script ✅

**File**: `run_tests_correctly.sh` (new)

**Purpose**: Demonstrates the proper TDD testing workflow.

**Key Features**:
- Shows RED phase (watching tests fail is expected initially)
- Runs tests in isolation following TDD principles
- Reminds developers of the TDD workflow:
  1. RED: Watch tests fail (this is expected initially)
  2. GREEN: Implement minimal code to make tests pass
  3. REFACTOR: Clean up if needed
  4. COMMIT: `git add -A && git commit -m 'feat: [description]'`
  5. REVIEW: Spec compliance first, then code quality

**Usage**:
```bash
chmod +x run_tests_correctly.sh
./run_tests_correctly.sh
```

### 3. Updated Critical Requirements List ✅

All implementation plans now include the new critical requirement:

1. ✅ XML format with newlines (but handle without as well)
2. ✅ Tools in system prompt correctly rendered
3. ✅ Tool calls and results properly formatted
4. ✅ Parser is general concept, not Qwen3-specific
5. ✅ Only qwen3_5.py knows about specific parser implementation
6. ✅ No parser = normal behavior (backward compatible)
7. ✅ **CRITICAL**: Tool parser instance MUST be isolated per request
8. ✅ **NEW**: parallel_tool_calls parameter support with default behavior

### 4. Updated Success Criteria ✅

All success criteria now include the parallel tool calls requirement:

1. ✅ XML parser handles newlines and malformed input
2. ✅ Tool calls correctly rendered in conversation flow
3. ✅ Tool results correctly rendered
4. ✅ Parser is general concept, only qwen3_5.py knows specifics
5. ✅ No parser set = normal behavior (backward compatible)
6. ✅ **CRITICAL**: Parser instance isolated per request
7. ✅ **NEW**: Parallel tool calls parameter supported with default behavior
8. ✅ All tests pass (unit + integration + regression)
9. ✅ Documentation updated with new requirements

## Files Modified

### Updated Implementation Plans
1. ✅ `QWEN3_TOOL_PARSER_IMPLEMENTATION_PLAN_V2.md`
   - Added parallel_tool_calls requirement to critical requirements section
   - Added detailed implementation strategy for parallel tool calls
   - Added test cases for parallel tool calls behavior
   - Updated success criteria

### New Supporting Files
1. ✅ `run_tests_correctly.sh` - Correct TDD testing workflow demonstration
2. ✅ `UPDATED_REQUIREMENTS_SUMMARY.md` - This summary document

## Impact on Implementation

### Minimal Changes Required
The parallel_tool_calls feature can be implemented with minimal changes:

1. **Add parameter to parser constructor**: `allow_parallel_tool_calls=False`
2. **Update streaming logic**: Add conditional break after first tool call
3. **Add tests**: Verify default behavior (stop after first) and enabled behavior (allow multiple)

### Backward Compatibility Maintained
- Default behavior: `allow_parallel_tool_calls=False` ensures backward compatibility
- Existing code continues to work without changes
- New feature is opt-in through parameter

## Testing Strategy

### Unit Tests for Parallel Tool Calls
1. Test default behavior (stop after first tool call)
2. Test enabled behavior (allow multiple tool calls)
3. Test buffer management with parallel calls
4. Test edge cases (malformed XML, partial tags)

### Integration Tests for Parallel Tool Calls
1. Test request isolation with parallel calls
2. Test serialization/deserialization with parallel_tool_calls parameter
3. Test end-to-end workflow with multiple tool calls
4. Test concurrent requests with different parallel settings

## Future Extensions

The implementation provides foundation for:
- Duplicate tool call prevention
- Repeating tool call detection
- Tool call rate limiting
- Advanced tool call orchestration

These can be added in future phases without breaking existing functionality.
