# Qwen3 Tool Parser Implementation Plan v2

## Critical Requirements (MUST be implemented)

1. **XML format has newlines** - but parser should handle without as well
2. **Tools part of system prompt** - must include available tools correctly rendered
3. **Tool calls and tool call results** - must be correctly rendered in conversation
4. **Tool parser placement** - should be used at the right place, most sensible location
5. **General concept** - tool parser should be a general concept usable by other models
6. **Specific knowledge isolation** - only qwen3_5 conversation template should know about specific parser implementation
7. **No parser behavior** - if no parser is set then normal behavior is expected (to be refactored to default tool parser in later phase)
8. **CRITICAL**: Tool parser instance MUST be isolated per request
9. **parallel_tool_calls parameter** - support parallel_tool_calls request parameter:
   - If false or not specified: request ends after first tool call (default behavior)
   - If true: allow multiple tool calls in single request
   - Future: duplicate/repeating tool call prevention mechanism

## Updated Requirements Analysis

### XML Format Specification (Updated)
```xml
<tool_call>
  <function=get_weather>
    <parameter=location>San Francisco</parameter>
    <parameter=unit>celsius</parameter>
  </function>
</tool_call>
```

**Key Requirements:**
- Parser must handle XML with newlines (but also work without)
- Tool calls and tool call results must be correctly rendered in conversation
- Tool parser should be a general concept usable by other models
- Only Qwen3.5 template knows about specific parser implementation
- No parser set = normal behavior (to be refactored to default parser later)
- **CRITICAL**: Tool parser instance MUST be isolated per request

## Architecture Overview

### Core Components
1. **Tool Parser Registry** - General concept, not Qwen3-specific
2. **BaseToolParser Interface** - Abstract base class for all parsers
3. **Qwen3CoderToolCallParser** - Implementation for Qwen3 XML format
4. **Conversation Protocol Extension** - Add tool_parser field with hydration
5. **Request Isolation Mechanism** - Per-request parser instances

### Integration Points
- **Conversation Template**: Uses string reference to parser ("qwen3_coder")
- **Engine/Server Layer**: Hydrates parser instance per request
- **Protocol Layer**: Serializes/deserializes conversation with parser reference

## Implementation Plan

### Phase 1: Tool Parser Infrastructure (General Concept)

#### Task 1.1: Create BaseToolParser Interface
**File**: `python/mlc_llm/serve/tool_parser.py`

```python
class BaseToolParser(ABC):
    """Abstract base class for tool call parsers."""
    
    @abstractmethod
    def parse(self, text: str) -> List[Dict[str, Any]]:
        """Parse complete tool call output."""
        pass
    
    @abstractmethod
    def parse_streaming(self, token: str) -> Optional[List[Dict[str, Any]]]:
        """Parse incremental token output. Returns None or list of complete tool calls."""
        pass
    
    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether this parser supports streaming."""
        pass
```

#### Task 1.2: Implement Qwen3CoderToolCallParser
**File**: `python/mlc_llm/serve/tool_parser.py`

```python
class Qwen3CoderToolCallParser(BaseToolParser):
    """Parser for Qwen3 XML-style tool calls."""
    
    def __init__(self):
        self._buffer = ""  # For streaming
    
    def parse(self, text: str) -> List[Dict[str, Any]]:
        """Parse complete XML tool call output."""
        # Implementation handles newlines and malformed XML
        pass
    
    def parse_streaming(self, token: str) -> Optional[List[Dict[str, Any]]]:
        """Parse incremental tokens with buffer management."""
        self._buffer += token
        
        # Check for complete tool calls
        if "</tool_call>" in self._buffer:
            result = self._extract_complete_tool_calls()
            self._buffer = ""  # Clear buffer after extraction
            return result
        
        return None
    
    @property
    def supports_streaming(self) -> bool:
        return True
```

#### Task 1.3: Create Parser Registry (General)
**File**: `python/mlc_llm/serve/tool_parser.py`

```python
class ToolParserRegistry:
    """Registry for tool parsers - general concept, not Qwen3-specific."""
    
    _parsers = {}  # name -> parser class
    
    @classmethod
    def register_parser(cls, name: str, parser_class: Type[BaseToolParser]):
        """Register a parser class by name."""
        cls._parsers[name] = parser_class
    
    @classmethod
    def get_parser_instance(cls, name: str) -> BaseToolParser:
        """Create a parser instance from its name."""
        if name not in cls._parsers:
            raise ValueError(f"Unknown parser: {name}")
        return cls._parsers[name]()
    
    @classmethod
    def get_available_parsers(cls) -> List[str]:
        """Get list of available parser names."""
        return list(cls._parsers.keys())

# Register Qwen3 parser (only this file knows about Qwen3-specific implementation)
ToolParserRegistry.register_parser("qwen3_coder", Qwen3CoderToolCallParser)
```

### Phase 2: Conversation Protocol Integration

#### Task 2.1: Extend Conversation Protocol
**File**: `python/mlc_llm/protocol/conversation_protocol.py`

Add to Conversation class:

```python
class Conversation:
    # ... existing fields ...
    
    tool_parser: Optional[str] = None  # String reference to parser name
    
    @property
    def tool_parser_instance(self) -> Optional[BaseToolParser]:
        """Get live parser instance. Returns None if no parser set."""
        if self.tool_parser is None:
            return None
        from mlc_llm.serve.tool_parser import ToolParserRegistry
        return ToolParserRegistry.get_parser_instance(self.tool_parser)
    
    def to_json_dict(self) -> Dict[str, Any]:
        """Serialize conversation including parser reference."""
        data = {
            # ... existing serialization ...
            "tool_parser": self.tool_parser,
        }
        return data
    
    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Deserialize conversation and hydrate parser."""
        conv = cls(
            # ... existing deserialization ...
            tool_parser=data.get("tool_parser"),
        )
        return conv
```

#### Task 2.2: Update Qwen3 Template for Tool Calling
**File**: `python/mlc_llm/conversation_template/qwen3_5.py`

Update to include:
- System prompt with tools section
- Function string for XML tool calls
- Tool call result rendering

```python
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="qwen3_5",
        system_template=f"""<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}
You are a helpful assistant. You must respond in XML format for tool calls.
Available tools: {MessagePlaceholders.TOOLS.value}<|im_end|>\n""",
        system_message="You are a helpful assistant.",
        roles={
            "user": "<|im_start|>user",
            "assistant": "<|im_start|>assistant\n<think>",
            "tool": "<|im_start|>tool",  # For tool call results
        },
        seps=["<|im_end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|endoftext|>", "<|im_end|>"],
        stop_token_ids=[248046, 248044],
        tool_parser="qwen3_coder",  # Only this template knows about specific parser
    )
)
```

### Phase 3: Request Isolation Mechanism

#### Task 3.1: Implement Per-Request Parser Instantiation
**File**: `python/mlc_llm/serve/engine_base.py` or appropriate engine file

```python
class EngineBase:
    # ... existing code ...
    
    def _get_conversation_for_request(self, request_id: str) -> Conversation:
        """
        Get conversation for a request with isolated parser instance.
        
        This is where parser isolation happens - each request gets its own
        conversation clone with fresh parser instance.
        """
        # Get base conversation (may have tool_parser string reference)
        base_conv = self._get_base_conversation()
        
        # Clone for this request
        conv = base_conv.clone()
        
        # If conversation has a parser, create isolated instance
        if conv.tool_parser:
            from mlc_llm.serve.tool_parser import ToolParserRegistry
            conv._tool_parser_instance = ToolParserRegistry.get_parser_instance(
                conv.tool_parser
            )
        
        return conv
```

#### Task 3.2: Update Conversation Clone Method
**File**: `python/mlc_llm/protocol/conversation_protocol.py`

```python
class Conversation:
    # ... existing code ...
    
    def clone(self) -> 'Conversation':
        """Create a deep copy of this conversation."""
        new_conv = Conversation(
            # ... copy all fields ...
            tool_parser=self.tool_parser,
        )
        return new_conv
```

### Phase 4: Testing and Validation

#### Task 4.1: Unit Tests for Parser
**File**: `tests/test_qwen3_tool_parser.py`

Test cases:
- XML with newlines vs without
- Fragmented streaming input
- Multiple concurrent tool calls
- Buffer management and clearing
- Error handling for malformed XML

#### Task 4.2: Integration Tests
**File**: `tests/test_qwen3_integration.py`

Test cases:
- Per-request parser isolation
- Conversation serialization/deserialization with parser
- Tool call rendering in conversation flow
- Tool result rendering
- No parser set = normal behavior

#### Task 4.3: Regression Tests
- Existing functionality unchanged
- Other conversation templates still work
- Backward compatibility maintained

## Critical Design Decisions

### 1. Parser Isolation Strategy
**Problem**: Multiple concurrent requests need isolated parsers

**Solution**: 
- Store parser as string reference in Conversation
- Create fresh instance per request in engine layer
- Never share parser instances between requests

```python
# Engine layer creates isolated conversation
request_conv = base_conversation.clone()
parser = request_conv.tool_parser_instance  # Fresh instance!
```

### 2. General vs Specific Knowledge
**Problem**: Tool parser should be general, but Qwen3 template needs specific implementation

**Solution**: 
- Registry and base class are general (know nothing about Qwen3)
- Only `tool_parser.py` knows about Qwen3-specific implementation
- Conversation templates reference parsers by string name only

**CRITICAL REQUIREMENT**: Only qwen3.5 conversation template should know about the specific parser implementation, not other parts of the codebase.

```python
# tool_parser.py - only file that knows about Qwen3 specifics
ToolParserRegistry.register_parser("qwen3_coder", Qwen3CoderToolCallParser)

# qwen3_5.py - only references parser by name
Conversation(tool_parser="qwen3_coder", ...)
```

### 3. No Parser = Normal Behavior
**Problem**: Backward compatibility when no parser is set

**Solution**: 
- `tool_parser_instance` property returns None if not set
- Engine checks for None and handles normally
- Later phase: Add default parser through configuration

```python
if conv.tool_parser_instance:
    # Use tool parser
    result = parser.parse_streaming(token)
else:
    # Normal behavior (backward compatible)
    result = normal_parse(token)
```

## XML Format Handling

### Requirements
1. Handle XML with newlines
2. Handle XML without newlines
3. Handle fragmented/malformed XML in streaming

### Implementation Strategy

```python
def parse_streaming(self, token: str) -> Optional[List[Dict[str, Any]]]:
    self._buffer += token.strip()  # Remove whitespace for robustness
    
    if "</tool_call>" not in self._buffer:
        return None  # Not complete yet
    
    # Extract all complete tool calls from buffer
    results = []
    while "</tool_call>" in self._buffer:
        try:
            result = self._parse_single_tool_call(self._buffer)
            if result:
                results.append(result)
            # Remove processed tool call from buffer
            self._buffer = self._buffer.split("</tool_call>", 1)[1]
        except Exception:
            break  # Stop on error, keep remaining in buffer
    
    return results if results else None
```

## Tool Call Rendering

### System Prompt with Tools
```python
system_template = f"""<|im_start|>system
{MessagePlaceholders.SYSTEM.value}
You are a helpful assistant. You must respond in XML format for tool calls.
Available tools: {MessagePlaceholders.TOOLS.value}<|im_end|>
"""
```

### Tool Call Format
```xml
<tool_call>
  <function=get_weather>
    <parameter=location>San Francisco</parameter>
    <parameter=unit>celsius</parameter>
  </function>
</tool_call>
```

### Tool Result Format
```xml
<tool_response>
  <function=get_weather>
    <parameter=status>success</parameter>
    <parameter=temperature>15.5</parameter>
  </function>
</tool_response>
```

## Parallel Tool Calls Support

### Requirements
1. **parallel_tool_calls parameter** - Add support for controlling multiple tool calls per request
2. **Default behavior** - If false or not specified, request ends after first tool call
3. **Future extension** - Foundation for duplicate/repeating tool call prevention

### Implementation Strategy

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

### Test Cases for Parallel Tool Calls

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

## Success Criteria (Updated)

1. ✅ XML parser handles newlines and malformed input
2. ✅ Tool calls correctly rendered in conversation flow
3. ✅ Tool results correctly rendered
4. ✅ Parser is general concept, only qwen3_5.py knows specifics
5. ✅ No parser set = normal behavior (backward compatible)
6. ✅ **CRITICAL**: Parser instance isolated per request
7. ✅ Parallel tool calls parameter supported with default behavior
8. ✅ All tests pass (unit + integration + regression)
9. ✅ Documentation updated with new requirements

## Risk Mitigation

### High Priority Risks
1. **Parser Isolation Leaks**
   - Mitigation: Comprehensive integration tests
   - Test multiple concurrent requests
   
2. **XML Parsing Complexity**
   - Mitigation: Extensive unit tests for edge cases
   - Test with real model outputs
   
3. **Backward Compatibility Breaks**
   - Mitigation: Regression tests for existing functionality
   - Feature flags if needed

### Contingency Plan
If implementation proves too complex:
1. Implement simpler version first (no streaming)
2. Add streaming as separate feature flag
3. Document limitations clearly
4. Provide clear upgrade path

## Files to Modify/Create

### Existing Files to Modify
1. `python/mlc_llm/serve/tool_parser.py` - Add parser infrastructure
2. `python/mlc_llm/protocol/conversation_protocol.py` - Extend with tool_parser
3. `python/mlc_llm/conversation_template/qwen3_5.py` - Update for tool calling
4. `python/mlc_llm/serve/engine_base.py` - Add request isolation logic

### New Test Files
1. `tests/test_qwen3_tool_parser.py` - Unit tests
2. `tests/test_qwen3_integration.py` - Integration tests
3. `tests/test_parser_isolation.py` - Request isolation tests

## Verification Steps

After implementation, verify:

```bash
# Test parser functionality
python -m pytest tests/test_qwen3_tool_parser.py -v

# Test integration and isolation
python -m pytest tests/test_qwen3_integration.py -v
python -m pytest tests/test_parser_isolation.py -v

# Verify conversation templates still work
python test_qwen3_registration.py

# End-to-end verification
python verify_integration_simple.py
```

All tests should pass with:
- No memory leaks from buffer management
- No shared state between requests
- Backward compatibility maintained
- XML format handling (with and without newlines)