# Qwen3 Tool Parser TDD Implementation Plan

## Overview

This plan uses **TDD with clear API boundaries** and **subagent-driven development** to implement the Qwen3 tool parser integration. Each task will be executed by isolated subagents using `git worktree` to ensure no conflicts.

## Implementation Strategy

### Phase 1: Test-Driven Development Setup

#### Task 1.1: Create API Boundary Tests for Tool Parser
**Goal**: Write failing tests that define the public API for Qwen3 tool parser

```python
def test_qwen3_parser_module_exports():
    """Module should export Qwen3CoderToolCallParser."""
    from mlc_llm.serve.tool_parser import Qwen3CoderToolCallParser
    assert callable(Qwen3CoderToolCallParser)

def test_qwen3_parser_handles_xml_format():
    """Public API: parse() should handle XML tool calls."""
    from mlc_llm.serve.tool_parser import Qwen3CoderToolCallParser
    
    parser = Qwen3CoderToolCallParser()
    xml_input = "<tool_call><function=get_weather><parameter=location>SF</parameter></function></tool_call>"
    result = parser.parse(xml_input)
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[1], list)

def test_qwen3_parser_streaming():
    """Public API: parse_streaming() should handle incremental tokens."""
    from mlc_llm.serve.tool_parser import Qwen3CoderToolCallParser
    
    parser = Qwen3CoderToolCallParser()
    chunk1 = "<tool_call>"
    chunk2 = "<function=get_weather>"
    chunk3 = "</function></tool_call>"
    
    # Should return None for partial chunks
    assert parser.parse_streaming(chunk1) is None
    assert parser.parse_streaming(chunk2) is None
    
    # Should return complete tool call for final chunk
    result = parser.parse_streaming(chunk3)
    assert result is not None
```

#### Task 1.2: Create API Boundary Tests for Conversation Protocol
**Goal**: Write failing tests that define the public API for conversation protocol extensions

```python
def test_conversation_has_tool_parser_field():
    """Public API: Conversation should have tool_parser field."""
    from mlc_llm.protocol.conversation_protocol import Conversation
    
    conv = Conversation(
        roles={"user": "user", "assistant": "assistant"},
        role_templates={"user": "{user_message}", "assistant": "{assistant_message}"},
        seps=["\n"],
        tool_parser="qwen3_coder"
    )
    
    assert hasattr(conv, 'tool_parser')
    assert conv.tool_parser == "qwen3_coder"

def test_conversation_hydrates_parser():
    """Public API: Conversation should hydrate parser instance."""
    from mlc_llm.protocol.conversation_protocol import Conversation
    
    conv = Conversation(
        roles={"user": "user", "assistant": "assistant"},
        role_templates={"user": "{user_message}", "assistant": "{assistant_message}"},
        seps=["\n"],
        tool_parser="qwen3_coder"
    )
    
    assert hasattr(conv, 'tool_parser_instance')
    assert conv.tool_parser_instance is not None
```

### Phase 2: Subagent-Driven Implementation

#### Task 2.1: Implement Qwen3 XML Parser (Worktree: qwen3-parser)
**Goal**: Implement the Qwen3CoderToolCallParser class to pass API boundary tests

**Subagent Context**:
```
TASK:
- Implement mlc_llm.serve.tool_parser.Qwen3CoderToolCallParser
- Must pass all API boundary tests in tests/test_qwen3_tool_parser.py
- Follow TDD: Write minimal code to make tests pass
- No mocks, test real behavior
- Commit when all tests pass

CONTEXT:
- XML format: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
- Must handle newlines and malformed XML
- Streaming support required
- Use existing PARSER_REGISTRY for registration
```

#### Task 2.2: Extend Conversation Protocol (Worktree: conv-protocol)
**Goal**: Add tool_parser field and hydration to Conversation class

**Subagent Context**:
```
TASK:
- Extend mlc_llm.protocol.conversation_protocol.Conversation
- Add tool_parser field (string reference)
- Add tool_parser_instance property (hydrated instance)
- Implement hydration in model_validator
- Must pass all API boundary tests
- Commit when all tests pass

CONTEXT:
- Use existing get_parser_instance() from tool_parser module
- Exclude tool_parser_instance from serialization
- Maintain backward compatibility
```

#### Task 2.3: Update Qwen3 Template (Worktree: qwen3-template)
**Goal**: Add tools support to qwen3_5 conversation template

**Subagent Context**:
```
TASK:
- Update python/mlc_llm/conversation_template/qwen3_5.py
- Add tools placeholder to system template
- Add function_string for XML tool calls
- Must include available tools in prompt
- Commit when tests pass

CONTEXT:
- System template format: <|im_start|>system\n{tools}\n<|im_end|>
- Function string format: <tool_call><function={function_name}>...</function></tool_call>
- Keep existing thinking blocks intact
```

### Phase 3: Integration Testing

#### Task 3.1: Request Isolation Tests (Worktree: isolation-tests)
**Goal**: Write and pass tests for per-request parser isolation

```python
def test_parser_isolation_per_request():
    """Parser instances should be isolated per request."""
    from mlc_llm.protocol.conversation_protocol import Conversation
    
    # Create base conversation with parser
    conv1 = Conversation(
        roles={"user": "user", "assistant": "assistant"},
        role_templates={"user": "{user_message}", "assistant": "{assistant_message}"},
        seps=["\n"],
        tool_parser="qwen3_coder"
    )
    
    # Clone for request 1
    req1_conv = conv1.clone()
    parser1 = req1_conv.tool_parser_instance
    
    # Clone for request 2
    req2_conv = conv1.clone()
    parser2 = req2_conv.tool_parser_instance
    
    # Should be different instances
    assert parser1 is not parser2
    assert type(parser1) == type(parser2)
```

#### Task 3.2: End-to-End Tests (Worktree: e2e-tests)
**Goal**: Write and pass end-to-end integration tests

```python
def test_qwen3_tool_call_workflow():
    """End-to-end workflow with Qwen3 tool calls."""
    from mlc_llm.protocol.conversation_protocol import Conversation
    
    # Create conversation with tools
    conv = Conversation(
        roles={"user": "user", "assistant": "assistant"},
        role_templates={"user": "{user_message}", "assistant": "{assistant_message}"},
        seps=["\n"],
        tool_parser="qwen3_coder",
        tools=[{"name": "get_weather", "parameters": {"location": "str"}}]
    )
    
    # Serialize
    json_data = conv.to_json_dict()
    assert "tool_parser" in json_data
    assert "tool_parser_instance" not in json_data
    
    # Deserialize and hydrate
    restored_conv = Conversation.from_json_dict(json_data)
    assert restored_conv.tool_parser == "qwen3_coder"
    assert restored_conv.tool_parser_instance is not None
```

### Phase 4: Review Process

#### Task 4.1: Spec Compliance Review (Worktree: spec-review)
**Goal**: Verify all implementations match original specifications

**Review Checklist**:
- [ ] XML format with/without newlines handled
- [ ] Tools in system prompt correctly rendered
- [ ] Tool calls and results properly formatted
- [ ] Parser is general concept (not Qwen3-specific)
- [ ] Only qwen3_5.py knows about specific parser
- [ ] No parser = normal behavior maintained
- [ ] Per-request isolation implemented

#### Task 4.2: Code Quality Review (Worktree: quality-review)
**Goal**: Ensure code quality standards are met

**Review Checklist**:
- [ ] Follows project conventions and style
- [ ] Proper error handling
- [ ] Clear variable/function names
- [ ] Adequate test coverage
- [ ] No obvious bugs or missed edge cases
- [ ] No security issues

## Worktree Strategy

### Worktree Setup

```bash
# Create worktrees for parallel development
git worktree add /tmp/qwen3-parser qwen3-parser-branch
git worktree add /tmp/conv-protocol conv-protocol-branch
git worktree add /tmp/qwen3-template qwen3-template-branch
git worktree add /tmp/isolation-tests isolation-tests-branch
git worktree add /tmp/e2e-tests e2e-tests-branch
```

### Worktree Cleanup

```bash
# After each task completes, clean up worktree
git worktree remove -f /tmp/qwen3-parser
git worktree prune
```

## TDD Cycle Per Task

For EACH implementation task:

1. **RED**: Write failing API boundary test
2. **GREEN**: Implement minimal code to pass test
3. **REFACTOR**: Produce clean, maintainable code
   - Apply design patterns appropriately (Strategy, Factory, etc.)
   - Remove duplication and code smells
   - Improve structure without changing behavior
   - Extract methods for clarity
   - Ensure proper error handling
4. **COMMIT**: `git add -A && git commit -m "feat: [description]"`
5. **REVIEW**: Spec compliance + Code quality
6. **FIX**: Any issues found in review
7. **MERGE**: Into main branch when approved

## Success Criteria

1. ✅ All API boundary tests pass
2. ✅ No mocks for dependencies we control
3. ✅ Each test verifies one behavior
4. ✅ Test names are descriptive
5. ✅ Watched each test fail before implementing
6. ✅ All tests pass consistently
7. ✅ Refactoring enabled (can change implementation safely)
8. ✅ Spec compliance verified
9. ✅ Code quality approved
10. ✅ Per-request isolation working

## Risk Mitigation

### Potential Issues

1. **XML Parsing Complexity**
   - Mitigation: Extensive unit tests for edge cases
   - Test with real model outputs

2. **Request Isolation Leaks**
   - Mitigation: Comprehensive integration tests
   - Test multiple concurrent requests

3. **Backward Compatibility Breaks**
   - Mitigation: Regression tests for existing functionality
   - Feature flags if needed

### Contingency Plan

If implementation proves too complex:
1. Implement simpler version first (no streaming)
2. Add streaming as separate feature flag
3. Document limitations clearly
4. Provide clear upgrade path

## Files to Modify

### New Test Files
- `tests/test_qwen3_tool_parser_api.py` - API boundary tests for parser
- `tests/test_conversation_protocol_api.py` - API boundary tests for protocol
- `tests/test_qwen3_integration_api.py` - API boundary tests for integration
- `tests/test_parser_isolation_api.py` - API boundary tests for isolation

### Existing Files to Modify
- `python/mlc_llm/serve/tool_parser.py` - Implement Qwen3CoderToolCallParser
- `python/mlc_llm/protocol/conversation_protocol.py` - Add tool_parser fields
- `python/mlc_llm/conversation_template/qwen3_5.py` - Add tools support

## Verification Steps

After implementation, verify:

```bash
# Run API boundary tests
pytest tests/test_qwen3_tool_parser_api.py -v
pytest tests/test_conversation_protocol_api.py -v
pytest tests/test_qwen3_integration_api.py -v
pytest tests/test_parser_isolation_api.py -v

# Verify no regressions
pytest tests/ -q --tb=short

# Check code quality
pylint python/mlc_llm/serve/tool_parser.py
pylint python/mlc_llm/protocol/conversation_protocol.py
```

All tests should pass with:
- No memory leaks from buffer management
- No shared state between requests
- Backward compatibility maintained
- XML format handling (with and without newlines)

## Implementation Timeline

**Estimated Duration**: 7-10 days with parallel worktrees

### Phase 1: Test Setup (2 days)
- Task 1.1: API boundary tests for tool parser
- Task 1.2: API boundary tests for conversation protocol

### Phase 2: Implementation (5 days, parallel)
- Task 2.1: Qwen3 XML Parser (worktree: qwen3-parser)
- Task 2.2: Conversation Protocol Extension (worktree: conv-protocol)
- Task 2.3: Update Qwen3 Template (worktree: qwen3-template)

### Phase 3: Integration Testing (2 days, parallel)
- Task 3.1: Request Isolation Tests (worktree: isolation-tests)
- Task 3.2: End-to-End Tests (worktree: e2e-tests)

### Phase 4: Review (1 day)
- Task 4.1: Spec Compliance Review (worktree: spec-review)
- Task 4.2: Code Quality Review (worktree: quality-review)

## Key Principles

```
Fresh subagent per task
Two-stage review every time
Spec compliance FIRST
Code quality SECOND
Never skip reviews
Catch issues early
Test API boundaries, not internals
No mocks for dependencies we control
```

**Quality is not an accident. It's the result of systematic process.**