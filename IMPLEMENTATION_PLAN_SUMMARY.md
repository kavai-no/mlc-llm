# Qwen3 Tool Parser Implementation - Plan Summary

## Overview
This document summarizes the implementation plans for integrating Qwen3 tool parser into mlc-llm.

## Available Implementation Plans

### 1. TDD Implementation Plan (Primary)
**File**: `TDD_IMPLEMENTATION_PLAN.md`

**Approach**: Test-Driven Development with clear API boundaries and subagent-driven development using git worktree isolation.

**Key Features**:
- ✅ Two-stage review process (spec compliance first, code quality second)
- ✅ Git worktree per task to prevent conflicts in parallel development
- ✅ Must use existing virtual environment at `.venv`
- ✅ Follow Boy Scout Rule: leave campsite cleaner than found

**Phases**:
1. **Phase 1**: Test-Driven Development Setup (Tasks 1.1-1.2)
   - Create API boundary tests for tool parser and conversation protocol
2. **Phase 2**: Subagent-Driven Implementation (Tasks 2.1-2.3)
   - Implement Qwen3 XML Parser
   - Extend Conversation Protocol
   - Update Qwen3 Template
3. **Phase 3**: Integration Testing (Tasks 3.1-3.2)
   - Request isolation tests
   - End-to-end integration tests
4. **Phase 4**: Review Process (Tasks 4.1-4.2)
   - Spec compliance review
   - Code quality review

**Success Criteria**:
- All API boundary tests pass
- No mocks for dependencies we control
- Each test verifies one behavior
- Test names are descriptive
- Watched each test fail before implementing
- All tests pass consistently
- Refactoring enabled (can change implementation safely)
- Spec compliance verified
- Code quality approved
- Per-request isolation working

### 2. Qwen3 Tool Parser Implementation Plan v2
**File**: `QWEN3_TOOL_PARSER_IMPLEMENTATION_PLAN_V2.md`

**Approach**: Detailed architectural plan with critical requirements.

**Critical Requirements**:
1. ✅ XML format has newlines (but parser should handle without as well)
2. ✅ Tools part of system prompt - must include available tools correctly rendered
3. ✅ Tool calls and tool call results - must be correctly rendered in conversation
4. ✅ Tool parser placement - should be used at the right place, most sensible location
5. ✅ General concept - tool parser should be a general concept usable by other models
6. ✅ Specific knowledge isolation - only qwen3_5 conversation template should know about specific parser implementation
7. ✅ No parser behavior - if no parser is set then normal behavior is expected (to be refactored to default tool parser in later phase)
8. ⚠️ **CRITICAL**: Tool parser instance MUST be isolated per request

**Architecture Overview**:
- **Tool Parser Registry** - General concept, not Qwen3-specific
- **BaseToolParser Interface** - Abstract base class for all parsers
- **Qwen3CoderToolCallParser** - Implementation for Qwen3 XML format
- **Conversation Protocol Extension** - Add tool_parser field with hydration
- **Request Isolation Mechanism** - Per-request parser instances

**Implementation Strategy**:
1. **Phase 1**: Tool Parser Infrastructure (General Concept)
   - Create BaseToolParser interface
   - Implement Qwen3CoderToolCallParser
   - Create Parser Registry
2. **Phase 2**: Conversation Protocol Integration
   - Extend Conversation protocol with tool_parser field
   - Update Qwen3 template for tool calling
3. **Phase 3**: Request Isolation Mechanism
   - Implement per-request parser instantiation
   - Update conversation clone method
4. **Phase 4**: Testing and Validation
   - Unit tests for parser
   - Integration tests
   - Regression tests

## Current Status

### Test Files Created
1. ✅ `tests/test_qwen3_tool_parser_api.py` (34 tests, 28 passing, 6 failing)
2. ✅ `tests/test_conversation_protocol_api.py` (18 tests, 17 passing, 1 failing)

### Test Results Summary
- **Total Tests**: 52
- **Passing**: 45 (86.5%)
- **Failing**: 7 (13.5%)

**Failing Tests Analysis**:
1. ✝️ Value conversion issues (4 failures) - JSON strings and numbers being converted to objects/integers
2. ✝️ Streaming logic issues (2 failures) - partial chunks not merged, buffer preservation incomplete
3. ✝️ Serialization issue (1 failure) - null tool_parser field always included in JSON

### Analysis Documents Created
1. ✅ `TEST_ANALYSIS_SUMMARY.md` - Detailed analysis of test failures and root causes
2. ✅ `QWEN3_TOOL_PARSER_TEST_SUMMARY.md` - Summary of test results and next steps
3. ✅ `IMPLEMENTATION_PLAN_SUMMARY.md` - This document

## Next Steps (From TDD Plan)

### Immediate Actions
1. **Verify Test Expectations**: Are the tests correct? Should JSON strings remain as strings or be converted to dicts?
2. **Check Specifications**: Review Qwen3 API spec and OpenAI function calling spec for expected behavior
3. **Decide on Conversion Strategy**: Keep values as strings OR convert them properly
4. **Fix Implementation**: Update `_try_convert_value()` and streaming logic
5. **Run All Tests Again**: Verify fixes work correctly

### Implementation Tasks (Following TDD)
1. ✅ Write failing tests that define public API (COMPLETE)
2. ⏳ Implement minimal code to pass tests
3. 🔄 Refactor if needed
4. 📝 Commit with descriptive message
5. 🔍 Two-stage review (spec compliance + code quality)
6. 🔀 Merge into main branch

### Subagent Tasks (Using Git Worktree Isolation)
1. **Task 2.1**: Implement Qwen3 XML Parser (Worktree: qwen3-parser)
   - Must pass all API boundary tests in `tests/test_qwen3_tool_parser_api.py`
   - Follow TDD: Write minimal code to make tests pass
   - No mocks, test real behavior

2. **Task 2.2**: Extend Conversation Protocol (Worktree: conv-protocol)
   - Add tool_parser field and hydration to Conversation class
   - Must pass all API boundary tests in `tests/test_conversation_protocol_api.py`

3. **Task 2.3**: Update Qwen3 Template (Worktree: qwen3-template)
   - Add tools support to qwen3_5 conversation template
   - Must include available tools in prompt

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

## Files to Modify

### New Test Files (Already Created)
- ✅ `tests/test_qwen3_tool_parser_api.py` - API boundary tests for parser
- ✅ `tests/test_conversation_protocol_api.py` - API boundary tests for protocol

### Existing Files to Modify
1. ⏳ `python/mlc_llm/serve/tool_parser.py` - Implement Qwen3CoderToolCallParser
2. ⏳ `python/mlc_llm/protocol/conversation_protocol.py` - Add tool_parser fields
3. ⏳ `python/mlc_llm/conversation_template/qwen3_5.py` - Add tools support
4. ⏳ Engine file (TBD) - Request isolation logic

## Verification Steps

After implementation, verify:

```bash
# Run API boundary tests
pytest tests/test_qwen3_tool_parser_api.py -v
pytest tests/test_conversation_protocol_api.py -v

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
