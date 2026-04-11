# Quick Reference Card

## 🚀 Essential Commands (Copy & Paste)

### Activate Virtual Environment (ALWAYS FIRST!)
```bash
source .venv/bin/activate
```

### Run Tests Correctly
```bash
./run_tests_correctly.sh
```

### Run Specific Test Class
```bash
source .venv/bin/activate && python -m pytest tests/test_qwen3_tool_parser_api.py::TestQwen3ToolParserXMLParsing -xvs
```

### Check Test Collection
```bash
source .venv/bin/activate && python -m pytest tests/test_qwen3_tool_parser_api.py --collect-only
```

## 📋 TDD Workflow (Do This Exactly)

```
1. RED: Watch tests fail (EXPECTED!)
   source .venv/bin/activate && python -m pytest tests/test_file.py::TestClass -xvs

2. GREEN: Implement minimal code
   nano python/mlc_llm/module.py
   # Add just enough to pass ONE test

3. REFACTOR: Make code excellent
   # Extract methods, remove duplication, apply patterns

4. COMMIT: Document changes
   git add -A && git commit -m "feat(scope): description"
```

## 🎯 Task Breakdown

### Task 2.1: Qwen3 XML Parser
- **File**: `python/mlc_llm/serve/tool_parser.py`
- **Tests**: `tests/test_qwen3_tool_parser_api.py` (34 tests)
- **Goal**: Make all tests pass

### Task 2.2: Conversation Protocol
- **File**: `python/mlc_llm/protocol/conversation_protocol.py`
- **Tests**: `tests/test_conversation_protocol_api.py` (18 tests)
- **Goal**: Add tool_parser field and hydration

### Task 2.3: Qwen3 Template
- **File**: `python/mlc_llm/conversation_template/qwen3_5.py`
- **Tests**: Already integrated in protocol tests
- **Goal**: Add tools support to system prompt

## ⚠️ Critical Requirements

1. ✅ XML format with newlines (but handle without)
2. ✅ Tools in system prompt correctly rendered
3. ✅ Tool calls and results properly formatted
4. ✅ Parser is general concept, not Qwen3-specific
5. ✅ Only qwen3_5.py knows about specific parser implementation
6. ✅ No parser = normal behavior (backward compatible)
7. ✅ **CRITICAL**: Tool parser instance MUST be isolated per request
8. ✅ **parallel_tool_calls parameter** - default: stop after first tool call

## 📝 Commit Message Format

```
type(scope): description

Examples:
feat(tool_parser): implement Qwen3CoderToolCallParser
fix(conversation): handle null tool_parser in serialization  
refactor(registry): extract parser factory pattern
```

## 🔧 File Structure

```
python/mlc_llm/
  serve/
    tool_parser.py      # Parser implementations and registry
  protocol/
    conversation_protocol.py  # Conversation class with tool_parser field
  conversation_template/
    qwen3_5.py          # Qwen3 templates (add tools support here)
```

## 📚 Documentation Quick Links

- `TDD_IMPLEMENTATION_PLAN.md` - Primary TDD approach
- `QWEN3_TOOL_PARSER_IMPLEMENTATION_PLAN_V2.md` - Detailed architecture
- `REFACTORING_GUIDE.md` - Proper refactoring techniques
- `QUICK_START_GUIDE.md` - Complete getting started guide
- `TEST_ANALYSIS_SUMMARY.md` - Test failure analysis

## 💡 Pro Tips

### 1. Git Worktree Isolation
```bash
git worktree add /tmp/task-name branch-name
cd /tmp/task-name
# Do work...
cd /workspace/projects/mlc-llm
git worktree remove -f /tmp/task-name
```

### 2. Test Incrementally
```bash
# Start with module exports (easiest)
source .venv/bin/activate && python -m pytest tests/test_qwen3_tool_parser_api.py::TestQwen3ToolParserModuleExports -xvs

# Then XML parsing
source .venv/bin/activate && python -m pytest tests/test_qwen3_tool_parser_api.py::TestQwen3ToolParserXMLParsing -xvs

# Finally streaming (most complex)
source .venv/bin/activate && python -m pytest tests/test_qwen3_tool_parser_api.py::TestQwen3ToolParserStreaming -xvs
```

### 3. Verify No Regressions
```bash
source .venv/bin/activate && python -m pytest tests/ -q --tb=short
```

## 🚨 Common Issues

### Issue: Module Not Found
**Fix**: Add python to sys.path
```python
import sys
sys.path.insert(0, 'python')
from mlc_llm.serve.tool_parser import Qwen3CoderToolCallParser
```

### Issue: Virtual Environment Not Activated
**Fix**: ALWAYS activate first
```bash
source .venv/bin/activate
# Then run your command
```

### Issue: Tests Hang or Crash
**Fix**: Kill processes and restart
```bash
pkill -f "python.*pytest"
source .venv/bin/activate && python -m pytest tests/test_file.py::TestClass -xvs
```

## ✅ Success Checklist

- [ ] Virtual environment activated (source .venv/bin/activate)
- [ ] Using run_tests_correctly.sh script
- [ ] Tests fail in RED phase (EXPECTED!)
- [ ] Minimal implementation in GREEN phase
- [ ] Proper refactoring applied
- [ ] Commit message follows format
- [ ] No regressions introduced

## 🎉 Remember

✅ **Failing tests are EXPECTED and CORRECT** in the RED phase
✅ **Always activate virtual environment first**
✅ **Use run_tests_correctly.sh script** - it handles everything properly
✅ **Refactoring is CRITICAL** - not optional cleanup
✅ **Test API boundaries, not internal implementation**
