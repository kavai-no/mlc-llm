# Startup Summary - Everything You Need to Begin Immediately

## 🎯 Your Mission (Should You Choose to Accept It)

Implement the Qwen3 tool parser integration following TDD principles. All tests are already written - your job is to make them pass.

### Key Facts:
- ✅ **52 tests** already written (45 passing, 7 failing)
- ⏳ **Implementation phase** just beginning
- 📝 **TDD approach**: Test public API boundaries, no mocks for dependencies we control
- 🔄 **Refactoring is CRITICAL**: Not optional cleanup - essential for producing excellent code
- 🚀 **Git worktree isolation**: Each task gets its own isolated workspace

## 🚀 Immediate Action Plan (Do This NOW)

### Step 1: Activate Virtual Environment (MANDATORY)
```bash
source .venv/bin/activate
# Verify it worked
echo $VIRTUAL_ENV  # Should show path to .venv
python --version   # Should show Python 3.13.7
```

**NEVER run Python commands without activating the virtual environment first!**

### Step 2: Run Tests Correctly (MANDATORY)
```bash
./run_tests_correctly.sh
```

This script:
- Shows RED phase (failing tests - EXPECTED and CORRECT)
- Demonstrates proper TDD workflow
- Reminds you of the complete process

**NEVER run tests without using this script or activating virtual environment!**

### Step 3: Pick Your Task

Choose ONE task based on your skills:

#### Task 2.1: Implement Qwen3 XML Parser (Most Important)
- **File**: `python/mlc_llm/serve/tool_parser.py`
- **Tests**: `tests/test_qwen3_tool_parser_api.py` (34 tests, 28 passing, 6 failing)
- **Goal**: Make all tests pass
- **Worktree**: `/tmp/qwen3-parser`

#### Task 2.2: Extend Conversation Protocol
- **File**: `python/mlc_llm/protocol/conversation_protocol.py`
- **Tests**: `tests/test_conversation_protocol_api.py` (18 tests, 17 passing, 1 failing)
- **Goal**: Add tool_parser field and hydration pattern
- **Worktree**: `/tmp/conv-protocol`

#### Task 2.3: Update Qwen3 Template
- **File**: `python/mlc_llm/conversation_template/qwen3_5.py`
- **Tests**: Already integrated in protocol tests
- **Goal**: Add tools support to system prompt
- **Worktree**: `/tmp/qwen3-template`

### Step 4: Create Git Worktree (Isolation)
```bash
# Replace "task-name" and "branch-name" with your task
git worktree add /tmp/task-name branch-name
cd /tmp/task-name
```

This ensures no conflicts between parallel development efforts.

## 📋 TDD Cycle (Do This Exactly - No Variations)

### 1. RED Phase: Watch Tests Fail (EXPECTED!)
```bash
# Run specific test class to see failures
source .venv/bin/activate && python -m pytest tests/test_qwen3_tool_parser_api.py::TestQwen3ToolParserXMLParsing -xvs
```

**This is EXPECTED and CORRECT**. Failing tests define the requirements.

### 2. GREEN Phase: Implement Minimal Code
```bash
# Edit the file
nano python/mlc_llm/serve/tool_parser.py

# Add JUST ENOUGH code to make ONE test pass
class Qwen3CoderToolCallParser:
    def parse(self, text):
        # Minimal implementation - just enough to pass
        return ("tool_call", [])
```

### 3. REFACTOR Phase: Make Code Excellent
```bash
# Now improve the code structure
# - Extract methods for clarity
# - Remove duplication
# - Apply design patterns (Strategy, Factory, etc.)
# - Fix code smells
# - Improve error handling

class Qwen3CoderToolCallParser:
    def parse(self, text):
        """Parse complete tool call output."""
        result = self._extract_tool_calls(text)
        return self._format_results(result)
    
    def _extract_tool_calls(self, text):
        # Proper implementation here
        pass
```

### 4. COMMIT Phase: Document Changes
```bash
# Commit with descriptive message following conventions
git add -A
git commit -m "feat(tool_parser): implement Qwen3CoderToolCallParser with XML support"
```

## 🎯 Critical Requirements (Must Follow)

### 1. Parallel Tool Calls Support
- **Default behavior**: `allow_parallel_tool_calls=False` (stop after first tool call)
- **Enabled behavior**: `allow_parallel_tool_calls=True` (allow multiple tool calls)
- **Implementation**: Add parameter to parser constructor and update streaming logic

### 2. Request Isolation
- **CRITICAL**: Tool parser instance MUST be isolated per request
- Each request gets its own fresh parser instance
- Never share parser instances between requests

### 3. Backward Compatibility
- If no parser is set, normal behavior is expected
- Existing functionality must continue to work

## 📚 Essential Documentation (Read These First)

1. **QUICK_START_GUIDE.md** - Complete getting started guide (read this first!)
2. **QUICK_REFERENCE.md** - Cheat sheet with essential commands
3. **TDD_IMPLEMENTATION_PLAN.md** - Primary TDD approach and workflow
4. **REFACTORING_GUIDE.md** - Proper refactoring techniques (CRITICAL!)
5. **TEST_ANALYSIS_SUMMARY.md** - Test failure analysis and root causes
6. **UPDATED_REQUIREMENTS_SUMMARY.md** - Latest requirements including parallel_tool_calls

## 🔧 Essential Tools (Use These)

### 1. Correct Test Runner (USE THIS!)
```bash
./run_tests_correctly.sh
```

This script handles all the proper setup and shows expected failures.

### 2. Git Worktree Setup
```bash
# Create worktree for your task
git worktree add /tmp/task-name branch-name
cd /tmp/task-name

# Do your work in isolation
# When done, clean up
cd /workspace/projects/mlc-llm
git worktree remove -f /tmp/task-name
```

### 3. Virtual Environment Activation (ALWAYS FIRST!)
```bash
source .venv/bin/activate
# Verify activation worked
echo $VIRTUAL_ENV  # Should show path to .venv
python --version   # Should show Python 3.13.7
```

## ⚠️ Common Pitfalls (Avoid These)

### ❌ Running Tests Without Virtual Environment
**Wrong**:
```bash
python -m pytest tests/test_file.py  # FAILS!
```

**Right**:
```bash
source .venv/bin/activate && python -m pytest tests/test_file.py  # WORKS!
```

### ❌ Not Using the Test Runner Script
**Wrong**:
```bash
python -m pytest tests/test_file.py  # May not work correctly
```

**Right**:
```bash
./run_tests_correctly.sh  # Handles everything properly
```

### ❌ Changing Test Behavior
**Wrong**:
```bash
# Modifying tests to match implementation
nano tests/test_file.py  # DON'T DO THIS!
```

**Right**:
```bash
# Make implementation match test requirements
nano python/mlc_llm/module.py  # CORRECT!
```

### ❌ Mocking Dependencies We Control
**Wrong**:
```python
# Using mocks for dependencies we control
from unittest.mock import MagicMock
mock_parser = MagicMock()  # DON'T DO THIS!
```

**Right**:
```python
# Test real behavior, not mocks
parser = Qwen3CoderToolCallParser()  # CORRECT!
result = parser.parse(xml_input)
assert result == expected_output  # Test actual implementation
```

## 💡 Pro Tips (Save Time)

### 1. Test Incrementally
```bash
# Start with module export tests (easiest to pass)
source .venv/bin/activate && python -m pytest tests/test_qwen3_tool_parser_api.py::TestQwen3ToolParserModuleExports -xvs

# Then move to XML parsing tests
source .venv/bin/activate && python -m pytest tests/test_qwen3_tool_parser_api.py::TestQwen3ToolParserXMLParsing -xvs

# Finally, streaming tests (most complex)
source .venv/bin/activate && python -m pytest tests/test_qwen3_tool_parser_api.py::TestQwen3ToolParserStreaming -xvs
```

### 2. Check Test Collection First
```bash
source .venv/bin/activate && python -m pytest tests/test_qwen3_tool_parser_api.py --collect-only
```

This shows you what tests exist before running them.

### 3. Verify No Regressions
```bash
# Run all existing tests to ensure nothing broke
source .venv/bin/activate && python -m pytest tests/ -q --tb=short
```

## 📞 Support Resources

### Documentation
- `README.md` - Project overview
- `TDD_IMPLEMENTATION_PLAN.md` - Primary implementation plan
- `QUICK_START_GUIDE.md` - Complete getting started guide
- `REFACTORING_GUIDE.md` - Proper refactoring techniques

### Quick Reference
```bash
# Essential commands
less QUICK_REFERENCE.md

# Test runner script
./run_tests_correctly.sh

# Git worktree setup
./setup_worktrees.sh
```

## ✅ Success Checklist (Before Committing)

- [ ] Virtual environment activated (source .venv/bin/activate)
- [ ] Using run_tests_correctly.sh script for testing
- [ ] Tests fail in RED phase (EXPECTED and CORRECT)
- [ ] Minimal implementation added in GREEN phase
- [ ] Proper refactoring applied (no duplication, proper patterns)
- [ ] Error handling is robust
- [ ] Commit message follows format: `type(scope): description`
- [ ] No regressions introduced (all existing tests still pass)

## 🎉 You're Ready to Begin!

Follow this exact sequence:
1. ✅ Read QUICK_START_GUIDE.md (you're doing it!)
2. ⏳ Activate virtual environment: `source .venv/bin/activate`
3. ⏳ Run tests correctly: `./run_tests_correctly.sh`
4. 📝 Pick your task (2.1, 2.2, or 2.3)
5. 🔄 Create git worktree for isolation
6. 🔴 Watch tests fail in RED phase (EXPECTED!)
7. 🟢 Implement minimal code to pass tests in GREEN phase
8. 🔄 Refactor properly to produce excellent code
9. 📝 Commit with descriptive message following conventions
10. 🎯 Move to next task or request review

**Remember**: Failing tests in the RED phase are EXPECTED and CORRECT!
They define the requirements that your implementation must satisfy.
