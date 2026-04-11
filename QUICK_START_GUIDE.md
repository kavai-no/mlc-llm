# Quick Start Guide for Qwen3 Tool Parser Implementation

## ⚠️ READ THIS FIRST - Critical Setup Instructions

### 1. Virtual Environment Activation (MANDATORY)
```bash
# ALWAYS activate the virtual environment BEFORE running any Python commands
source .venv/bin/activate

# Verify activation worked
echo $VIRTUAL_ENV  # Should show path to .venv
python --version   # Should show Python 3.13.7
```

**NEVER run Python commands without activating the virtual environment first!**

### 2. Use the Correct Test Runner (MANDATORY)
```bash
# Use this script - it handles everything correctly
./run_tests_correctly.sh

# OR manually:
source .venv/bin/activate && python -m pytest tests/test_qwen3_tool_parser_api.py::TestQwen3ToolParserModuleExports -xvs
```

**NEVER run tests without the virtual environment activated!**

## 🚀 Getting Started in 5 Minutes

### Step 1: Clone the Repository (if not already done)
```bash
cd /workspace/projects
git clone https://github.com/mlc-ai/mlc-llm.git qwen3-tool-parser 2>/dev/null || echo "Already cloned"
cd qwen3-tool-parser
```

### Step 2: Check Current Status
```bash
# See what tests exist and their status
./run_tests_correctly.sh

# Or check specific test classes
source .venv/bin/activate && python -m pytest tests/test_qwen3_tool_parser_api.py --collect-only
source .venv/bin/activate && python -m pytest tests/test_conversation_protocol_api.py --collect-only
```

### Step 3: Understand the Implementation Plan
```bash
# Read the primary implementation plan
cat TDD_IMPLEMENTATION_PLAN.md | head -100

# Or read specific sections
less TDD_IMPLEMENTATION_PLAN.md
```

## 📋 Current Task Breakdown

### Task 2.1: Implement Qwen3 XML Parser (Worktree: qwen3-parser)
**Goal**: Make all tests in `tests/test_qwen3_tool_parser_api.py` pass

**Steps**:
1. ✅ Tests already written and running
2. ⏳ Implement minimal code to pass tests
3. 🔄 Refactor properly (remove duplication, apply patterns)
4. 📝 Commit: `git add -A && git commit -m "feat(tool_parser): implement Qwen3CoderToolCallParser"`
5. 🔍 Review: spec compliance + code quality

**Files to Modify**:
- `python/mlc_llm/serve/tool_parser.py` (main implementation)

### Task 2.2: Extend Conversation Protocol (Worktree: conv-protocol)
**Goal**: Make all tests in `tests/test_conversation_protocol_api.py` pass

**Steps**:
1. ✅ Tests already written and running
2. ⏳ Add tool_parser field to Conversation class
3. ⏳ Implement hydration pattern
4. 🔄 Refactor properly
5. 📝 Commit: `git add -A && git commit -m "feat(conversation): add tool_parser support"`

**Files to Modify**:
- `python/mlc_llm/protocol/conversation_protocol.py`

### Task 2.3: Update Qwen3 Template (Worktree: qwen3-template)
**Goal**: Add tools support to qwen3_5 conversation template

**Steps**:
1. ✅ Tests already written and running
2. ⏳ Add function_string for XML format
3. ⏳ Include available tools in system prompt
4. 🔄 Refactor properly
5. 📝 Commit: `git add -A && git commit -m "feat(qwen3): add tool calling support"`

**Files to Modify**:
- `python/mlc_llm/conversation_template/qwen3_5.py`

## ⚡ TDD Workflow (Do This Exactly)

### 1. RED Phase - Watch Tests Fail (Expected!)
```bash
# Run specific test class to see it fail
source .venv/bin/activate && python -m pytest tests/test_qwen3_tool_parser_api.py::TestQwen3ToolParserXMLParsing -xvs

# This is EXPECTED and CORRECT - failing tests define requirements
```

### 2. GREEN Phase - Implement Minimal Code
```bash
# Edit the file
nano python/mlc_llm/serve/tool_parser.py

# Add just enough code to make ONE test pass
class Qwen3CoderToolCallParser:
    def parse(self, text):
        # Minimal implementation
        return ("tool_call", [])
```

### 3. REFACTOR Phase - Make Code Excellent
```bash
# Now improve the code structure
# - Extract methods
# - Remove duplication
# - Apply patterns
# - Fix code smells

class Qwen3CoderToolCallParser:
    def parse(self, text):
        """Parse complete tool call output."""
        result = self._extract_tool_calls(text)
        return self._format_results(result)
    
    def _extract_tool_calls(self, text):
        # Implementation here
        pass
```

### 4. COMMIT Phase - Document Changes
```bash
# Commit with descriptive message following conventions
git add -A
git commit -m "feat(tool_parser): implement Qwen3CoderToolCallParser with XML support"
```

## 🔧 Common Issues and Fixes

### Issue 1: Module Not Found Error
```bash
# Problem: "ModuleNotFoundError: No module named mlc_llm"

# Solution: Add python to sys.path BEFORE importing
python -c "import sys; sys.path.insert(0, 'python'); from mlc_llm.serve.tool_parser import Qwen3CoderToolCallParser"
```

### Issue 2: Virtual Environment Not Activated
```bash
# Problem: Commands fail with "command not found" or wrong Python version

# Solution: ALWAYS activate virtual environment first
source .venv/bin/activate
# Then run your command
python -m pytest ...
```

### Issue 3: Tests Not Running Correctly
```bash
# Problem: Tests fail for unknown reasons

# Solution: Use the correct test runner script
./run_tests_correctly.sh

# This handles all the proper setup and shows expected failures
```

## 📝 Code Conventions

### Commit Message Format
```
type(scope): description

Examples:
- feat(tool_parser): implement Qwen3CoderToolCallParser
- fix(conversation): handle null tool_parser in serialization
- refactor(registry): extract parser factory pattern
- test: add streaming edge case tests
```

### File Structure Conventions
```
python/mlc_llm/
  serve/
    tool_parser.py      # Parser implementations and registry
  protocol/
    conversation_protocol.py  # Conversation class with tool_parser field
  conversation_template/
    qwen3_5.py          # Qwen3 templates (add tools support here)
```

### Import Conventions
```python
# In test files:
import sys
sys.path.insert(0, 'python')  # Add project root to path
from mlc_llm.serve.tool_parser import Qwen3CoderToolCallParser

# In implementation files:
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
```

## 🎯 Test Running Cheat Sheet

### Run All Tests
```bash
source .venv/bin/activate && python -m pytest tests/test_qwen3_tool_parser_api.py tests/test_conversation_protocol_api.py -v
```

### Run Specific Test Class
```bash
source .venv/bin/activate && python -m pytest tests/test_qwen3_tool_parser_api.py::TestQwen3ToolParserXMLParsing -xvs
```

### Run Single Test Method
```bash
source .venv/bin/activate && python -m pytest tests/test_qwen3_tool_parser_api.py::TestQwen3ToolParserModuleExports::test_qwen3_parser_module_exports -xvs
```

### Check Test Collection (See What Tests Exist)
```bash
source .venv/bin/activate && python -m pytest tests/test_qwen3_tool_parser_api.py --collect-only
```

## 📚 Documentation Overview

### Implementation Plans
1. `TDD_IMPLEMENTATION_PLAN.md` - Primary TDD approach with subagent workflows
2. `QWEN3_TOOL_PARSER_IMPLEMENTATION_PLAN_V2.md` - Detailed architecture and requirements
3. `QWEN3_TOOL_PARSER_IMPLEMENTATION_PLAN.md` - Original implementation plan

### Analysis Documents
1. `TEST_ANALYSIS_SUMMARY.md` - Test failure analysis and root causes
2. `QWEN3_TOOL_PARSER_TEST_SUMMARY.md` - Test results summary
3. `FINAL_PLAN_SUMMARY.md` - Comprehensive overview of all plans
4. `UPDATED_REQUIREMENTS_SUMMARY.md` - Latest requirements including parallel_tool_calls
5. `REFACTORING_GUIDE.md` - Proper refactoring techniques and examples

### Quick Reference Scripts
1. `run_tests_correctly.sh` - Correct TDD testing workflow (USE THIS!)
2. `setup_worktrees.sh` - Create isolated worktrees for parallel development
3. `run_tests_with_venv.sh` - Test runner using virtual environment

## 💡 Pro Tips

### 1. Use Git Worktrees for Isolation
```bash
# Create worktree for your task
git worktree add /tmp/qwen3-parser qwen3-parser-branch
cd /tmp/qwen3-parser

# Do your work in isolation
# When done, clean up
cd /workspace/projects/mlc-llm
git worktree remove -f /tmp/qwen3-parser
```

### 2. Test Incrementally
```bash
# Start with module export tests (easiest to pass)
source .venv/bin/activate && python -m pytest tests/test_qwen3_tool_parser_api.py::TestQwen3ToolParserModuleExports -xvs

# Then move to XML parsing tests
source .venv/bin/activate && python -m pytest tests/test_qwen3_tool_parser_api.py::TestQwen3ToolParserXMLParsing -xvs

# Finally, streaming tests (most complex)
source .venv/bin/activate && python -m pytest tests/test_qwen3_tool_parser_api.py::TestQwen3ToolParserStreaming -xvs
```

### 3. Verify No Regressions
```bash
# Run all existing tests to ensure nothing broke
source .venv/bin/activate && python -m pytest tests/ -q --tb=short
```

## 🚨 Emergency Recovery

### If Tests Hang or Crash
```bash
# Kill any hanging processes
pkill -f "python.*pytest"

# Restart with clean state
source .venv/bin/activate && python -m pytest tests/test_qwen3_tool_parser_api.py::TestQwen3ToolParserModuleExports -xvs
```

### If Virtual Environment Corrupts
```bash
# Recreate virtual environment (last resort)
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -e .  # Install project in development mode
```

## 📞 Support

### Common Questions
**Q: Do I need to create new tests?**
A: NO! All tests are already written. Just make them pass.

**Q: Should I mock dependencies?**
A: NO! Test real behavior, not mocks (per TDD principles).

**Q: What if a test seems wrong?**
A: The tests define the API requirements. If you think a test is wrong, discuss it first.

### Getting Help
```bash
# Check existing documentation
less README.md
less TDD_IMPLEMENTATION_PLAN.md

# Look at test files for examples
grep -A 10 "def test_" tests/test_qwen3_tool_parser_api.py | head -20
```

## ✅ Success Checklist

Before committing your work, verify:
- [ ] All relevant tests pass
- [ ] Code follows project conventions
- [ ] No duplication or code smells
- [ ] Proper design patterns applied
- [ ] Error handling is robust
- [ ] Commit message follows format: `type(scope): description`
- [ ] Virtual environment was activated for all testing

## 🎉 You're Ready to Start!

Follow these steps:
1. ✅ Read this guide (you're doing it!)
2. ⏳ Pick your task (2.1, 2.2, or 2.3)
3. 📝 Create git worktree for isolation
4. 🔴 Run tests to see them fail (RED phase)
5. 🟢 Implement minimal code to pass tests (GREEN phase)
6. 🔄 Refactor properly (remove duplication, apply patterns)
7. 📝 Commit with descriptive message
8. 🎯 Move to next task or review

**Remember**: Failing tests in the RED phase are EXPECTED and CORRECT!
