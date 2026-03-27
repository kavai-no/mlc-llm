---
description: MLC LLM testing patterns, environment setup, and best practices for writing unit tests
applyTo:
  - "**.py"
  - "**/tests/**/*.py"
---

# MLC LLM Testing Instructions

## Environment Setup

### Activating the Conda Environment
**REQUIRED BEFORE RUNNING TESTS OR CODE:**
```bash
micromamba activate mlc-chat-venv
```
All Python commands must be executed after activating this environment.

## Test Structure and Patterns

### Test Categorization (pytest marks)

Use `pytestmark` at the top of test files to categorize tests:

```python
import pytest

# For unit tests that don't require GPU or models
pytestmark = [pytest.mark.unittest]

# For operator correctness tests requiring GPU
pytestmark = [pytest.mark.op_correctness]

# For engine feature tests requiring both model and GPU
pytestmark = [pytest.mark.engine]
```

See `tests/python/conftest.py` for all available markers.

### Test File Patterns

**Location:** All tests should be in `tests/python/` directory
**Naming:** Use descriptive names with `test_` prefix: `test_module_feature.py`

#### Example: Simple Unit Test (no GPU required)
```python
"""Example of a simple unittest with no dependencies."""
import pytest
from mlc_llm.serve import PagedRadixTree

pytestmark = [pytest.mark.unittest]

def test_add():
    prt = PagedRadixTree()
    prt.add(0)
    assert list(prt.get(0)) == []
```

#### Example: Mock Testing Pattern (no actual model)
```python
"""Example using mocked tokenizer for testing without models."""
import pytest
from mlc_llm.serve.tool_parsers.qwen3coder import Qwen3CoderToolParser

pytestmark = [pytest.mark.unittest]

class MockTokenizer:
    """Minimal mock tokenizer for unit testing."""
    @property
    def vocab(self):
        return {"<tool_call>": 1001, "</tool_call>": 1002}
    
    def encode(self, text):
        res = self.vocab.get(text, None)
        return [res] if res is not None else [0]

@pytest.fixture
def parser():
    """Create a Qwen3CoderToolParser instance for testing."""
    return Qwen3CoderToolParser(MockTokenizer())

def test_parameterless_function(parser):
    xml_input = '<tool_call><function=get_current_timestamp></function></tool_call>'
    result = parser.extract_tool_calls(xml_input, None)
    
    assert result.tools_called
    assert len(result.tool_calls) == 1
```

### Test Fixtures

Use fixtures for:
- Mock dependencies (tokenizers, engines)
- Expensive setup/teardown operations
- Common test data

**Example:**
```python
import pytest
from mlc_llm.testing import require_test_model

@pytest.fixture
def small_model():
    """Fixture providing a small test model."""
    return require_test_model("Llama-3-8B-Instruct-q4f16_1-MLC")

def test_with_fixture(small_model):
    # Test using the fixture
    pass
```

### Test Organization

**Group related tests in classes:**
```python
class TestParameterlessToolCalls:
    """Test cases for tool calls without parameters."""
    
    def test_parameterless_function_no_tool_context(self, parser):
        """Test parameterless function when no tool context is provided."""
        # ...
    
    def test_parameterless_function_with_whitespace(self, parser):
        """Test parameterless function with whitespace in XML."""
        # ...
```

## Running Tests

### Basic Test Execution
```bash
# Run all tests in a directory
python -m pytest tests/python/serve/ -v

# Run specific test file
python -m pytest tests/python/test_qwen3_tool_parser_refactored.py -v

# Run tests by marker
python -m pytest tests/python/ -m unittest -v
```

### Test Discovery
- Pytest automatically discovers tests in `tests/` directory
- Tests are identified by:
  - Functions starting with `test_`
  - Classes starting with `Test` with methods starting with `test_`

## Best Practices

1. **Use fixtures for dependencies** - Don't create real resources in every test
2. **Keep tests isolated** - Each test should be independent
3. **Name tests clearly** - Describe what is being tested
4. **Use assertions liberally** - Verify expected behavior
5. **Mock external services** - Don't rely on network/APIs in unit tests
6. **Add docstrings** - Explain purpose of each test case
7. **Use pytest marks** - Categorize all tests appropriately

## Tool Parser Testing (Specific Patterns)

### Mock Tokenizer Pattern
```python
class MockTokenizer:
    """Mock tokenizer for testing without loading actual model."""
    
    @property
    def vocab(self):
        return {"<tool_call>": 1001, "</tool_call>": 1002}
    
    def encode(self, text):
        res = self.vocab.get(text, None)
        return [res] if res is not None else [0]
```

### XML Format Validation
- Empty arguments: `<tool_call><function=name></function></tool_call>`
- With parameters: `<tool_call><function=search><parameter=query>test</parameter></function></tool_call>`

### Testing Parameterless Functions
**Important:** Functions without required parameters should be accepted with empty dict:
```python
def test_parameterless_function():
    result = parser.extract_tool_calls(xml_input, None)
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args == {}, "Parameterless functions should have empty arguments"
```

## Common Testing Pitfalls

1. **Forgetting to activate conda environment** - Always run `micromamba activate mlc-chat-venv` first
2. **Testing without pytest marks** - Tests won't be categorized properly
3. **Hardcoding paths** - Use relative paths or environment variables
4. **Not mocking dependencies** - Can cause tests to fail in CI environments
5. **Slow integration tests in unit test suite** - Keep unittests fast (no GPU, no models)
6. **Not cleaning up resources** - Can cause test pollution between runs
