---
description: Tool parsing and rendering patterns for Qwen3Coder, including XML format specifications and testing instructions
applyTo:
  - "**.py"
  - "**/tool_parsers/**/*.py"
---

# Tool Parsing and Rendering

## Environment Setup

**REQUIRED BEFORE RUNNING TESTS OR CODE:**
```bash
micromamba activate mlc-chat-venv
```
All Python commands must be executed after activating this environment.

## XML Format for Qwen3Coder
- Tool parsers render tool calls as XML for Qwen3Coder models
- XML structure: `<tool_call><function=name>\n<parameter=paramName>value</parameter>\n</function>\n</tool_call>`
- Empty arguments: `<tool_call><function=name>\n</function>\n</tool_call>`
- DefaultToolParser uses JSON format:
```json
{
  "name": "function_name",
  "arguments": {...}
}
```

## Key Methods
- `render_tool_calls(tool_calls: List[ChatToolCall]) -> str` - Renders tool calls for conversation history
- `extract_tool_calls(model_output: str, request) -> ExtractedToolCallInformation` - Parses model output to extract tool calls
- Abstract base class methods must be implemented by all parsers

## Testing

### Running Tests
```bash
# Activate environment first
micromamba activate mlc-chat-venv

# Run specific test file
python -m pytest tests/python/test_qwen3_tool_parser_refactored.py -v

# Run all unittests
python -m pytest tests/python/ -m unittest -v
```

### Test Patterns
See `.github/instructions/testing.instructions.md` for comprehensive testing patterns including:
- Mock tokenizer pattern for unit testing
- XML format validation
- Testing parameterless functions
- Best practices for test organization

## Reference Files
- `tests/python/serve/reference_response_structure.txt` - Expected format for Qwen3Coder tool call responses
- `tests/python/serve/templates/qwen3coder.jinja` - Jinja template for rendering Qwen3 conversation prompts with tool support
- `vllm_reference.py` - Reference implementation from vLLM project showing expected behavior patterns

## Qwen3 Coder Tool Parser Implementation
### Key Files
- `python/mlc_llm/serve/tool_parsers/qwen3coder.py` - Main Qwen3Coder parser implementation
- `python/mlc_llm/conversation_template/qwen3.py` - Conversation template for Qwen3 models
- `python/mlc_llm/protocol/openai_api_protocol.py` - OpenAI API protocol definitions (ChatToolCall, ChatFunctionCall)

### Implementation Notes
1. **Tokenizer Requirements**: Qwen3Coder parser needs `<tool_call>` and `</tool_call>` tokens in tokenizer vocab
2. **XML Format**: Parser generates XML for tool calls, not JSON
3. **Streaming Support**: Enhanced streaming state tracking required for real-time responses
4. **Validation**: Use `test_validation_fix.py` to verify argument validation logic
5. **Mock Testing**: Use mocked tokenizer with vocab containing sentinel tokens when testing6. **Tool Choice Decision Logic**: The decision about whether to parse tools should happen OUTSIDE the parser. The parser itself should not check for `tool_choice="none"` - that logic belongs in the calling code. This keeps concerns separate and makes the parser a pure extraction/validation component.
7. Incomplete XML Handling: The parser correctly treats incomplete `<tool_call>` tags (without function definitions) as normal text content, not tool calls.
### Template Integration
- Jinja template handles placement of tool definitions and calls in conversation flow
- Must coordinate with `MessagePlaceholders.TOOL` and `MessagePlaceholders.FUNCTION`
- Tool call XML is wrapped in `<tool_call>...</tool_call>` tags

## Test Data Factory (NEW)
See `.github/instructions/TEST_DATA_FACTORY.md` for comprehensive documentation.

### Usage Guidelines
1. **Use constants** for standard test cases: `SINGLE_PARAM_XML`, `MULTI_PARAM_XML`, `EMPTY_PARAMS_XML`
2. **Use fluent builder** when creating custom or varied XML: `XmlToolCallBuilder().with_function(...)`
3. **Reuse tool factories** to maintain consistency: `create_search_web_tool()`
4. **Always use proper XML formatting** with newlines matching the actual implementation