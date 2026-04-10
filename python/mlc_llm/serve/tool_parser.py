"""
Qwen3-Coder tool call parser.

Format uses XML-style nested tags:
    <tool_call>
    <function=function_name>
    <parameter=param_name>value</parameter>
    <parameter=param_name2>value2</parameter>
    </function>
    </tool_call>

Parameters are extracted from <parameter=name>value</parameter> tags and
type-converted using the schema if available, otherwise treated as strings.

Based on VLLM's Qwen3CoderToolParser.extract_tool_calls()
"""

import ast
import json
import re
import uuid
from typing import Any, Dict, List, Optional, Callable, TypeVar
from mlc_llm.protocol.openai_api_protocol import (
    ChatToolCall,
    ChatFunctionCall,
)
from mlc_llm.protocol.conversation_protocol import BaseToolParser

T = TypeVar('T')

# Global registry for parser lookup during hydration
PARSER_REGISTRY: Dict[str, Any] = {}

def register_parser(name: str) -> Callable[[T], T]:
    """Decorator to register a parser with the enough name."""
    def decorator(cls: T) -> T:
        PARSER_REGISTRY[name] = cls
        return cls
    return decorator

def get_parser_instance(name: str) -> Any:
    """Retrieve a parser instance from the registry."""
    parser_cls = PARSER_REGISTRY.get(name)
    if parser_cls:
        try:
            return parser_cls()
        except Exception:
            return None
    return None

def _try_convert_value(value: str) -> Any:
    """
    Try to convert a parameter value string to a native Python type.
    Handles null, numbers, booleans, JSON objects/arrays, and falls back to string.
    """
    stripped = value.strip()

    # Handle null
    if stripped.lower() == "null":
        return None

    # Try JSON first (handles objects, arrays, strings, numbers, booleans)
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try Python literal eval (handles tuples, etc.)
    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError, TypeError):
        pass

    # Return as string
    return stripped


# Define ParseResult type
ParseResult = tuple[str, List[ChatToolCall]]  # (content: str, tool_calls: List[ChatToolCall])

@register_parser("qwen3_coder")
class Qwen3CoderToolCallParser(BaseToolParser):
    """
    Parser for Qwen3-Coder XML-format tool calls.

    Uses nested XML tags: <tool_call><function=name><parameter=key>val</parameter></function></tool_call>
    """

    START_TOKEN="***"
    FUNCTION_PREFIX = "<function="

    # Find complete tool_call blocks (or unclosed at end)
    TOOL_CALL_REGEX = re.compile(
        r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
    )

    # Find function blocks within a tool_call
    FUNCTION_REGEX = re.compile(
        r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
    )

    # Find parameter blocks within a function
    PARAMETER_REGEX = re.compile(
        r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
        re.DOTALL,
    )

    def __init__(self):
        super().__init__()
        self._streaming_buffer = ""

    def _parse_function_call(self, function_str: str) -> Optional[ChatToolCall]:
        """Parse a single <function=name>...</function> block into a ChatToolCall."""
        try:
            # Extract function name: everything before the first '>'
            gt_idx = function_str.index(">")
            func_name = function_str[:gt_idx].strip()
            params_str = function_str[gt_idx + 
                                        1:]

            # Check if parameters are missing or malformed
            if not self.PARAMETER_REGEX.search(params_str):
                return None

            # Extract parameters
            param_dict: Dict[str, Any] = {}
            for match_text in self.PARAMETER_REGEX.findall(params_str):
                if ">" not in match_text:
                    continue
                eq_idx = match_text.index(">")
                param_name = match_text[:eq_idx].strip()
                param_value = match_text[eq_idx + 1:]

                # Clean up whitespace
                if param_value.startswith("\n"):
                    param_value = param_value[1:]
                if param_value.endswith("\n"):
                    param_value = param_value[:-1]

                param_dict[param_name] = _try_convert_value(param_value)

            return ChatToolCall(
                id=f"call_{uuid.uuid4().hex[:24]}",
                type="function",
                function=ChatFunctionCall(
                    name=func_name,
                    arguments=param_dict,  # Pass the dictionary directly
                ),
            )
        except (ValueError, IndexError):
            return None

    def parse(self, text: str) -> ParseResult:
        if not text.strip():
            return "", []

        # Find where the tool call starts (either <tool_call> or <function=)
        first_tc_idx = text.find("<tool_call>")
        if first_tc_idx < 0:
            first_tc_idx = text.find(self.FUNCTION_PREFIX)
            
        # If no tool call markers found, the entire text is content
        if first_tc_idx == -1:
            return text, []

        # Content is everything BEFORE the actual tool call start
        content = text[:first_tc_idx].strip() if first_tc_idx > 0 else ""

        try:
            # Find all tool_call blocks (including unclosed tags)
            tc_matches = self.TOOL_CALL_REGEX.findall(text)
            raw_blocks = [m[0] or m[1] for m in tc_matches if m[0] or m[1]]

            if not raw_blocks:
                return content, []

            # Find function blocks within each tool_call (including unclosed tags)
            function_strs: List[str] = []
            for block in raw_blocks:
                func_matches = self.FUNCTION_REGEX.findall(block)
                function_strs.extend(m[0] or m[1] for m in func_matches if m[0] or m[1])

            if not function_strs:
                return content, []

            # Parse each function call
            tool_calls: List[ChatToolCall] = []
            for func_str in function_strs:
                tc = self._parse_function_call(func_str)
                if tc is not None:
                    tool_calls.append(tc)

            return content, tool_calls

        except Exception:
            return content, []

    def parse_streaming(self, chunk: str) -> Any:
        """
        Method for incremental parsing of streaming tokens.
        
        Maintains internal buffer state to handle partial XML tags across chunks.
        Returns:
            - None if no tool call content detected
            - {"type": "partial_tool_call", "data": str} for incomplete tool calls
            - {"type": "complete_tool_call", "data": List[ChatToolCall]} for complete tool calls
        """
        if not chunk:
            return None
        
        # Add the new chunk to our buffer
        self._streaming_buffer += chunk
        
        # Check if we have any tool call content
        if "<tool_call>" not in self._streaming_buffer and self.FUNCTION_PREFIX not in self._streaming_buffer:
            return None
        
        # Try to find complete tool calls in the buffer
        tc_matches = self.TOOL_CALL_REGEX.findall(self._streaming_buffer)
        raw_blocks = [m[0] or m[1] for m in tc_matches if m[0] or m[1]]
        
        if not raw_blocks:
            # No complete tool calls yet, return partial
            return {"type": "partial_tool_call", "data": self._streaming_buffer}
        
        # We found at least one complete tool call block
        # Now we need to check if it's actually complete (has closing tags)
        function_strs: List[str] = []
        for block in raw_blocks:
            # Use findall to get capture groups, same as parse method
            func_matches = self.FUNCTION_REGEX.findall(block)
            function_strs.extend(m[0] or m[1] for m in func_matches if m[0] or m[1])
        
        if not function_strs:
            return {"type": "partial_tool_call", "data": self._streaming_buffer}
        
        # Check if we have complete tool calls (with closing tags)
        # A tool call is complete only if it has </tool_call>
        # We need to check the original blocks, not just extracted content
        complete_tool_calls: List[ChatToolCall] = []
        
        # Check if we have a complete tool call (with closing tag)
        has_complete_tool_call = "</tool_call>" in self._streaming_buffer and raw_blocks
        
        for func_str in function_strs:
            tc = self._parse_function_call(func_str)
            if tc is not None and has_complete_tool_call:
                complete_tool_calls.append(tc)
        
        # If we have complete tool calls with proper closing tags, return them
        if complete_tool_calls:
            # Clear the buffer since we've processed complete tool calls
            self._streaming_buffer = ""
            return {"type": "complete_tool_call", "data": complete_tool_calls}
        
        # Otherwise still partial (missing closing tags)
        return {"type": "partial_tool_call", "data": self._streaming_buffer}
