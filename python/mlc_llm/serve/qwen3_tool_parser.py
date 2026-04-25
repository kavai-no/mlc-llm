import json
import re
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar
from mlc_llm.protocol.openai_api_protocol import ChatToolCall, ChatFunctionCall

T = TypeVar('T', bound='BaseToolParser')

class BaseToolParser(ABC):
    @abstractmethod
    def parse(self, text: str) -> tuple[str, List[ChatToolCall]]:
        pass

    @abstractmethod
    def parse_streaming(self, token: str) -> Any:
        pass

    @abstractmethod
    def render_tool_call(self, tool_call: ChatToolCall) -> str:
        pass

    @abstractmethod
    def render_tool_result(self, tool_call_id: str, result: str) -> str:
        pass

# Global registry for parser lookup during hydration
PARSER_REGISTRY: Dict[str, Type[BaseToolParser]] = {}

def register_parser(name: str):
    """Decorator to register a parser class with the registry."""
    def decorator(cls: Type[T]) -> Type[T]:
        PARSER_REGISTRY[name] = cls
        return cls
    return decorator

def get_parser_instance(name: str) -> Optional[BaseToolParser]:
    """Retrieve a parser instance from the registry."""
    if not name or name not in PARSER_REGISTRY:
        return None
    return PARSER_REGISTRY[name]()

# --- Helper Functions ---

def _try_convert_value(value: str) -> Any:
    """Try to convert a parameter value string to a de-facto Python type."""
    stripped = value.strip()
    if not stripped: return ""
    if stripped.lower() == "null": return None

    # Try numeric conversion
    try:
        if "." in stripped or "e" in stripped.lower():
            return float(stripped)
        else:
            return int(stripped)
    except ValueError:
        pass

    if stripped.lower() == "true": return True
    if stripped.lower() == "false": return False

    return stripped

# --- Qwen3 Implementation ---

@register_parser("qwen3_coder")
class Qwen3CoderToolCallParser(BaseToolParser):
    """Parser for Qwen3-Coder XML-style tool calls."""

    def __init__(self):
        self._buffer = ""

    def parse(self, text: str) -> tuple[str, List[ChatToolCall]]:
        if not text.strip():
            return text, []

        # Qwen3 outputs tool calls in this format:
        # <function=name><parameter=key>value</parameter>...</function></tool_call>
        # Note: The opening <tool_call> is missing, only </tool_call> appears at the end.
        
        # Find all function blocks that end with </function></tool_call>
        func_regex = re.compile(
            r"<function=(.*?)>(.*?)</function>\s*</tool_call>",
            re.DOTALL
        )
        matches = list(func_regex.finditer(text))
        
        if not matches:
            return text, []

        # The content is everything before the first match
        first_match_start = matches[0].start()
        content = text[:first_match_start]
        
        tool_calls: List[ChatToolCall] = []
        for m in matches:
            func_name = m.group(1).strip()
            params_content = m.group(2)

            param_dict = {}
            # Extract parameters within the function block
            param_matches = re.findall(
                r"<parameter=(.*?)>(.*?)</parameter>", params_content, re.DOTALL
            )
            for p_name, p_val in param_matches:
                p_name = p_name.strip()
                p_val = p_val.strip()
                param_dict[p_name] = _try_convert_value(p_val)

            tc = ChatToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}",
                type="function",
                function=ChatFunctionCall(
                    name=func_name,
                    arguments=param_dict,
                ),
            )
            tool_calls.append(tc)

        return content, tool_calls

    def parse_streaming(self, token: str) -> Any:
        self._buffer += token
        
        # Qwen3 outputs tool calls in this format:
        # <function=name><parameter=key>value</parameter>...</function></tool_call>
        # Note: The opening <tool_call> is missing, only </tool_call> appears at the end.
        
        # Check for complete function blocks that end with </function></tool_call>
        func_regex = re.compile(
            r"<function=(.*?)>(.*?)</function>\s*</tool_call>",
            re.DOTALL
        )
        matches = list(func_regex.finditer(self._buffer))
        
        if matches:
            # We found complete function blocks!
            # The residue is everything before the first match.
            first_match_start = matches[0].start()
            residue = self._buffer[:first_match_start] if first_match_start > 0 else ""
            
            extracted_calls: List[ChatToolCall] = []
            last_end_idx = 0
            
            for m in matches:
                func_name = m.group(1).strip()
                params_content = m.group(2)
                
                param_dict = {}
                param_matches = re.findall(
                    r"<parameter=(.*?)>(.*?)</parameter>", params_content, re.DOTALL
                )
                for p_name, p_val in param_matches:
                    p_name = p_name.strip()
                    p_val = p_val.strip()
                    param_dict[p_name] = _try_convert_value(p_val)
                
                tc = ChatToolCall(
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    type="function",
                    function=ChatFunctionCall(
                        name=func_name,
                        arguments=param_dict,
                    ),
                )
                extracted_calls.append(tc)
                last_end_idx = m.end()
            
            # Update buffer: only keep what is after the last completed match.
            self._buffer = self._buffer[last_end_idx:]
            return residue, extracted_calls
        
        # Check for "unclosed" potential tags.
        # We look for the start of any tool call related tags.
        partial_tag_regex = re.compile(
            r"<function|<parameter|</parameter|</function|</tool_call"
        )
        partial_match = partial_tag_regex.search(self._buffer)
        
        if partial_match:
            # Found the start of a potential tag!
            # Return everything BEFORE this tag as residue.
            split_idx = partial_match.start()
            residue = self._buffer[:split_idx] if split_idx > 0 else ""
            
            # Keep the partial tag in the buffer.
            if split_idx > 0:
                self._buffer = self._buffer[split_idx:]
            
            return residue, []
        else:
            # No tool call tags found at all.
            # Flush everything as residue.
            residue = self._buffer
            self._buffer = ""
            return residue, []

    def _try_convert(self, val: str) -> Any:
        """Helper to convert string values from XML-like params."""
        try:
            if val.lower() == "true": return True
            if val.lower() == "false": return False
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return val

    def render_tool_call(self, tool_call: ChatToolCall) -> str:
        """Render a tool call into Qwen3 XML format."""
        func = tool_call.function
        params = func.arguments if isinstance(func.arguments, dict) else json.loads(func.arguments)
        
        param_str = ""
        for k, v in params.items():
            param_str += f"<parameter={k}>{v}</parameter>"
            
        return f"<tool_call><function={func.name}>{param_str}</function></tool_call>"

    def render_tool_result(self, tool_call_id: str, result: str) -> str:
        """Render a tool result into Qwen3 XML format."""
        # Note: Qwen3 results often just use <tool_response> or similar.
        # Based on common patterns for this model:
        return f"<tool_response>\n{result}\n</tool_response>"

