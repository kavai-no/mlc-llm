import json
import re
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from mlc_llm.protocol.openai_api_protocol import (
    ChatToolCall,
    ChatFunctionCall,
)

T = TypeVar('T', bound='BaseToolParser')

class BaseToolParser(ABC):
    """Abstract base class for all tool parsas."""
    @abstractmethod
    def parse(self, text: str) -> tuple[str, List[ChatToolCall]]:
        pass
    
    @abstractmethod
    def parse_streaming(self, token: str) -> Any:
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

        # Find all <tool_call> blocks
        tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        matches = list(tool_call_regex.finditer(text))
        
        if not matches:
            return text, []

        # The content is everything before the first match
        first_match_start = matches[0].start()
        content = text[:first_match_start]
        
        tool_calls: List[ChatToolCall] = []
        for m in matches:
            block = m.group(1)
            # Find <function=name>...</function> inside the block
            func_match = re.search(r"<function=(.*?)>(.*?)</function>", block, re.DOTALL)
            if not func_match:
                continue
            
            func_name = func_match.group(1).strip()
            params_content = func_match.group(2)

            param_dict = {}
            # Extract parameters within the function block
            param_matches = re.findall(r"<parameter=(.*?)>(.*?)</parameter>", params_content, re.DOTALL)
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
        
        # 1. Look for complete <tool_call> blocks in the buffer.
        tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        matches = list(tool_call_regex.finditer(self._buffer))
        
        if not matches:
            return None

        # 2. Extract all complete tool calls from the buffer.
        extracted_tool_calls: List[ChatToolCall] = []
        last_end_idx = 0
        
        for m in matches:
            block = m.group(1)
            # Find <function=name>...</function> inside the block
            func_match = re.search(s=r"<function=(.*?)>(.*?)</function>", block, re.DOTALL)
            if not func_match:
                continue
            
            func_name = func_match.group(1).strip()
            params_content = func_match.group(2)

            param_dict = {}
            # Extract parameters within the function block
            param_matches = re.findall(r"<parameter=(.*?)>(.*?)</parameter>", params_content, re.DOTALL)
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
            extracted_tool_calls.append(tc)
            last_end_idx = m.end()

        # 3. The residue is everything after the last match's closing tag.
        self._buffer = self._buffer[last_end_idx:]
        
        if extracted_tool_calls:
            return {"type": "complete_tool_call", "data": extracted_tool_calls}

        return None
