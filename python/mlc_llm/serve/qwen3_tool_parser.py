import json
import re
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from mlc_llm.protocol.openai_api_protocol import ChatToolCall, ChatFunctionCall

T = TypeVar('T', bound='BaseToolParser')


class BaseToolParser(ABC):
    """Base class for tool parsers."""

    @abstractmethod
    def parse(self, text: str) -> tuple[str, List[ChatToolCall]]:
        """Parse non-streaming text and return (content, tool_calls)."""
        pass

    @abstractmethod
    def parse_streaming(self, token: str) -> tuple[Optional[str], List[ChatToolCall]]:
        """Parse streaming token and return (content_delta, tool_calls)."""
        pass

    @abstractmethod
    def render_tool_call(self, tool_call: ChatToolCall) -> str:
        """Render a tool call to string format for the model."""
        pass

    @abstractmethod
    def render_tool_result(self, tool_call_id: str, result: str) -> str:
        """Render a tool result to string format for the model."""
        pass


# Global registry for parser lookup
PARSER_REGISTRY: Dict[str, Type[BaseToolParser]] = {}


def register_parser(name: str):
    """Decorator to register a parser class."""
    def decorator(cls: Type[T]) -> Type[T]:
        PARSER_REGISTRY[name] = cls
        return cls
    return decorator


def get_parser_instance(name: str) -> Optional[BaseToolParser]:
    """Get a parser instance by name."""
    if not name or name not in PARSER_REGISTRY:
        return None
    return PARSER_REGISTRY[name]()


def _try_convert_value(value: str) -> Any:
    """Try to convert a string value to appropriate type."""
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped.lower() == "null":
        return None
    try:
        if "." in stripped or "e" in stripped.lower():
            return float(stripped)
        else:
            return int(stripped)
    except ValueError:
        pass
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    return stripped


@register_parser("qwen3_coder")
class Qwen3CoderToolCallParser(BaseToolParser):
    """
    Robust parser for Qwen3-Coder XML-style tool calls.
    Follows the pattern used in major inference engines: 
    detecting <tool_call> blocks and parsing them with flexible regex.
    """

    def __init__(self):
        self._buffer = ""
        self._in_tool_call_block = False

    def parse(self, text: str) -> tuple[str, List[ChatToolCall]]:
        """Parse non-streaming complete text."""
        self._buffer = text
        self._in_tool_call_block = False
        
        tool_calls: List[ChatToolCall] = []
        content_parts = []
        last_end = 0

        # Pattern for the whole block. We use [\s\n]* to handle fragmented tags.
        pattern = re.compile(r"<tool_call>[\s\n]*(.*?)</tool_call>", re.DOTALL)
        for match in pattern.finditer(text):
            content_parts.append(text[last_end:match.start()])
            inner = match.group(1)
            
            # Pattern for function name and body, allowing whitespace/newlines inside tags
            func_pattern = re.compile(r"<function\s*=\s*(.*?)\s*>(.*?)</function>", re.DOTALL)
            for f_match in func_pattern.finditer(inner):
                f_name = f_match.group(1).strip()
                f_body = f_match.group(2)
                params = {}
                param_pattern = re.compile(r"<parameter\s*=\s*(.*?)\s*>(.*?)</parameter>", re.DOTALL)
                for p_match in param_pattern.finditer(f_body):
                    params[p_match.group(1).strip()] = _try_convert_value(p_match.group(2))
                
                tool_calls.append(ChatToolCall(
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    type="function",
                    function=ChatFunctionCall(name=f_name, arguments=params)
                ))
            last_end = match.end()
        
        content_parts.append(text[last_end:])
        return "".join(content_parts), tool_calls

    def parse_streaming(self, token: str) -> tuple[Optional[str], List[ChatToolCall]]:
        """Parse streaming token with state awareness."""
        self._buffer += token
        tool_calls = []
        content_to_send = None

        # 1. Check if we have a complete block in the buffer
        pattern = re.compile(r"<tool_call>[\s\n]*(.*?)</tool_call>", re.DOTALL)
        match = pattern.search(self._buffer)
        if match:
            content_to_send = self._buffer[:match.start()]
            inner = match.group(1)
            
            func_pattern = re.compile(r"<function\s*=\s*(.*?)\s*>(.*?)</function>", re.DOTALL)
            for f_match in func_pattern.finditer(inner):
                f_name = f_match.group(1).strip()
                f_body = f_match.group(2)
                params = {}
                param_pattern = re.compile(r"<parameter\s*=\s*(.*?)\s*>(.*?)</parameter>", re.DOTALL)
                for p_match in param_pattern.finditer(f_body):
                    params[p_match.group(1).strip()] = _try_convert_value(p_match.group(2))
                
                tool_calls.append(ChatToolCall(
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    type="function",
                    function=ChatFunctionCall(name=f_name, arguments=params)
                ))
            
            self._buffer = self._buffer[match.end():]
            if self._buffer:
                content_to_send += self._buffer
                self._buffer = ""
            
            self._in_tool_call_block = False
            return content_to_send, tool_calls

        # 2. Check if we are currently in a <tool_call> block (but not finished)
        if "<tool_call" in self._buffer:
            self._in_tool_call_block = True
            idx = self._buffer.find("<tool_call")
            if idx > 0:
                content_to_send = self._buffer[:idx]
                self._buffer = self._buffer[idx:]
                return content_to_send, []
            else:
                return None, []

        # 3. If we are in a tool call block but no complete match was found, do NOT flush.
        if self._in_tool_call_block:
            return None, []

        # 4. No tool call in sight. Flush as normal text.
        if self._buffer:
            content_to_send = self._buffer
            self._buffer = ""
            return content_to_send, []
        
        return None, []

    def render_tool_call(self, tool_call: ChatToolCall) -> str:
        """Render a tool call to XML format."""
        func = tool_call.function
        params = func.arguments if isinstance(func.arguments, dict) else json.loads(func.arguments)
        param_str = "".join([f"<parameter={k}>{v}</parameter>\n" for k, v in params.items()])
        return f"<tool_call><function={func.name}>{param_str}</function></tool_call>"

    def render_tool_result(self, tool_call_id: str, result: str) -> str:
        """Render a tool result to XML format."""
        return f"<tool_result>{result}</tool_result>"
