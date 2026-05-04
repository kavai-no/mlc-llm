import json
import re
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar

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
    Parser for Qwen3-Coder XML-style tool calls.

    Model output format:
    <tool_call>
    <function=name>
    <parameter=key>value</parameter>
    ...
    </function>
    </tool_call>
    """

    def __init__(self):
        self._buffer = ""

    def parse(self, text: str) -> tuple[str, List[ChatToolCall]]:
        """Parse non-streaming complete text."""
        if not text.strip():
            return text, []

        tool_calls: List[ChatToolCall] = []
        content_parts = []
        last_end = 0

        # Find all <tool_call> blocks
        tool_call_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        matches = list(tool_call_pattern.finditer(text))

        for match in matches:
            content_parts.append(text[last_end:match.start()])
            inner_content = match.group(1)
            
            func_pattern = re.compile(r"<function=(.*?)>(.*?)</function>", re.DOTALL)
            for func_match in func_pattern.finditer(inner_content):
                func_name = func_match.group(1).strip()
                func_body = func_match.group(2)

                param_dict: Dict[str, Any] = {}
                param_pattern = re.compile(r"<parameter=(.*?)>(.*?)</parameter>", re.DOTALL)
                for p_match in param_pattern.finditer(func_body):
                    key = p_match.group(1).strip()
                    val = p_match.group(2).strip()
                    param_dict[key] = _try_convert_value(val)

                tool_calls.append(
                    ChatToolCall(
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        type="function",
                        function=ChatFunctionCall(
                            name=func_name,
                            arguments=param_dict,
                        ),
                    )
                )
            last_end = match.end()

        content_parts.append(text[last_end:])
        return "".join(content_parts), tool_calls

    def parse_streaming(self, token: str) -> tuple[Optional[str], List[ChatToolCall]]:
        """Parse streaming token."""
        self._buffer += token
        tool_calls: List[ChatToolCall] = []
        content_to_send: Optional[str] = None

        # 1. Check for complete <tool_call> block
        tool_call_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        match = tool_call_pattern.search(self._buffer)

        if match:
            # Everything before the <tool_call> is content to send
            content_to_send = self._buffer[:match.start()]
            inner_content = match.group(1)
            
            func_pattern = re.compile(r"<function=(.*?)>(.*?)</function>", re.DOTALL)
            for func_match in func_pattern.finditer(inner_content):
                func_name = func_match.group(1).strip()
                func_body = func_match.group(2)

                param_dict: Dict[str, Any] = {}
                param_pattern = re.compile(r"<parameter=(.*?)>(.*?)</parameter>", re.DOTALL)
                for p_match in param_pattern.finditer(func_body):
                    key = p_match.group(1).strip()
                    val = p_match.group(2).strip()
                    param_dict[key] = _try_convert_value(val)

                tool_calls.append(
                    ChatToolCall(
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        type="function",
                        function=ChatFunctionCall(
                            name=func_name,
                            arguments=param_dict,
                        ),
                    )
                )

            # Update buffer to everything AFTER the </tool_call> tag
            self._buffer = self._buffer[match.end():]
            
            if self._buffer:
                content_to_send += self._buffer
                self._buffer = ""
                
            return content_to_send, tool_calls

        # 2. Check if we are currently inside a <tool_call> block (but not finished)
        if "<tool_call" in self._buffer:
            # We find the start of the tag to see if there is text before it
            idx = self._buffer.find("<tool_call")
            if idx > 0:
                # There is text before the <tool_call> tag. Send it and buffer the rest.
                content_to_send = self._buffer[:idx]
                self._buffer = self._buffer[idx:]
                return content_to_send, []
            else:
                # The tool call starts at index 0. Buffer everything and return None.
                return None, []

        # 3. No <tool_call> tag in the buffer; flush as text.
        content_to_send = self._buffer
        self._buffer = ""
        return content_to_send, []

    def render_tool_call(self, tool_call: ChatToolCall) -> str:
        """Render a tool call to XML format."""
        func = tool_call.function
        params = func.arguments if isinstance(func.arguments, dict) else json.loads(func.arguments)
        param_str = "".join([f"<parameter={k}>{v}</parameter>\n" for k, v in params.items()])
        return f"<tool_call><function={func.name}>{param_str}</function></tool_call>"

    def render_tool_result(self, tool_call_id: str, result: str) -> str:
        """Render a tool result to XML format."""
        return f"<tool_result>{result}</tool_result>"
