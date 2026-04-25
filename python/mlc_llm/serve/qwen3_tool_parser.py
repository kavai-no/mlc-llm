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
    <parameter=key>
    value
    </parameter>
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

        # Look for complete tool_call blocks: <tool_call>...</tool_call>
        invoke_pattern = re.compile(
            r"<tool_call>(.*?)</tool_call>", re.DOTALL
        )

        tool_calls: List[ChatToolCall] = []
        content_parts = []
        last_end = 0

        for invoke_match in invoke_pattern.finditer(text):
            # Add content before this tool_call block
            content_parts.append(text[last_end:invoke_match.start()])
            
            invoke_content = invoke_match.group(1)
            
            # Look for function blocks within tool_call
            func_pattern = re.compile(
                r"<function=(.*?)>(.*?)</function>", re.DOTALL
            )
            
            for func_match in func_pattern.finditer(invoke_content):
                func_name = func_match.group(1).strip()
                func_content = func_match.group(2)

                # Parse parameters
                param_dict: Dict[str, Any] = {}
                for p_match in re.finditer(
                    r"<parameter=(.*?)>(.*?)</parameter>", func_content, re.DOTALL
                ):
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
            
            last_end = invoke_match.end()

        # Add remaining content after last tool_call block
        content_parts.append(text[last_end:])
        
        # Join all content parts
        content = "".join(content_parts)
        
        return content, tool_calls

    def parse_streaming(self, token: str) -> tuple[Optional[str], List[ChatToolCall]]:
        """
        Parse streaming token.
        
        Returns:
            (content_delta, tool_calls) where:
            - content_delta is text to send to the user (None if buffering)
            - tool_calls is a list of complete tool calls found
        """
        self._buffer += token
        tool_calls: List[ChatToolCall] = []
        content_to_send: Optional[str] = None

        # Pattern for complete tool_call block
        invoke_pattern = re.compile(
            r"<tool_call>(.*?)</tool_call>", re.DOTALL
        )

        # Check for complete tool_call block in buffer
        match = invoke_pattern.search(self._buffer)
        if match:
            # Extract the complete tool_call block
            invoke_content = match.group(1)
            
            # Look for function blocks within tool_call
            func_pattern = re.compile(
                r"<function=(.*?)>(.*?)</function>", re.DOTALL
            )
            
            for func_match in func_pattern.finditer(invoke_content):
                func_name = func_match.group(1).strip()
                func_content = func_match.group(2)

                # Parse parameters
                param_dict: Dict[str, Any] = {}
                for p_match in re.finditer(
                    r"<parameter=(.*?)>(.*?)</parameter>", func_content, re.DOTALL
                ):
                    key = p_match.group(1).strip()
                    val = p_match.group(2).strip()
                    param_dict[key] = _try_convert_value(val)

                tool_call = ChatToolCall(
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    type="function",
                    function=ChatFunctionCall(
                        name=func_name,
                        arguments=param_dict,
                    ),
                )
                tool_calls.append(tool_call)

            # Content before the tool_call block should be sent
            content_to_send = self._buffer[:match.start()]
            
            # Update buffer to content after the tool_call block
            self._buffer = self._buffer[match.end():]
            
            return content_to_send, tool_calls

        # No complete tool_call block yet
        # Check if we're potentially starting an tool_call block
        if "<tool_call>" in self._buffer:
            invoke_idx = self._buffer.index("<tool_call>")
            if invoke_idx > 0:
                # Send content before the tool_call tag
                content_to_send = self._buffer[:invoke_idx]
                self._buffer = self._buffer[invoke_idx:]
                return content_to_send, []
            # All content is part of potential tool_call block, buffer it
            return None, []

        # No tool_call block detected, flush buffer
        content_to_send = self._buffer
        self._buffer = ""
        return content_to_send, []

    def render_tool_call(self, tool_call: ChatToolCall) -> str:
        """Render a tool call to XML format."""
        func = tool_call.function
        params = (
            func.arguments 
            if isinstance(func.arguments, dict) 
            else json.loads(func.arguments)
        )
        
        param_str = ""
        for k, v in params.items():
            param_str += f"<parameter={k}>\n{v}\n</parameter>\n"
        
        return f"<tool_call>\n<function={func.name}>\n{param_str}</function>\n</tool_call>\n"

    def render_tool_result(self, tool_call_id: str, result: str) -> str:
        """Render a tool result to XML format."""
        return f"<tool_result>\n{result}\n</tool_result>"
