# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the MLC LLM project
"""Abstract tool parser base class for MLC LLM."""

import abc
import json
import uuid
from typing import Any, Dict, List, Optional, Union

import pydantic


from mlc_llm.protocol.openai_api_protocol import (
    ChatCompletionRequest,
    ChatFunctionCall, ChatCompletionMessage,
    ChatTool, ChatToolCall)
from mlc_llm.tokenizers import Tokenizer
from .tool_parser_manager import ToolParserManager

# Alias for consistency with other parsers
BaseModel = pydantic.BaseModel


# Define the missing ExtractedToolCallInformation class
class ExtractedToolCallInformation(BaseModel):
    tools_called: bool
    tool_calls: List[ChatToolCall]
    content: Optional[str]

class ToolParser(abc.ABC):
    """Abstract base class for tool parsers."""
    def __init__(self, tokenizer: Tokenizer):
        self.model_tokenizer = tokenizer
        self.vocab = tokenizer.vocab if hasattr(tokenizer, 'vocab') else {}

    @abc.abstractmethod
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from model output."""

    @abc.abstractmethod
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: List[int],
        current_token_ids: List[int],
        delta_token_ids: List[int],
        request: ChatCompletionRequest,
    ) -> Union[ChatCompletionMessage, None]:
        """Extract tool calls from streaming model output."""

    @abc.abstractmethod
    def render_tools(self, tools: Optional[List[ChatTool]] = None) -> str:
        """Render tool definitions to string."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    

@ToolParserManager.register_module("default")
class DefaultToolParser(ToolParser):
    """Default tool parser."""

    def __init__(self, tokenizer: Tokenizer):
        """Initialize the tool parser."""
        super().__init__(tokenizer)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from JSON-formatted model output."""
        tools_called = False
        tool_calls = []
        content = None

        try:
            # Parse the model output to extract tool calls
            data = json.loads(model_output)
            if isinstance(data, dict) and "tool_calls" in data:
                for tool_call in data["tool_calls"]:
                    tool_calls.append(
                        ChatToolCall(
                            id=tool_call.get("id", str(uuid.uuid4())),
                            type="function",
                            function=ChatFunctionCall(
                                name=tool_call.get("name", "unknown_function"),
                                arguments=json.dumps(tool_call.get("arguments", {})),
                            ),
                        )
                    )
                tools_called = True
            content = data.get("content", None)
        except json.JSONDecodeError:
            # Fallback to plain text if JSON parsing fails
            pass

        return ExtractedToolCallInformation(
            tools_called=tools_called,
            tool_calls=tool_calls,
            content=content,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: List[int],
        current_token_ids: List[int],
        delta_token_ids: List[int],
        request: ChatCompletionRequest,
    ) -> Union[ChatCompletionMessage, None]:
        """Extract tool calls from streaming JSON-formatted model output."""
        try:
            # Parse the current text to extract tool calls
            data = json.loads(current_text)
            if isinstance(data, dict) and "tool_calls" in data:
                for tool_call in data["tool_calls"]:
                    return ChatCompletionMessage(
                        role="assistant",
                        content=None,
                        function=ChatFunctionCall(
                            name=tool_call.get("name", "unknown_function"),
                            arguments=json.dumps(tool_call.get("arguments", {})),
                        ),
                    )
        except json.JSONDecodeError:
            pass
        return None

    def render_tools(self, tools: Optional[List[ChatTool]] = None) -> str:
        """Render tool definitions to a JSON string."""
        if not tools or len(tools) == 0:
            return ""

        rendered_output = [
            "# Tools\n",
            "You have access to the following functions:\n\n"
        ]

        for tool in tools:
            rendered_output.append(f"```json\n{tool}\n```\n")

        return "".join(rendered_output)

