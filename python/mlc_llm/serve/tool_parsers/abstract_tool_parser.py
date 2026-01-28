# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the MLC LLM project
"""Abstract tool parser base class for MLC LLM."""

import abc
from typing import Any, Dict, List, Optional, Union

from mlc_llm.protocol.openai_api_protocol import (
    ChatCompletionRequest,
    ChatFunctionCall, ChatCompletionMessage,
    ChatToolCall)
from pydantic import BaseModel
from mlc_llm.tokenizers import Tokenizer

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ToolParserManager:
    """Tool parser manager for registering and retrieving tool parsers."""

    _parsers: Dict[str, type] = {}

    @classmethod
    def register_module(cls, name: str):
        """Register a tool parser class."""
        def decorator(tool_parser_class: type):
            cls._parsers[name] = tool_parser_class
            return tool_parser_class
        return decorator

    @classmethod
    def get_parser(cls, name: str) -> type:
        """Get a tool parser class by name."""
        return cls._parsers.get(name)

    @classmethod
    def get_parser_names(cls) -> List[str]:
        """Get all registered parser names."""
        return list(cls._parsers.keys())