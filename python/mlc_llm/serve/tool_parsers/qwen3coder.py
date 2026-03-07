# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the MLC LLM project
"""Qwen3 Coder tool parser for MLC LLM."""

import ast
import json
import uuid
from collections.abc import Sequence
from typing import Any, List, Optional, Union

import re
from pydantic import BaseModel

from mlc_llm.protocol.openai_api_protocol import (
    ChatCompletionRequest,
    ChatFunctionCall, ChatCompletionMessage,
    ChatToolCall)
from mlc_llm.serve.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager, ExtractedToolCallInformation)
from mlc_llm.support import logging
from mlc_llm.tokenizers import Tokenizer

logger = logging.getLogger(__name__)

def render_extra_keys(json_dict: dict, handled_keys: List[str]) -> str:
    """Render extra keys in a JSON-like dictionary dynamically."""
    if not isinstance(json_dict, dict):
        return ""

    extra_keys = []
    for json_key, value in json_dict.items():
        if json_key not in handled_keys:
            if isinstance(value, (dict, list)) and not isinstance(value, str):
                extra_keys.append(f"\n<{json_key}>{json.dumps(value, ensure_ascii=False)}</{json_key}>")
            else:
                extra_keys.append(f"\n<{json_key}>{str(value)}</{json_key}>")

    return ''.join(extra_keys)

@ToolParserManager.register_module("qwen3_coder")
class Qwen3CoderToolParser(ToolParser):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__(tokenizer)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []

        # Sentinel tokens for streaming mode
        self.tool_call_start_token: str = "\u25a1"
        self.tool_call_end_token: str = "\u25a2"
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"
        self.is_tool_call_started: bool = False
        self.failed_count: int = 0

        # Enhanced streaming state - reset for each new message
        self._reset_streaming_state()

        # Regex patterns
        self.tool_call_complete_regex = re.compile(
            r"\u25a1(.*?)\u25a2", re.DOTALL)
        self.tool_call_regex = re.compile(
            r"\u25a1(.*?)\u25a2|\u25a1(.*?)$", re.DOTALL)
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL)
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=<function>)|$)",
            re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction.")
        self.vocab = {
            "\u25a1": 151657,
            "\u25a2": 151658,
        }
        self.tool_call_start_token_id = self.vocab.get(
            self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        # Check if tokens are found in vocabulary
        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            logger.warning(
                f"Qwen3 XML Tool parser could not locate tool call start/end "
                f"tokens in the tokenizer! Start token '{self.tool_call_start_token}' "
                f"ID: {self.tool_call_start_token_id}, End token '{self.tool_call_end_token}' "
                f"ID: {self.tool_call_end_token_id}")
            # Try to find these tokens in a different way
            # This is a fallback for when the tokens are not directly in the vocab
            if self.tool_call_start_token_id is None:
                # Try to find the token by its string representation
                for token_id, token_str in self.vocab.items():
                    if token_str == self.tool_call_start_token:
                        self.tool_call_start_token_id = token_id
                        break
            if self.tool_call_end_token_id is None:
                # Try to find the token by its string representation
                for token_id, token_str in self.vocab.items():
                    if token_str == self.tool_call_end_token:
                        self.tool_call_end_token_id = token_id
                        break

        # If still not found, set to a default value to prevent crashes
        if self.tool_call_start_token_id is None:
            self.tool_call_start_token_id = 0
        if self.tool_call_end_token_id is None:
            self.tool_call_end_token_id = 0

        logger.info(
            f"MLC LLM Successfully import tool parser {self.__class__.__name__} !")

    def render_tools(self, tools: Optional[List[ChatToolCall]] = None) -> str:
        """Render qwen3 coder xml tool definitions to string."""
        if tools is None or not tools:
            return ""

        rendered_output = [
            "# Tools\n",
            "You have access to the following functions:\n\n"
        ]

        for tool in tools:
            # Start function block
            rendered_output.append("<function>\n")

            # Name
            if hasattr(tool, 'name') and tool.name:
                rendered_output.append(f"<name>{tool.name}</name>\n")

            # Description
            if hasattr(tool, 'description') and tool.description:
                rendered_output.append(f"<description>{tool.description.strip()}</description>\n")

            # Parameters
            if hasattr(tool, 'parameters') and tool.parameters:
                rendered_output.append("<parameters>\n")

                for param_name, param_fields in tool.parameters.items():
                    rendered_output.append("\n<parameter>\n")

                    # Name
                    if hasattr(param_fields, 'name') and param_fields.name:
                        rendered_output.append(f"<name>{param_fields.name}</name>\n")

                    # Type
                    if hasattr(param_fields, 'type') and param_fields.type:
                        rendered_output.append(f"<type>{str(param_fields.type)}</type>\n")

                    # Description
                    if hasattr(param_fields, 'description') and param_fields.description:
                        rendered_output.append(f"<description>{param_fields.description.strip()}</description>\n")

                    # Extra keys (e.g., enum, format)
                    handled_keys = ['name', 'type', 'description']
                    extra_keys = render_extra_keys(param_fields, handled_keys)
                    rendered_output.append(extra_keys)

                    rendered_output.append("</parameter>\n")

                # Extra keys for parameters (e.g., additionalProperties)
                handled_keys = ['properties', 'type']
                extra_keys = render_extra_keys(tool.parameters, handled_keys)
                rendered_output.append(extra_keys)

                rendered_output.append("</parameters>\n")

            # Extra keys for tool (e.g., externalDocs)
            handled_keys = ['name', 'description', 'parameters']
            extra_keys = render_extra_keys(tool, handled_keys)
            rendered_output.append(extra_keys)

            # End function block
            rendered_output.append("</function>\n")

        # Add instructions for tool calls
        rendered_output.extend([
            "If you choose to call a function ONLY reply in the following format with NO suffix:\n",
            "\n\u25a1\n",
            "<function=example_function_name>\n",
            "<parameter=example_parameter_1>\n",
            "value_1\n",
            "</parameter>\n",
            "<parameter=example_parameter_2>\n",
            "This is the value for the second parameter\n",
            "that can span\n",
            "multiple lines\n",
            "</parameter>\n",
            "</function>\n",
            "\u25a2\n",
            "\n<IMPORTANT>\n",
            "Reminder:\n",
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within \u25a1\u25a2 XML tags\n",
            "- Required parameters MUST be specified\n",
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n",
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n",
            "</IMPORTANT>\n"
        ])

        return ''.join(rendered_output)

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return str(uuid.uuid4())

    def _reset_streaming_state(self):
        """Reset streaming state for a new message."""
        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.streamed_args_for_tool = []
        self.is_tool_call_started = False
        self.failed_count = 0

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from model output."""
        self._reset_streaming_state()
        tools_called = False
        tool_calls = []
        content = None

        # Check if there's any content before tool calls
        if model_output.strip():
            # Extract content before the first tool call
            match = self.tool_call_complete_regex.search(model_output)
            if match:
                content = model_output[:match.start()].strip()

        # Find all tool calls
        for match in self.tool_call_complete_regex.finditer(model_output):
            tool_call_content = match.group(1)
            
            # Parse the tool call
            tool_call = self._parse_tool_call(tool_call_content)
            if tool_call:
                tools_called = True
                tool_calls.append(tool_call)

        return ExtractedToolCallInformation(
            tools_called=tools_called,
            tool_calls=tool_calls,
            content=content
        )

    def _parse_tool_call(self, tool_call_content: str) -> Optional[ChatToolCall]:
        """Parse a tool call from the content."""
        tool_call = ChatToolCall()
        tool_call.id = self._generate_tool_call_id()
        tool_call.type = "function"

        # Extract function name
        function_match = self.tool_call_function_regex.search(tool_call_content)
        if function_match:
            func_name = function_match.group(1) or function_match.group(2)
            tool_call.function = ChatFunctionCall(name=func_name)

            # Extract parameters
            params_content = self.tool_call_parameter_regex.findall(tool_call_content)
            params = {}
            for param_content in params_content:
                param_name, param_value = self._parse_parameter(param_content)
                if param_name and param_value is not None:
                    params[param_name] = param_value
            
            if tool_call.function:
                tool_call.function.name = tool_call.function.name
                tool_call.function.arguments = json.dumps(params)

        return tool_call

    def _parse_parameter(self, param_content: str) -> tuple:
        """Parse a parameter from the content."""
        param_name = None
        param_value = None

        # Try to extract parameter name
        name_match = re.search(r'<name>(.*?)</name>', param_content)
        if name_match:
            param_name = name_match.group(1).strip()

        # Try to extract parameter value
        value_match = re.search(r'<value>(.*?)</value>', param_content)
        if value_match:
            param_value = value_match.group(1).strip()

        return param_name, param_value

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
        self._reset_streaming_state()

        # Check if we're entering a tool call
        if self.tool_call_start_token_id in delta_token_ids:
            self.is_tool_call_started = True
            self.current_tool_id += 1

        if not self.is_tool_call_started:
            return None

        # Check if we're exiting a tool call
        if self.tool_call_end_token_id in delta_token_ids:
            self.is_tool_call_started = False
            self.current_tool_name_sent = False

        # Extract tool calls from the current text
        if self.tool_call_start_token_id in current_token_ids:
            # Extract content between start and end tokens
            match = self.tool_call_complete_regex.search(current_text)
            if match:
                tool_call_content = match.group(1)
                tool_call = self._parse_tool_call(tool_call_content)
                if tool_call:
                    return ChatCompletionMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[tool_call]
                    )

        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tokenizer={self.model_tokenizer})"
