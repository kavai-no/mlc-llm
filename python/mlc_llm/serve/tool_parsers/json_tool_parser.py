# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the MLC LLM project
"""JSON tool parser for MLC LLM."""

import json
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from mlc_llm.protocol.openai_api_protocol import (
    ChatCompletionRequest,
    ChatFunctionCall, ChatCompletionMessage,
    ChatToolCall)
from mlc_llm.serve.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager, ExtractedToolCallInformation)
from mlc_llm.tokenizers import Tokenizer


@ToolParserManager.register_module("json")
class JsonToolParser(ToolParser):
    """A parser for JSON-formatted tool calls."""
    
    def __init__(self, tokenizer: Tokenizer):
        """Initialize the JSON tool parser.
        
        Args:
            tokenizer: The model tokenizer.
        """
        super().__init__(tokenizer)
        self.model_tokenizer = tokenizer
        self.vocab = tokenizer.vocab if hasattr(tokenizer, 'vocab') else {}
    
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from JSON-formatted model output.
        
        Args:
            model_output: The model output string.
            request: The completion request.
            
        Returns:
            ExtractedToolCallInformation with tool calls and content.
        """
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
        """Extract tool calls from streaming JSON-formatted model output.
        
        Args:
            previous_text: Previous accumulated text.
            current_text: Current accumulated text.
            delta_text: Delta text since last update.
            previous_token_ids: Previous token IDs.
            current_token_ids: Current token IDs.
            delta_token_ids: Delta token IDs.
            request: The completion request.
            
        Returns:
            ChatCompletionMessage with tool calls or None.
        """
        try:
            # Parse the current text to extract tool calls
            data = json.loads(current_text)
            if isinstance(data, dict) and "tool_calls" in data:
                for tool_call in data["tool_calls"]:
                    return ChatCompletionMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[ChatToolCall(
                            id=tool_call.get("id", str(uuid.uuid4())),
                            type="function",
                            function=ChatFunctionCall(
                                name=tool_call.get("name", "unknown_function"),
                                arguments=json.dumps(tool_call.get("arguments", {})),
                            ),
                        )],
                    )
        except json.JSONDecodeError:
            pass
        
        return None
    
    def render_tools(self, tools: Optional[List[ChatToolCall]] = None) -> str:
        """Render tool definitions to string.
        
        Args:
            tools: List of tool calls to render.
            
        Returns:
            Rendered tool definitions string.
        """
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
            if hasattr(tool, 'name') and tool.function and tool.function.name:
                rendered_output.append(f"<name>{tool.function.name}</name>\n")
            
            # Description
            if hasattr(tool, 'function') and hasattr(tool.function, 'description') and tool.function.description:
                rendered_output.append(f"<description>{tool.function.description.strip()}</description>\n")
            
            # Parameters
            if hasattr(tool, 'function') and hasattr(tool.function, 'arguments') and tool.function.arguments:
                try:
                    params = json.loads(tool.function.arguments)
                    rendered_output.append("<parameters>\n")
                    
                    for param_name, param_value in params.items():
                        rendered_output.append(f"<parameter>\n")
                        rendered_output.append(f"<name>{param_name}</name>\n")
                        rendered_output.append(f"<value>{str(param_value)}</value>\n")
                        rendered_output.append("</parameter>\n")
                    
                    rendered_output.append("</parameters>\n")
                except json.JSONDecodeError:
                    pass
            
            # End function block
            rendered_output.append("</function>\n")
        
        # Add instructions for tool calls
        rendered_output.extend([
            "If you choose to call a function ONLY reply in the following format with NO suffix:\n",
            "\n<function=example_function_name>\n",
            "<parameter=example_parameter_1>\n",
            "value_1\n",
            "</parameter>\n",
            "<parameter=example_parameter_2>\n",
            "This is the value for the second parameter\n",
            "that can span\n",
            "multiple lines\n",
            "</parameter>\n",
            "</function>\n",
            "\n<IMPORTANT>\n",
            "Reminder:\n",
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within XML tags\n",
            "- Required parameters MUST be specified\n",
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n",
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n",
            "</IMPORTANT>\n"
        ])
        
        return ''.join(rendered_output)
