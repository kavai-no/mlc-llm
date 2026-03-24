# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the MLC LLM project
"""Qwen3 Coder tool parser for MLC LLM."""

import ast
import json
import re
import uuid
from typing import Any, List, Optional, Union

from mlc_llm.protocol.openai_api_protocol import (
    ChatCompletionRequest,
    ChatFunctionCall, ChatCompletionMessage,
    ChatTool, ChatToolCall)
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
        # Track current function name during streaming
        self.current_function_name: str = ""

        # Sentinel tokens for streaming mode
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
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
            r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL | re.IGNORECASE)
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL | re.IGNORECASE)
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=<function>)|$)",
            re.DOTALL | re.IGNORECASE)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction.")

        # Get token IDs using tokenizer or vocab lookup
        try:
            # MLC tokenizer.encode returns list of ids
            start_ids = self.model_tokenizer.encode(self.tool_call_start_token)
            self.tool_call_start_token_id = start_ids[0] if isinstance(start_ids, (list, tuple)) and start_ids else None
            end_ids = self.model_tokenizer.encode(self.tool_call_end_token)
            self.tool_call_end_token_id = end_ids[0] if isinstance(end_ids, (list, tuple)) and end_ids else None
        except Exception:
            # Fallback to vocab lookup (vocab may be token->id or id->token)
            self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
            self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        logger.debug(f"Qwen3 tool call tokens: {self.tool_call_start_token}={self.tool_call_start_token_id}, "
                     f"{self.tool_call_end_token}={self.tool_call_end_token_id}")

        # Check if tokens are found in vocabulary
        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError(
                f"Qwen3 XML Tool parser could not locate tool call start/end "
                f"tokens in the tokenizer! Start token '{self.tool_call_start_token}' "
                f"ID: {self.tool_call_start_token_id}, End token '{self.tool_call_end_token}' "
                f"ID: {self.tool_call_end_token_id}")

        logger.info(
            f"MLC LLM Successfully imported tool parser {self.__class__.__name__} !")

    def render_tools(self, tools: Optional[List[ChatTool]] = None) -> str:
        """Render qwen3 coder xml tool definitions to string.
        
        Only renders the function definitions (to be placed inside <tools> by the template).
        Instructions are already in the qwen3_coder template.
        """
        logger.info(f"render_tools called: tools={tools}, type={type(tools)}")
        
        if not tools:
            # Accept both None and empty list; log and return empty string
            logger.warning("render_tools: tools is empty or None")
            return ""

        rendered_output = []

        logger.info(f"Rendering {len(tools)} tools for qwen3_coder")

        loop_count = 0
        for item in tools:
            loop_count += 1
            logger.debug(f"Processing tool {loop_count}: type={type(item).__name__}, has_function={hasattr(item, 'function')}")
            
            # Support both ChatTool and direct ChatFunction (from tool_choice branch)
            if hasattr(item, "function") and item.function is not None:
                func = item.function
                tool_obj = item
            else:
                func = item
                tool_obj = None
            
            logger.debug(f"Tool {loop_count}: func type={type(func).__name__}, has_name={hasattr(func, 'name')}")
            
            # Start function block
            rendered_output.append("<function>\n")

            # Name
            if hasattr(func, 'name') and func.name:
                rendered_output.append(f"<name>{func.name}</name>\n")

            # Description
            if hasattr(func, 'description') and func.description:
                rendered_output.append(f"<description>{func.description.strip()}</description>\n")

            # Parameters (access through func)  
            params = getattr(func, 'parameters', None) or (getattr(tool_obj, 'parameters', None) if tool_obj else None)
            if params and isinstance(params, dict):
                rendered_output.append("<parameters>\n")

                # Standard OpenAI schema has 'properties' key
                properties = params.get("properties", params)
                if isinstance(properties, dict):
                    for param_name, param_fields in properties.items():
                        rendered_output.append("\n<parameter>\n")

                        # Name
                        rendered_output.append(f"<name>{param_name}</name>\n")

                        if isinstance(param_fields, dict):
                            # Type
                            if "type" in param_fields:
                                rendered_output.append(f"<type>{str(param_fields['type'])}</type>\n")

                            # Description
                            if "description" in param_fields and param_fields["description"]:
                                rendered_output.append(f"<description>{str(param_fields['description']).strip()}</description>\n")

                            # Extra keys (e.g., enum, format)
                            handled_keys = ['name', 'type', 'description']
                            extra_keys = render_extra_keys(param_fields, handled_keys)
                            rendered_output.append(extra_keys)

                        rendered_output.append("</parameter>\n")

                # Extra keys for parameters (e.g., additionalProperties, required)
                handled_keys = ['properties', 'type', 'required']
                extra_keys = render_extra_keys(params, handled_keys)
                rendered_output.append(extra_keys)

                rendered_output.append("</parameters>\n")

            # Extra keys for function 
            if hasattr(func, 'model_dump'):
                func_dict = func.model_dump()
                handled_keys = ['name', 'description', 'parameters']
                extra_keys = render_extra_keys(func_dict, handled_keys)
                rendered_output.append(extra_keys)
            elif hasattr(func, '__dict__'):
                handled_keys = ['name', 'description', 'parameters']
                extra_keys = render_extra_keys(func.__dict__, handled_keys)
                rendered_output.append(extra_keys)

            # End function block
            rendered_output.append("</function>\n")

        logger.info(f"Rendered output list has {len(rendered_output)} items")
        rendered = ''.join(rendered_output)
        logger.info(f"After join: rendered length={len(rendered)}")
        if rendered:
            token_count = len(self.model_tokenizer.encode(rendered))
            logger.info(f"Rendered {len(tools)} tools for qwen3_coder (length {len(rendered)} chars, {token_count} tokens)")
            logger.debug(f"Rendered tools preview:\n{rendered[:500]}")
        else:
            logger.warning(f"No tools rendered! Input had {len(tools)} tools, output list had {len(rendered_output)} items")
        return rendered

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self):
        """Reset all streaming state."""
        self.current_tool_index = 0
        self.is_tool_call_started = False
        self.header_sent = False
        self.current_tool_id = None
        self.current_function_name = None
        self.current_param_name = None
        self.current_param_value = ""
        self.param_count = 0
        self.in_param = False
        self.in_function = False
        self.accumulated_text = ""
        self.json_started = False
        self.json_closed = False
        # Store accumulated parameters for type conversion
        self.accumulated_params = {}
        self.streaming_request = None


    def _get_arguments_config(
            self, func_name: str,
            tools: Optional[list[ChatTool]]) -> dict:
        """Extract argument configuration for a function."""
        if tools is None:
            return {}
        
        # Handle both ChatTool objects and direct function configs
        for config in tools:
            # Try different ways to access the function info
            func_obj = getattr(config, 'function', None)
            if func_obj is None:
                # Direct function object (from tool_choice branch)
                if hasattr(config, 'name') and config.name == func_name:
                    params = getattr(config, 'parameters', None)
                    if isinstance(params, dict) and 'properties' in params:
                        return params['properties']
                    elif isinstance(params, dict):
                        return params
                    else:
                        return {}
                continue
                
            # Normal ChatTool case
            if hasattr(func_obj, 'name') and func_obj.name == func_name:
                params = getattr(func_obj, 'parameters', None)
                if isinstance(params, dict) and 'properties' in params:
                    return params['properties']
                elif isinstance(params, dict):
                    return params
                else:
                    return {}
        
        logger.warning(f"Tool '{func_name}' is not defined in the tools list.")
        return {}

    def _convert_param_value(self, param_value: Any, param_name: str,
                             param_config: dict, func_name: str) -> Union[str, int, float, bool, dict, list, None]:
        """Convert parameter value based on its type in the schema."""
        
        # Handle null value for any type
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config != {}:
                logger.warning(
                    f"Parsed parameter '{param_name}' is not defined in the tool "
                    f"parameters for tool '{func_name}', directly returning the string value."
                )
            return param_value

        # Get parameter type from config
        param_type_info = param_config[param_name]
        if isinstance(param_type_info, dict) and "type" in param_type_info:
            param_type = str(param_type_info["type"]).strip().lower()
        else:
            param_type = "string"
            
        # Type conversion based on schema
        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        elif param_type.startswith("int") or param_type.startswith(
                "uint") or param_type.startswith(
                    "long") or param_type.startswith(
                        "short") or param_type.startswith("unsigned"):
            try:
                param_value = int(param_value)
            except ValueError:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not an integer in tool "
                    f"'{func_name}', degenerating to string.")
            return param_value
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                float_param_value = float(param_value)
                param_value = float_param_value if float_param_value - int(
                    float_param_value) != 0 else int(float_param_value)
            except ValueError:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a float in tool "
                    f"'{func_name}', degenerating to string.")
            return param_value
        elif param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            if param_value not in ["true", "false"]:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a boolean (`true` of `false`) in tool '{func_name}', degenerating to false."
                )
            return param_value == "true"
        else:
            # For complex types (object, array), try JSON parsing first
            if param_type in ["object", "array", "arr"] or param_type.startswith("dict") or param_type.startswith("list"):
                try:
                    param_value = json.loads(param_value)
                    return param_value
                except json.JSONDecodeError:
                    logger.warning(
                        f"Parsed value '{param_value}' of parameter '{param_name}' cannot be parsed with json.loads in tool "
                        f"'{func_name}', will try other methods to parse it.")
            
            # Fallback to Python literal evaluation
            try:
                param_value = ast.literal_eval(param_value)
                return param_value
            except (ValueError, SyntaxError):
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' cannot be converted via Python `ast.literal_eval()` in tool '{func_name}', degenerating to string.")
            return param_value

    def _get_function_calls(self, model_output: str) -> list[str]:
        """Extract function call strings from model output, supporting both wrapped and unwrapped formats."""
        # First try wrapped format: <tool_call>...<function=...></function>...</tool_call>
        tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL | re.IGNORECASE)
        matched_ranges = tool_call_regex.findall(model_output)
        raw_tool_calls = [match[0] if match[0] else match[1] for match in matched_ranges]
        
        # Fallback to unwrapped format if no wrapped calls found
        if len(raw_tool_calls) == 0:
            raw_tool_calls = [model_output]
        
        raw_function_calls = []
        function_regex = re.compile(r"<function=([^>]+?)>(.*?)</function>|<function=(.*)$", re.DOTALL | re.IGNORECASE)
        for tool_call in raw_tool_calls:
            raw_function_calls.extend(function_regex.findall(tool_call))
        
        # Extract function strings (name + content)
        function_calls = []
        for match in raw_function_calls:
            if match[0]:  # <function=NAME>content</function> format
                function_calls.append("<function=" + match[0] + ">" + match[1] + "</function>")
            elif match[2]:  # Incomplete tag
                function_calls.append(match[2])
        
        return function_calls

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from model output, supporting vllm format."""
        self._reset_streaming_state()
        
        # Quick check to avoid unnecessary processing
        if self.tool_call_prefix not in model_output:  
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )
        
        try:
            function_calls = self._get_function_calls(model_output)
            
            # Log debugging info for troubleshooting
            logger.debug(f"Extracted {len(function_calls)} function calls from model output")
            if len(function_calls) == 0:
                logger.warning(f"No function calls found in model output. Output preview: {model_output[:200]!r}")
            
            if len(function_calls) == 0:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output
                )
            
            tool_calls = []
            for i, func_str in enumerate(function_calls):
                try:
                    tool_call = self._parse_tool_call(func_str, request.tools if request else None)
                    if tool_call:
                        logger.debug(f"Successfully parsed tool call {i+1}: {tool_call.function.name}")
                        # Add index field to match OpenAI API specification
                        tool_call.index = i
                        tool_calls.append(tool_call)
                    else:
                        logger.warning(f"Failed to parse tool call at index {i}: {func_str[:100]!r}...")
                except Exception as e:
                    logger.exception(f"Exception while parsing tool call at index {i}: {e}")
            
            # Populate prev_tool_call_arr for serving layer to set finish_reason
            self.prev_tool_call_arr.clear()  # Clear previous calls
            for tool_call in tool_calls:
                if tool_call and hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name'):
                    self.prev_tool_call_arr.append({
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    })
            
            # Extract content before tool calls
            content_index = model_output.find(self.tool_call_start_token)
            if content_index < 0:
                content_index = model_output.find(self.tool_call_prefix)
            content = model_output[:content_index] if content_index >= 0 else None
            
            logger.info(f"Qwen3CoderToolParser.extract result: tools_called={len(tool_calls) > 0}, num_tools={len(tool_calls)}, content_preview={str(content)[:100] if content else None}")
            return ExtractedToolCallInformation(
                tools_called=(len(tool_calls) > 0),
                tool_calls=tool_calls,
                content=content
            )
        except Exception as e:
            logger.exception(f"Error in extracting tool call from response: {e}")
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )

    def _parse_tool_call(self, function_call_str: str, tools: Optional[list[ChatTool]] = None) -> Optional[ChatToolCall]:
        """Parse an XML function call string to ChatToolCall.
        Format: <function=FUNCTION_NAME>\\n<parameter=PARAM1>value1</parameter>\\n...</function>
        """
        try:
            # Extract function name from <function=NAME>
            end_index = function_call_str.find(">")
            if end_index < 0:
                logger.warning(f"Could not find function name end in: {function_call_str[:100]!r}")
                return None
            
            function_name = function_call_str[:end_index].strip()
            if function_name.startswith("<function="):
                function_name = function_name[10:]  # Remove '<function='
            function_name = function_name.strip()
            
            # Extract parameters
            parameters_str = function_call_str[end_index + 1:]
            param_dict = {}
            
            # Use regex to find all parameters
            param_regex = re.compile(
                r'<parameter\s*=\s*([^>]+)>(.*?)</parameter>',
                re.DOTALL | re.IGNORECASE
            )
            for param_name_match, param_value in param_regex.findall(parameters_str):
                param_name = param_name_match.strip()
                param_value = param_value.strip()
                
                # Clean up newlines
                if param_value.startswith("\n"):
                    param_value = param_value[1:]
                if param_value.endswith("\n"):
                    param_value = param_value[:-1]
                
                param_dict[param_name] = param_value
            
            # Convert parameter values based on schema
            param_config = self._get_arguments_config(function_name, tools)
            converted_param_dict = {}
            for param_name, param_value in param_dict.items():
                try:
                    converted_value = self._convert_param_value(param_value, param_name, param_config.get(param_name, {}), function_name)
                    converted_param_dict[param_name] = converted_value
                except Exception as e:
                    logger.warning(f"Failed to convert parameter '{param_name}' for tool '{function_name}': {e}")
                    # Fallback to original value if conversion fails
                    converted_param_dict[param_name] = param_value
            
            # Create and return ChatToolCall with JSON string arguments (matching vllm format)
            return ChatToolCall(
                type="function",
                function=ChatFunctionCall(
                    name=function_name,
                    arguments=json.dumps(converted_param_dict, ensure_ascii=False) if converted_param_dict else "{}"
                )
            )
        except Exception as e:
            logger.exception(f"Failed to parse tool call from: {function_call_str[:100]!r} - {e}")
            return None

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
        """Extract tool calls from streaming model output.

        The logic follows the vLLM reference implementation with fine-grained
        incremental parsing for real-time parameter updates.
        """

        # Store request for type conversion during streaming (matching vLLM)
        if not previous_text:
            self._reset_streaming_state()
            self.streaming_request = request

        # If no delta text, return None unless it's an EOS token after tool calls
        if not delta_text:
            # Check if this is an EOS token after all tool calls are complete
            if (self.tool_call_end_token_id in delta_token_ids or 
                self.tool_call_end_token in delta_text):
                pass  # Let the normal flow handle it for completeness check

            return None

        # Update accumulated text and content tracking
        self.accumulated_text = current_text

        # Handle state transitions when not yet started a tool call
        if not self.is_tool_call_started:
            # Check if entering first part of new token sequence (for custom tokens)
            start_token_in_delta = (
                delta_text.startswith(self.tool_call_prefix) or 
                any(token in str(delta_text).encode('utf-8', 'replace').decode() for token in [self.tool_call_start_token]) or
                self.tool_call_start_token_id in getattr(request, '_token_ids', [])
            )
            
            # Fallback: check if we see the start tokens directly  
            tool_starts_in_text = current_text.count(self.tool_call_prefix)
            prev_tool_calls_processed = len([t for t in self.prev_tool_call_arr if 'name' in t])
            
            is_new_content_starting_with_function_delta = (
                previous_text.endswith('\n') and delta_text.lstrip().startswith('<function=') or 
                (not hasattr(self, '_last_seen_prefix_len')) # first time seeing content after reset
            )

        # Check if we need to advance to next tool call based on completed vs processing count
        current_tool_starts_count = (
            self.accumulated_text.count('<function=') + 
            (self.tool_call_start_token in delta_text)
        )
        
        has_more_calls_available = False

    def _handle_new_message_reset(self):
        """Helper method to reset for new message processing."""
        pass # Already handled above    
        return f"{self.__class__.__name__}(tokenizer={self.model_tokenizer})"
