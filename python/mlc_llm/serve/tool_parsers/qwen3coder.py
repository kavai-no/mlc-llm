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
        self.current_tool_id: Optional[str] = None
        self.streamed_args_for_tool: list[str] = []
        # Track current function name during streaming
        self.current_function_name: Optional[str] = None

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
            logger.info(f"No tool calls detected in model output (length: {len(model_output)})")
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )
        
        try:
            function_calls = self._get_function_calls(model_output)
            
            # Enhanced debugging and logging
            logger.debug(f"Extracted {len(function_calls)} function calls from model output (length: {len(model_output)})")
            if len(function_calls) == 0:
                logger.warning(f"No function calls found in model output. Output preview: {model_output[:200]!r}")
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output
                )
            
            # Detailed logging for each function call found
            logger.info(f"Found {len(function_calls)} function calls:")
            for i, func_call in enumerate(function_calls):
                logger.debug(f"Function call {i+1}: {func_call[:200]!r}...")
            
            tool_calls = []
            parsing_errors = 0
            for i, func_str in enumerate(function_calls):
                try:
                    tool_call = self._parse_tool_call(func_str, request.tools if request else None)
                    if tool_call:
                        logger.debug(f"Successfully parsed tool call {i+1}: name={tool_call.function.name}")
                        # Add index field to match OpenAI API specification
                        tool_call.index = i
                        tool_calls.append(tool_call)
                    else:
                        logger.warning(f"Failed to parse tool call at index {i}: {func_str[:100]!r}...")
                        parsing_errors += 1
                except Exception as e:
                    logger.exception(f"Exception while parsing tool call at index {i}: {e}")
                    parsing_errors += 1
            
            # Log summary of parsing results
            successful_parses = len(tool_calls)
            total_attempts = len(function_calls)
            if successful_parses < total_attempts:
                logger.warning(f"Parsing summary: {successful_parses}/{total_attempts} function calls successfully parsed")
            
            # Populate prev_tool_call_arr for serving layer to set finish_reason
            self.prev_tool_call_arr.clear()  # Clear previous calls
            for tool_call in tool_calls:
                if tool_call and hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name'):
                    self.prev_tool_call_arr.append({
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    })
            
            # Enhanced content extraction
            # Look for both tool_call wrapper and direct function= prefix
            content_index = model_output.find(self.tool_call_start_token)
            if content_index < 0:
                content_index = model_output.find(self.tool_call_prefix)
            
            if content_index >= 0:
                content = model_output[:content_index].strip()
                # Remove trailing whitespace and newlines
                while content.endswith('\n') or content.endswith(' '):
                    content = content[:-1]
                logger.debug(f"Extracted content before tool calls (length: {len(content)})")
            else:
                content = None
            
            # Comprehensive result logging
            result_summary = {
                "tools_called": len(tool_calls) > 0,
                "num_tools": len(tool_calls),
                "parsing_success_rate": successful_parses / total_attempts if total_attempts > 0 else 1.0,
                "content_length": len(content) if content else 0
            }
            
            logger.info(f"Qwen3CoderToolParser.extract result: {result_summary}")
            return ExtractedToolCallInformation(
                tools_called=(len(tool_calls) > 0),
                tool_calls=tool_calls,
                content="" if not content else content
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
            
            # Use the same regex as VLLM for parameter extraction
            param_regex = self.tool_call_parameter_regex
            for match_text in param_regex.findall(parameters_str):
                idx = match_text.index(">")
                param_name = match_text[:idx]
                param_value = str(match_text[idx + 1:])
                
                # Remove prefix and trailing \n (matching VLLM behavior)
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
            
            # Validate required parameters
            tools_list = tools or []
            required_params = []
            additional_properties = True
            properties = {}
            for config in tools_list:
                func_obj = getattr(config, 'function', None)
                if func_obj is not None and hasattr(func_obj, 'parameters'):
                    params = func_obj.parameters
                    if isinstance(params, dict) and 'required' in params:
                        required_params = params['required']
                        properties = params.get('properties', {})
                        additional_properties = params.get('additionalProperties', True)
                        break
            
            # Check if any arguments provided at all - check both param_dict and converted_param_dict
            if not param_dict or len(param_dict) == 0:
                logger.warning(f"No parameters found in tool call XML for: {function_name}")
                return None
            
            if not converted_param_dict or len(converted_param_dict) == 0:
                logger.warning(f"Empty argument dict detected after conversion for tool call: {function_name}")
                return None
            
            # Check if all required parameters are present
            missing_required = []
            for param_name in required_params:
                if param_name not in converted_param_dict:
                    missing_required.append(param_name)
            
            if missing_required:
                logger.warning(f"Tool '{function_name}' is missing required parameters: {missing_required}. Converted params available: {list(converted_param_dict.keys())}")
                return None
            
            # Check for extra properties not in schema (if additionalProperties is False)
            if not additional_properties and len(properties) > 0:
                extra_props = []
                for key in converted_param_dict.keys():
                    if key not in properties:
                        extra_props.append(key)
                
                if extra_props:
                    logger.warning(f"Tool '{function_name}' has extra properties not in schema: {extra_props}. Allowed properties: {list(properties.keys())}")
                    return None
            
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
        
        # If no delta text, return structured message instead of None
        if not delta_text:
            # Check if this is an EOS token after all tool calls are complete
            # We check for tool calls in the text even if is_tool_call_started is False
            # because it might have been reset after processing all tools
            if delta_token_ids and self.tool_call_end_token_id not in delta_token_ids:
                # Count complete tool calls
                complete_calls = len(
                    self.tool_call_complete_regex.findall(current_text))

                # If we have completed tool calls and populated prev_tool_call_arr
                if complete_calls > 0 and len(self.prev_tool_call_arr) > 0:
                    # Check if all tool calls are closed
                    open_calls = current_text.count(
                        self.tool_call_start_token) - current_text.count(
                            self.tool_call_end_token)
                    if open_calls == 0:
                        # Return empty delta message with vLLM format
                        return ChatCompletionMessage(
                            content="",
                            role="assistant", 
                            name=None,
                            tool_calls=[],
                            tool_call_id=None
                        )
                elif not self.is_tool_call_started and current_text:
                    # This is a regular content response that's now complete - use vLLM format
                    return ChatCompletionMessage(
                        content="",
                        role="assistant", 
                        name=None,
                        tool_calls=[],
                        tool_call_id=None
                    )
                
            # When tool call has started but we haven't extracted function info yet,
            # and no other conditions matched, return structured message with empty tool_calls
            if self.is_tool_call_started and not self.current_function_name:
                return ChatCompletionMessage(
                    content="",
                    role="assistant",
                    name=None,
                    tool_calls=[],
                    tool_call_id=self.current_tool_id if self.current_tool_id else None
                )
            
            # Always return structured message, never None
            return ChatCompletionMessage(
                content="",
                role="assistant",
                name=None,
                tool_calls=[],
                tool_call_id=self.current_tool_id if self.current_tool_id else None
            )

        # Update accumulated text
        self.accumulated_text = current_text

        # Check if we need to advance to next tool
        if self.json_closed and not self.in_function:
            # Check if this tool call has ended
            tool_ends = current_text.count(self.tool_call_end_token)
            if tool_ends > self.current_tool_index:
                # This tool has ended, advance to next
                self.current_tool_index += 1
                self.header_sent = False
                self.param_count = 0
                self.json_started = False
                self.json_closed = False
                self.accumulated_params = {}

                # Check if there are more tool calls
                tool_starts = current_text.count(self.tool_call_start_token)
                if self.current_tool_index >= tool_starts:
                    # No more tool calls
                    self.is_tool_call_started = False
                # Continue processing next tool - return structured message
                return ChatCompletionMessage(
                    content="",
                    role="assistant",
                    name=None,
                    tool_calls=[],
                    tool_call_id=self.current_tool_id if self.current_tool_id else None
                )
                
            # When tool call has started but we haven't extracted function info yet,
            # and no other conditions matched, try to extract function information first
            if self.is_tool_call_started and not self.current_function_name:
                # Check accumulated text for function information (might span multiple deltas)
                if "function" in current_text.lower():
                    # Try to extract function name from the accumulated text
                    func_match = re.search(r'<function=(.*?)>', current_text, re.IGNORECASE)
                    if func_match:
                        self.current_function_name = func_match.group(1).strip()
                        # Only generate tool ID once when function name is first detected
                        if not self.current_tool_id:
                            self.current_tool_id = self._generate_tool_call_id()
                        return ChatCompletionMessage(
                            content="",
                            role="assistant", 
                            name=None,
                            tool_calls=[ChatToolCall(
                                type="function",
                                id=str(self.current_tool_id),
                                index=0,
                                function=ChatFunctionCall(name=self.current_function_name, arguments="")
                            )],
                            tool_call_id=None
                        )
                    
                    # Also try matching without closing > for partial output in accumulated text
                    func_match = re.search(r'<function=(.*)$', current_text, re.IGNORECASE)
                    if func_match:
                        self.current_function_name = func_match.group(1).strip()
                        # Only generate tool ID once when function name is first detected
                        if not self.current_tool_id:
                            self.current_tool_id = self._generate_tool_call_id()
                        return ChatCompletionMessage(
                            content="",
                            role="assistant", 
                            name=None,
                            tool_calls=[ChatToolCall(
                                type="function",
                                id=str(self.current_tool_id),
                                index=0,
                                function=ChatFunctionCall(name=self.current_function_name, arguments="")
                            )],
                            tool_call_id=None
                        )
                
                # Return any content before the tool call with proper structure
                if self.tool_call_start_token in delta_text:
                    content_before = delta_text[:delta_text.index(
                        self.tool_call_start_token)]
                    if content_before and not self.current_function_name:
                        # No function name extracted yet, return empty tool_calls
                        return ChatCompletionMessage(
                            content="", 
                            role="assistant", 
                            name=None,
                            tool_calls=[],
                            tool_call_id=None
                        )
                
                # When we detect tool call start but no content/function info yet,
                # still return empty tool_calls to signal structure, not None
                return ChatCompletionMessage(
                    content="",
                    role="assistant", 
                    name=None,
                    tool_calls=[],
                    tool_call_id=None
                )
            else:
                # Check if we're between tool calls - skip whitespace
                if current_text.rstrip().endswith(self.tool_call_end_token):
                    # We just ended a tool call, skip whitespace
                    if delta_text.strip() == "":
                        return ChatCompletionMessage(
                            content="",
                            role="assistant", 
                            name=None,
                            tool_calls=[],
                            tool_call_id=self.current_tool_id if self.current_tool_id else None
                        )
                # When tool call has started but we haven't extracted function info yet,
                # and we get whitespace/empty content, return structured message with empty tool_calls
                if self.is_tool_call_started and not self.current_function_name:
                    return ChatCompletionMessage(
                        content="",
                        role="assistant",
                        name=None,
                        tool_calls=[],
                        tool_call_id=self.current_tool_id if self.current_tool_id else None
                    )
                
                # Normal content, no tool call - but return in vLLM format
                if delta_text.strip():
                    # Only send as tool call if we have a function name and ID
                    if self.current_function_name and self.current_tool_id:
                        return ChatCompletionMessage(
                            role="assistant", 
                            name=None,
                            tool_calls=[ChatToolCall(
                                type="function",
                                id=str(self.current_tool_id),
                                index=0,
                                function=ChatFunctionCall(name=self.current_function_name, arguments="")
                            )],
                            tool_call_id=None
                        )
                
# Check if this is just the opening tag with no content yet
            if delta_text.strip() == "<tool_call>" and not self.is_tool_call_started:
                # Return empty structured message to indicate tool call started
                return ChatCompletionMessage(
                    role="assistant", 
                    name=None,
                    tool_calls=[],
                    tool_call_id=None
                )
            elif delta_text.strip() == "" and current_text.startswith(self.tool_call_start_token):
                # Return empty structured message for whitespace in tool call
                return ChatCompletionMessage(
                    role="assistant", 
                    name=None,
                    tool_calls=[],
                    tool_call_id=None
                )
            
            return ChatCompletionMessage(
                content="",
                role="assistant",
                name=None,
                tool_calls=[],
                tool_call_id=self.current_tool_id if self.current_tool_id else None
            )
        
        # Check if we're between tool calls (waiting for next one)
        # Count tool calls we've seen vs processed
        tool_starts_count = current_text.count(self.tool_call_start_token)
        if self.current_tool_index >= tool_starts_count:
            # We're past all tool calls, shouldn't be here
            return ChatCompletionMessage(
                content="",
                role="assistant",
                name=None,
                tool_calls=[],
                tool_call_id=self.current_tool_id if self.current_tool_id else None
            )
        # Need to find the correct tool call based on current_tool_index
        tool_starts = []
        idx = 0
        while True:
            idx = current_text.find(self.tool_call_start_token, idx)
            if idx == -1:
                break
            tool_starts.append(idx)
            idx += len(self.tool_call_start_token)

        if self.current_tool_index >= len(tool_starts):
            # No more tool calls to process yet
            return ChatCompletionMessage(
                content="",
                role="assistant",
                name=None,
                tool_calls=[],
                tool_call_id=self.current_tool_id if self.current_tool_id else None
            )

        tool_start_idx = tool_starts[self.current_tool_index]
        # Find where this tool call ends (or current position if not ended yet)
        tool_end_idx = current_text.find(self.tool_call_end_token,
                                         tool_start_idx)
        if tool_end_idx == -1:
            tool_text = current_text[tool_start_idx:]
        else:
            tool_text = current_text[tool_start_idx:tool_end_idx +
                                     len(self.tool_call_end_token)]

        # Looking for function header
        if not self.header_sent:
            if self.tool_call_prefix in tool_text:
                func_start = tool_text.find(self.tool_call_prefix) + len(
                    self.tool_call_prefix)
                func_end = tool_text.find(">", func_start)

                if func_end != -1:
                    # Found complete function name
                    self.current_function_name = tool_text[func_start:func_end]
                    self.current_tool_id = self._generate_tool_call_id()
                    self.header_sent = True
                    self.in_function = True

                    # IMPORTANT: Add to prev_tool_call_arr immediately when we detect a tool call
                    # This ensures finish_reason="tool_calls" even if parsing isn't complete
                    already_added = any(
                        tool.get("name") == self.current_function_name
                        for tool in self.prev_tool_call_arr)
                    if not already_added:
                        self.prev_tool_call_arr.append({
                            "name": self.current_function_name,
                            "arguments":
                            "{}",  # Placeholder, will be updated later
                        })

                        # Send header with function info - include name immediately (LM Studio format)
                    return ChatCompletionMessage(
                        content="",
                        role="assistant", 
                        name=None,
                        tool_calls=[ChatToolCall(
                            type="function",
                            id=str(self.current_tool_id),
                            index=0,  # Always use index 0 for first tool call
                            function=ChatFunctionCall(name=self.current_function_name, arguments="{}")
                        )],
                        tool_call_id=None
                    )
            
            # Return empty message to maintain structure even when no action taken
            return ChatCompletionMessage(
                content="",
                role="assistant",
                tool_calls=[],
                tool_call_id=self.current_tool_id if self.current_tool_id else None
            )

        # We've sent header, now handle function body
        if self.in_function:
            # Send function name with empty arguments initially (LM Studio format)
            if not self.json_started and self.parameter_prefix in current_text:
                self.json_started = True
                return ChatCompletionMessage(
                    content="",
                    role="assistant", 
                    name=None,
                    tool_calls=[ChatToolCall(
                        type="function",
                        id=str(self.current_tool_id),
                        index=0,  # Always use index 0 for first (and only) tool call
                        function=ChatFunctionCall(name=self.current_function_name, arguments="{}")
                    )],
                    tool_call_id=None
                )

            # Check for parameter updates in delta text using the proper regex
            param_match = self.tool_call_parameter_regex.search(delta_text)
            if param_match and not self.in_param:
                # Found new parameter opening - don't send anything yet, just note we're in a parameter
                self.in_param = True
                self.param_count += 1
                param_text = param_match.group(0)
                eq_pos = param_text.find("=")
                if eq_pos != -1:
                    self.current_param_name = param_text[eq_pos+1:].split(">", 1)[0]
                return ChatCompletionMessage(
                    content="",
                    role="assistant",
                    name=None,
                    tool_calls=[],
                    tool_call_id=self.current_tool_id if self.current_tool_id else None
                )
            elif self.in_param:
                # Accumulate parameter value for JSON building
                self.current_param_value += delta_text.strip()
                
                # Check for closing tag to finalize the parameter and build complete JSON
                if "</parameter>" in tool_text:
                    self.in_param = False
                    self.current_param_value = ""
                    return ChatCompletionMessage(
                        content="",
                        role="assistant", 
                        name=None,
                        tool_calls=[ChatToolCall(
                            type="function",
                            id=str(self.current_tool_id),
                            index=0,  # Always use index 0 for first (and only) tool call
                            function=ChatFunctionCall(name=self.current_function_name, arguments="")
                        )],
                        tool_call_id=None
                    )
                # Return incremental updates during parameter accumulation
                return ChatCompletionMessage(
                    content="",
                    role="assistant",
                    name=None,
                    tool_calls=[ChatToolCall(
                        type="function",
                        id=str(self.current_tool_id),
                        index=0,
                        function=ChatFunctionCall(name=self.current_function_name, arguments="")
                    )],
                    tool_call_id=None
                )

            # Check for function end in accumulated text
            if not self.json_closed and self.function_end_token in tool_text:
                # Close JSON
                self.json_closed = True

                # Extract the complete tool call to update prev_tool_call_arr with final arguments
                # Find the function content
                func_start = tool_text.find(self.tool_call_prefix) + len(
                    self.tool_call_prefix)
                func_content_end = tool_text.find(self.function_end_token,
                                                  func_start)
                if func_content_end != -1:
                    func_content = tool_text[func_start:func_content_end]
                    # Parse to get the complete arguments
                    try:
                        parsed_tool = self._parse_tool_call(
                            func_content, self.streaming_request.tools
                            if self.streaming_request else None)
                        
                        # Only create tool call response if parsing succeeded and validation passed
                        if parsed_tool:
                            # Update existing entry in prev_tool_call_arr with complete arguments
                            for i, tool in enumerate(self.prev_tool_call_arr):
                                if tool.get("name") == parsed_tool.function.name:
                                    self.prev_tool_call_arr[i]["arguments"] = parsed_tool.function.arguments
                                    break
                        else:
                            # Parsing failed or validation rejected - don't return a tool call
                            logger.debug(f"Streaming tool call parsing failed for {self.current_function_name}, not creating response")
                            # Return empty message to maintain structure but without tool_calls
                            return ChatCompletionMessage(
                                content="",
                                role="assistant", 
                                name=None,
                                tool_calls=[]
                            )
                    except Exception as e:
                        logger.warning(f"Exception during streaming tool call parsing: {e}")
                        # Return empty message on exception
                        return ChatCompletionMessage(
                            content="",
                            role="assistant", 
                            name=None,
                            tool_calls=[]
                        )

                # Close JSON - don't add literal '}' to arguments
                if parsed_tool:
                    return ChatCompletionMessage(
                        content="",
                        role="assistant", 
                        name=None,
                        tool_calls=[ChatToolCall(
                            type="function",
                            id=str(self.current_tool_id),
                            index=0,
                            function=ChatFunctionCall(name=self.current_function_name, arguments=parsed_tool.function.arguments)
                        )],
                        tool_call_id=None
                    )
                else:
                    # Should not reach here if we handled parsed_tool == None above, but be safe
                    return ChatCompletionMessage(
                        content="",
                        role="assistant", 
                        name=None,
                        tool_calls=[]
                    )
            
        # Return empty message to maintain structure when no tool call detected
        return ChatCompletionMessage(
            content="",
            role="assistant", 
            name=None,
            tool_calls=[],
            tool_call_id=self.current_tool_id if self.current_tool_id else None
        )
