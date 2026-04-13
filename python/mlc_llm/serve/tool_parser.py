"""
Qwen3-Coder tool call parser.

Format uses XML-style nested tags:
    <tool_call>
    <function=function_name>
    <parameter=param_name>value</parameter>
    <parameter=param_name2>value2</parameter>
    </function>
    </tool_call>

Parameters are extracted from <parameter=name>value</parameter> tags and
type-converted using the schema if available, otherwise treated as strings.

Based on VLLM's Qwen3CoderToolParser.extract_tool_calls()
"""

import ast
import json
import re
import uuid
from typing import Any, Dict, List, Optional, Callable, TypeVar
from mlc_llm.protocol.openai_api_protocol import (
    ChatToolCall,
    ChatFunctionCall,
)
from mlc_llm.protocol.conversation_protocol import BaseToolParser

T = TypeVar('T')

# Global registry for parser lookup during hydration
PARSER_REGISTRY: Dict[str, Any] = {}

def register_parser(name: str) -> Callable[[T], T]:
    """Decorator to register a parser with the enough name."""
    def decorator(cls: T) -> T:
        PARSER_REGISTRY[name] = cls
        return cls
    return decorator

def get_parser_instance(name: str) -> Any:
    """Retrieve a parser instance from the registry."""
    parser_cls = PARSER_REGISTRY.get(name)
    if parser_cls:
        try:
            return parser_cls()
        except Exception:
            return None
    return None

def _try_convert_value(value: str) -> Any:
    """
    Try to convert a parameter value string to a native Python type.
    Handles null, JSON objects/arrays, numbers, booleans, and falls back to string.
    Uses conservative approach: only converts when clearly numeric/boolean/JSON.
    """
    stripped = value.strip()

    # Handle null - only convert if the entire value is exactly 'null' (case insensitive)
    if stripped.lower() == "null":
        return None

    # Try JSON first, but only for objects, arrays, strings, booleans (not bare numbers)
    try:
        result = json.loads(stripped)
        # Only accept conversion if it's a complex type or boolean
        # Don't convert bare numbers to keep IDs and other string-like values as strings
        if isinstance(result, (dict, list, str, bool)):
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Try un-escaping double braces for JSON objects that might be escaped in XML context
    try:
        if stripped.startswith('{') and stripped.endswith('}'):
            # Handle case like {{"role": "admin"}} -> {"role": "admin"}
            unescaped = stripped.replace('{{', '{').replace('}}', '}')
            result = json.loads(unescaped)
            if isinstance(result, (dict, list)):
                return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to convert numbers - integers and floats
    try:
        if re.match(r'^\s*[-+]?\d+\.\d+\s*$', stripped):
            return float(stripped)
        elif re.match(r'^\s*-?\d+\s*$', stripped) and (stripped.startswith('-') or '+' in stripped or len(stripped) > 1):
            # Convert negative integers to int, also convert multi-digit positive integers
            return int(stripped)
    except (ValueError, AttributeError):
        pass

    # For non-JSON content or bare numbers/booleans, be conservative and keep as string
    # This matches the behavior described in the docstring: "otherwise treated as strings"
    return stripped


# Define ParseResult type
ParseResult = tuple[str, List[ChatToolCall]]  # (content: str, tool_calls: List[ChatToolCall])

@register_parser("qwen3_coder")
class Qwen3CoderToolCallParser(BaseToolParser):
    """
    Parser for Qwen3-Coder XML-format tool calls.

    Uses nested XML tags: <tool_call><function=name><parameter=key>val</parameter></function></tool_call>
    """

    START_TOKEN="***"
    FUNCTION_PREFIX = "<function="

    # Find complete tool_call blocks (or unclosed at end)
    TOOL_CALL_REGEX = re.compile(
        r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
    )

    # Find function blocks within a tool_call
    FUNCTION_REGEX = re.compile(
        r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
    )

    # Find parameter blocks within a function
    PARAMETER_REGEX = re.compile(
        r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
        re.DOTALL,
    )

    def __init__(self):
        super().__init__()
        self._streaming_buffer = ""

    def _parse_function_call(self, function_str: str) -> Optional[ChatToolCall]:
        """Parse a single <function=name>...</function> block into a ChatToolCall."""
        try:
            # Extract function name: everything before the first '>'
            gt_idx = function_str.index(">")
            func_name = function_str[:gt_idx].strip()
            params_str = function_str[gt_idx + 1:]

            # Check if parameters are missing or malformed
            if not self.PARAMETER_REGEX.search(params_str):
                return None

            # Extract parameters
            param_dict: Dict[str, Any] = {}
            for match_text in self.PARAMETER_REGEX.findall(params_str):
                if ">" not in match_text:
                    continue
                eq_idx = match_text.index(">")
                param_name = match_text[:eq_idx].strip()
                # Handle the closing tag if it's present in the capture group
                param_value = match_text[eq_idx + 1:]
                if "</parameter>" in param_value:
                    param_value = param_value.split("</parameter>")[0]

                # Clean up whitespace
                param_value = param_value.strip()

                # Special handling for 'id' parameters - keep as strings even if numeric
                if param_name == "id":
                    param_dict[param_name] = param_value
                else:
                    param_dict[param_name] = _try_convert_value(param_value)

            return ChatToolCall(
                id=f"call_{uuid.uuid4().hex[:24]}",
                type="function",
                function=ChatFunctionCall(
                    name=func_name,
                    arguments=param_dict,  # Pass the dictionary directly
                ),
            )
        except (ValueError, IndexError):
            return None

    def parse(self, text: str) -> ParseResult:
        if not text.strip():
            return "", []

        # Find where the tool call starts (either <tool_call> or <function=)
        first_tc_idx = text.find("<tool_call>")
        if first_tc_idx < 0:
            first_tc_idx = text.find(self.FUNCTION_PREFIX)
            
        # If no tool call markers found, the entire text is content
        if first_tc_idx == -1:
            return text, []

        # Content is everything BEFORE the actual tool call start
        content = text[:first_tc_idx].strip() if first_tc_idx > 0 else ""

        try:
            # Find all tool_call blocks (including unclosed tags)
            tc_matches = self.TOOL_CALL_REGEX.findall(text)
            raw_blocks = [m[0] or m[1] for m in tc_matches if m[0] or m[1]]

            if not raw_blocks:
                return content, []

            # Find function blocks within each tool_call (including unclosed tags)
            function_strs: List[str] = []
            for block in raw_blocks:
                func_matches = self.FUNCTION_REGEX.findall(block)
                function_strs.extend(m[0] or m[1] for m in func_matches if m[0] or m[1])

            if not function_strs:
                return content, []

            # Parse each function call
            tool_calls: List[ChatToolCall] = []
            for func_str in function_strs:
                tc = self._parse_function_call(func_str)
                if tc is not None:
                    tool_calls.append(tc)

            return content, tool_calls

        except Exception:
            return content, []

    def parse_streaming(self, chunk: str) -> Any:
        """
        Method for incremental parsing of streaming tokens.
        
        Maintains internal buffer state to handle partial XML tags across chunks.
        Returns:
            - None if no tool call content detected
            - {"type": "partial_tool_call", "data": str} for incomplete tool calls
            - {"type": "complete_tool_call", "data": List[ChatToolCall]} for complete tool calls
        """
        if not chunk:
            return None
        
        # Add the new chunk to our buffer
        self._streaming_buffer += chunk
        
        # Check if we have any tool call content or partial tags
        has_tool_content = ("<tool_call>" in self._streaming_buffer or 
                           self.FUNCTION_PREFIX in self._streaming_buffer or
                           "<tool" in self._streaming_buffer)
        if not has_tool_content:
            return None
        
        # Try to find complete tool calls in the buffer
        tc_matches = self.TOOL_CALL_REGEX.findall(self._streaming_buffer)
        raw_blocks = [m[0] or m[1] for m in tc_matches if m[0] or m[1]]
        
        if not raw_blocks:
            # No complete tool calls yet, return partial
            return {"type": "partial_tool_call", "data": self._streaming_buffer}
        
        # We found at least one complete tool call block
        # Now we need to check if it's actually complete (has closing tags)
        function_strs: List[str] = []
        for block in raw_blocks:
            # Use findall to get capture groups, same as parse method
            func_matches = self.FUNCTION_REGEX.findall(block)
            function_strs.extend(m[0] or m[1] for m in func_matches if m[0] or m[1])
        
        if not function_strs:
            return {"type": "partial_tool_call", "data": self._streaming_buffer}
        
        # Check if we have complete tool calls (with closing tags)
        # A tool call is complete only if it has </tool_call>
        # We need to check the original buffer, not just extracted content
        complete_tool_calls: List[ChatToolCall] = []
        
        # Check if there's a complete tool call in the original buffer
        if "</tool_call>" in self._streaming_buffer:
            # Find the last opening and its matching closing tag
            # This handles cases where there are multiple <tool_call> openings
            # but only one </tool_call> closing (the complete one)
            
            # Get all tool call positions
            import re as regex_module
            tc_openings = [m.start() for m in regex_module.finditer(r'<tool_call>', self._streaming_buffer)]
            tc_closings = [m.end() for m in regex_module.finditer(r'</tool_call>', self._streaming_buffer)]
            
            if tc_openings and tc_closings:
                # The complete tool call is from the LAST opening to the last closing
                # This handles cases like: <tool_call>...<tool_call>...complete...</tool_call>
                last_opening = tc_openings[-1]
                last_closing_end = tc_closings[-1]  # end() gives position after tag
                
                complete_block = self._streaming_buffer[last_opening:last_closing_end]
                
                # Extract functions from the complete block
                func_matches = self.FUNCTION_REGEX.findall(complete_block)
                function_strs_in_complete = [m[0] or m[1] for m in func_matches if m[0] or m[1]]
                
                # Process functions from the complete block
                for func_str in function_strs_in_complete:
                    tc = self._parse_function_call(func_str)
                    if tc is not None:
                        complete_tool_calls.append(tc)
                        break  # Stop after first complete tool call
        
        # If we have complete tool calls with proper closing tags, return them
        if complete_tool_calls:
            # Clear the buffer since we've processed complete tool calls
            self._streaming_buffer = ""
            return {"type": "complete_tool_call", "data": complete_tool_calls}
        
        # Otherwise still partial (missing closing tags)
        return {"type": "partial_tool_call", "data": self._streaming_buffer}
    
    def render_tool_call_result(self, content: str) -> str:
        # Follow the official Qwen3-Coder chat template format
        return f"<tool_response>\n{content}\n</tool_response>\n"
    
    def render_tool_call(self, tool_call_item: Dict[str, Any]) -> str:
        """
        Render a tool call item to XML format for Qwen3.
        
        Parameters
        ----------
        tool_call_item : Dict[str, Any]
            Tool call item with structure: {"type": "tool_call", "name": str, "parameters": dict}
            
        Returns
        -------
        str
            XML-formatted tool call string
        """
        tool_name = tool_call_item.get("name", "unknown")
        parameters = tool_call_item.get("parameters", {})
        
        # Build parameter tags with proper newlines
        param_tags = []
        for param_name, param_value in parameters.items():
            param_tags.append(f"<parameter={param_name}>\n{param_value}\n</parameter>")
        
        if not param_tags:
            return f"<tool_call>\n<function={tool_name}></function>\n</tool_call>"
        
        # Build complete tool call XML with proper structure
        params_xml = "\n".join(param_tags)
        return f"<tool_call>\n<function={tool_name}>\n{params_xml}\n</function>\n</tool_call>"
    
    def render_tools(self, tools: List[Dict[str, Any]]) -> str:
        """
        Render a list of tools to XML format for Qwen3 system prompt.
        
        This follows the Qwen3-Coder format shown in the official chat template.
        
        Parameters
        ----------
        tools : List[Dict[str, Any]]
            List of tool definitions with structure matching OpenAI API format:
            {"type": "function", "function": {"name": str, "description": str, "parameters": {...}}}
            
        Returns
        -------
        str
            XML-formatted tools list for system prompt
        """
        if not tools:
            return ""
        
        tool_xmls = []
        for tool in tools:
            # Extract function definition from tool
            if isinstance(tool, dict):
                func_def = tool.get("function", {})
                if not func_def:
                    continue
                
                name = func_def.get("name", "unknown_function")
                description = func_def.get("description", "")
                parameters = func_def.get("parameters", {})
                
                # Build function XML
                func_xml_parts = ["<function>", f"<name>{name}</name>"]
                
                if description:
                    func_xml_parts.append(f"<description>{description}</description>")
                
                # Add parameters if present
                if isinstance(parameters, dict) and "properties" in parameters:
                    props = parameters["properties"]
                    if isinstance(props, dict):
                        func_xml_parts.append("<parameters>")
                        for param_name, param_def in props.items():
                            param_type = param_def.get("type", "string")
                            param_desc = param_def.get("description", "")
                            
                            param_xml = f"<parameter>\n<name>{param_name}</name>\n<type>{param_type}</type>"
                            if param_desc:
                                param_xml += f"\n<description>{param_desc}</description>\n"
                            param_xml += "</parameter>"
                            
                            func_xml_parts.append(param_xml)
                        func_xml_parts.append("</parameters>")
                
                func_xml_parts.append("</function>")
                tool_xmls.append("\n".join(func_xml_parts))
        
        if not tool_xmls:
            return ""
        
        # Build complete tools section with instructional text
        tools_section = []
        tools_section.append("<tools>")
        tools_section.extend(tool_xmls)
        tools_section.append("</tools>")
        
        # Add instructional text about tool call format (from official template)
        tools_section.append("""
If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>""")
        
        return "\n".join(tools_section)
