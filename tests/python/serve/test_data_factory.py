"""
Test data factory for Qwen3CoderToolParser tests.
Provides a fluent interface for creating properly formatted XML tool calls.
"""

from mlc_llm.protocol.openai_api_protocol import ChatTool, ChatFunction


class XmlToolCallBuilder:
    """Fluent builder for creating well-formed Qwen3Coder XML tool call strings."""
    
    def __init__(self):
        self.function_name = None
        self.parameters = {}
    
    def with_function(self, name: str) -> 'XmlToolCallBuilder':
        """Set the function name for this tool call."""
        self.function_name = name
        return self
    
    def add_parameter(self, name: str, value: str) -> 'XmlToolCallBuilder':
        """Add a parameter to the tool call."""
        self.parameters[name] = value
        return self
    
    def build(self) -> str:
        """Build the complete XML string with proper newlines."""
        if not self.function_name:
            raise ValueError("Function name must be set")
        
        parts = [
            "<tool_call>",
            f"<function={self.function_name}>"
        ]
        
        for param_name, param_value in self.parameters.items():
            parts.append(f"<parameter={param_name}>")
            parts.append(str(param_value))
            parts.append("</parameter>")
        
        parts.extend([
            "</function>",
            "</tool_call>"
        ])
        
        return '\n'.join(parts)


def create_search_web_tool() -> ChatTool:
    """Create a standard search_web tool definition for testing."""
    return ChatTool(
        type="function",
        function=ChatFunction(
            name="search_web",
            description="Search the web.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "count": {"type": "integer"}
                },
                "required": ["query"],
                "additionalProperties": False
            }
        )
    )


def create_get_current_timestamp_tool() -> ChatTool:
    """Create a parameterless get_current_timestamp tool definition."""
    return ChatTool(
        type="function",
        function=ChatFunction(
            name="get_current_timestamp",
            description="Get the current timestamp.",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    )


# Pre-built XML examples for common test cases
SINGLE_PARAM_XML = """<tool_call>
<function=search_web>
<parameter=query>test</parameter>
</function>
</tool_call>"""

MULTI_PARAM_XML = """<tool_call>
<function=search_web>
<parameter=query>test query</parameter>
<parameter=count>10</parameter>
</function>
</tool_call>"""

EMPTY_PARAMS_XML = """<tool_call>
<function=get_current_timestamp>
</function>
</tool_call>"""

MALFORMED_XML_MINIMAL = "<tool_call><function=search_web><parameter=query>test"
MALFORMED_XML_UNCLOSED_TAG = "<tool_call><function=search_web><parameter=query>test</parameter></function>"
INCOMPLETE_TOOL_CALL = "<tool_call>"
