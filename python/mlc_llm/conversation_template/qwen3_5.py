from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders
from typing import Any, Dict, List, Optional

class Qwen3_5_Template(Conversation):
    """
    Template for Qwen 3.5 models using XML-style tool calling.
    Format: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
    """

    def __init__(self, system_message: str = ""):
        role_templates = {
            "user": f"{MessagePlaceholders.FUNCTION.value}{{user_message}}",
            "assistant": "{assistant_message}",
            "tool": "{tool_message}",
        }
        
        # The function string is the template for the tool call XML structure.
        # We use the placeholder from our protocol to allow runtime injection if needed, 
        # but here we define the structural pattern.
        function_string = "<tool_call><function={function_name}><parameter={param_name}>{param_value}</parameter></function></tool_call>"

        super().__init__(
            roles={"user": "user", "assistant": "assistant", "tool": "tool"},
            role_templates=role_templates,
            system_message=system_message,
            seps=["<|im_start|>user\n", "<|im_end|>\n"], # Using ChatML style as base for Qwen
            role_content_sep="",
            role_empty_sep="",
            function_string=function_string,
        )

    def _render_tools_xml(self) -> str:
        """Render the list of available tools in XML format for the system prompt."""
        if not self.tools:
            return ""
        
        tool_xmls = []
        for tool in self.tools:
            name = tool.get("name", "unknown")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})
            
            param_xmls = []
            if "properties" in parameters:
                for param_name, param_info in parameters["properties"].items():
                    param_type = param_info.get("type", "string")
                    param_desc = param_info.get("description", "")
                    
                    param_xmls.append(f"<parameter><name>{param_name}</name>")
                    param_xmls.append(f"<type>{param_type}</type>")
                    if param_desc:
                        param_xmls.append(f"<description>{param_desc}</description>")
                    param_xmls.append("</parameter>")
            
            required_params = parameters.get("required", [])
            required_xml = ""
            if required_params:
                required_xml = f"<required>[{', '.join(required_params)}]</required>"
            
            tool_xmls.append(f"<function><name>{name}</name>")
            if description:
                tool_xmls.append(f"<description>{description}</description>")
            tool_xmls.append("<parameters>")
            tool_xmls.extend(param_xmls)
            tool_xmls.append(required_xml)
            tool_xmls.append("</parameters>")
            tool_xmls.append("</function>")
        
        return "<tools>" + "".join(tool_xmls) + "</tools>"

    def as_prompt(self, config=None) -> List[Any]:
        # Note: The actual logic for prompt construction is inherited from Conversation.as_prompt.
        # We just need to ensure the templates are populated correctly.
        return super().as_prompt(config)
