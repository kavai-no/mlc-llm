"""Qwen3 default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders
from textwrap import dedent
from .registry import ConvTemplateRegistry

# Qwen3 conversation template - similar to qwen2 but with updated stop tokens and system message
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="qwen3",
        system_template=f"<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}<|im_end|>\n",
        system_message="You are a helpful assistant.",
        roles={"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"},
        seps=["<|im_end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|endoftext|>", "<|im_end|>"],
        stop_token_ids=[151643, 151645],
    )
)

# Qwen3 coder conversation template with tool calling support
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="qwen3_coder",
        system_template=dedent(f"""<|im_start|>system
{MessagePlaceholders.SYSTEM.value}

# Tools

You have access to the following functions:

<tools>
{MessagePlaceholders.FUNCTION.value}
</tools>

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
</IMPORTANT>
<|im_end|>
""").strip(),
        system_message="You are a helpful assistant.",
        role_templates={
            "assistant": f"<|im_start|>user\n{MessagePlaceholders.ASSISTANT.value}\n<|im_end|>\n",
            "user": f"<|im_start|>user\n{MessagePlaceholders.USER.value}\n<|im_end|>\n",
            "tool": f"<|im_start|>user\n<tool_response>\n{MessagePlaceholders.TOOL.value}\n</tool_reponse><|im_end|>\n"
        },
        roles={"user": "", "assistant": "", "tool": ""},
        seps=["<|im_end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["<|endoftext|>", "<|im_end|>"],
        stop_token_ids=[151643, 151645],
        tool_parser="qwen3_coder"
    )
)
