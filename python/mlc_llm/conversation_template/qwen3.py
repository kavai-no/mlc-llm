"""Qwen3 default templates"""

from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders

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
        stop_str=["</s>", "<|im_end|>"],
        stop_token_ids=[151643, 151645],
    )
)

# Qwen3 coder conversation template with tool calling support
ConvTemplateRegistry.register_conv_template(
    Conversation(
        name="qwen3_coder",
        system_template=f"<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}<|im_end|>\n",
        system_message="""<|im_start|>system\n{system_message}\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{function_string}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>""",
        roles={"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"},
        seps=["<|im_end|>\n"],
        role_content_sep="\n",
        role_empty_sep="\n",
        stop_str=["</s>", "<|im_end|>"],
        stop_token_ids=[151643, 151645],
        tool_parser="qwen3_coder"
    )
)