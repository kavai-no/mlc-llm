"""
Qwen3.5 conversation template with tool call support.

This template uses the Qwen3CoderToolCallParser to handle tool calls in XML-style format:
    <tool_call>
    <function=function_name>
    <parameter=param_name>value</parameter>
    </function>
    </tool_call>
"""

from typing import Any, Dict, List, Optional

# Import the Conversation class and register the parser
from mlc_llm.protocol.conversation_protocol import Conversation

def create_qwen3_5_conversation() -> Conversation:
    """
    Create a Qwen3.5 conversation template with tool call support.
    
    Returns:
        A Conversation object representing the conversation template.
    """
    return Conversation(
        name="qwen3_5",
        system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
        system_message="You are a helpful assistant.",
        roles={
            "user": "<|im_start|>user",
            "assistant": "<|im_start|>assistant\n<think>",
        },
        seps=["<|im_end|>\n"],
        stop_str=["<|endoftext|>", "<|im_end|>"],
        stop_token_ids=[248046, 248044],
        tool_parser="qwen3_coder",  # Use string-based parser name for hydration
    )


if __name__ == "__main__":
    # Example usage
    conversation = create_qwen3_5_conversation()
    print("Qwen3.5 Conversation Template:", conversation)