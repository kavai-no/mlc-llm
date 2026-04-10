import pytest
from typing import Any
from mlc_llm.protocol.conversation_protocol import BaseToolParser, Conversation

class MockFragmentedParser(BaseToolParser):
    """A simple parser for testing fragmented XML strings."""
    def parse(self, text: str) -> Any:
        if "<tool_call>" in text and "</tool_call>" in text:
            return {"type": "tool_call", "status": "complete"}
        return {"type": "incomplete"}

    def parse_streaming(self, chunk: str) -> Any:
        # In a real scenario, this would maintain state. 
        # For this test, we just check if it receives the chunk.
        if chunk == "<tool_call>":
            return "received_start"
        return "received_other"

def test_protocol_holds_parser_name():
    """Verify that the Conversation object holds the parser name (string)."""
    conv = Conversation(
        roles={"user": "{user_message}", "assistant": "{assistant_message}"},
        seps=["<sep>", "<sep>"],
        tool_parser="qwen3_coder"  # Using the registered string name
    )
    assert conv.tool_parser == "qwen3_coder"

def test_protocol_hydration_logic():
    """Verify that loading from JSON hydratess the parser instance."""
    json_data = {
        "roles": {"user": "{user_message}", "assistant": "{assistant_message}"},
        "seps": ["<sep>", "<sep>"],
        "tool_parser": "qwen3_coder"
    }
    conv = Conversation.from_json_dict(json_data)
    assert conv.tool_parser == "qwen3_coder"
    # Check if the hydrated instance is actually present and functional
    assert conv.tool_parser_instance is not None
    assert hasattr(conv.tool_parser_instance, 'parse')
    
    # Test the parser logic through the hydrated instance
    result = conv.tool_parser_instance.parse("<tool_call>...</tool_call>")
    # Note: Qwen3CoderToolParser returns a tuple (content, tool_calls) 
    # based on its implementation in tool_parser.py
    assert isinstance(result, tuple)

def test_fragmented_parsing_logic():
    """Verify the logic for handling fragmented XML tokens."""
    parser = MockFragmentedParser()
    
    # Test complete call
    result = parser.parse("<tool_call><function name='test'></function></tool_call>")
    assert result["status"] == "complete"
    
    # Test incomplete call
    result = parser.parse("<tool_call><function")
    assert result["type"] == "incomplete"

def test_streaming_fragment_handling():
    """Verify the streaming method handles individual tokens."""
    parser = MockFragmentedParser()
    
    assert parser.parse_streaming("<tool_call>") == "received_start"
    assert parser.parse_streaming("random_token") == "received_other"

if __name__ == "__main__":
    pytest.main([__file__])
