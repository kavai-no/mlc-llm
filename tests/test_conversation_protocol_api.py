"""
API boundary tests for Conversation protocol tool_parser field and hydration mechanism.

These tests define the public API for conversation protocol extensions,
testing tool_parser field existence and functionality, tool_parser_instance
property and hydration, and serialization/deserialization cycle.
"""

import pytest
from mlc_llm.protocol.conversation_protocol import Conversation


class TestToolParserField:
    """Test the tool_parser field existence and basic functionality."""

    def test_tool_parser_field_exists(self):
        """Conversation class should have a tool_parser field."""
        conv = Conversation(
            roles={"user": "{user_message}", "assistant": "{assistant_message}"},
            role_templates={"user": "{user_message}", "assistant": "{assistant_message}"},
            seps=["\n"]
        )
        assert hasattr(conv, 'tool_parser')

    def test_tool_parser_field_is_none_by_default(self):
        """tool_parser field should be None when not specified."""
        conv = Conversation(
            roles={"user": "{user_message}", "assistant": "{assistant_message}"},
            role_templates={"user": "{user_message}", "assistant": "{assistant_message}"},
            seps=["\n"]
        )
        assert conv.tool_parser is None

    def test_tool_parser_field_can_be_set(self):
        """tool_parser field should accept string values."""
        conv = Conversation(
            roles={"user": "{user_message}", "assistant": "{assistant_message}"},
            role_templates={"user": "{user_message}", "assistant": "{assistant_message}"},
            seps=["\n"],
            tool_parser="qwen3_coder"
        )
        assert conv.tool_parser == "qwen3_coder"

    def test_tool_parser_field_accepts_custom_names(self):
        """tool_parser field should accept custom parser names."""
        conv = Conversation(
            roles={"user": "{user_message}", "assistant": "{assistant_message}"},
            role_templates={"user": "{user_message}", "assistant": "{assistant_message}"},
            seps=["\n"],
            tool_parser="custom_parser"
        )
        assert conv.tool_parser == "custom_parser"


class TestToolParserHydration:
    """Test the tool_parser_instance property and hydration mechanism."""

    def test_tool_parser_instance_field_exists(self):
        """Conversation class should have a tool_parser_instance field."""
        conv = Conversation(
            roles={"user": "{user_message}", "assistant": "{assistant_message}"},
            role_templates={"user": "{user_message}", "assistant": "{assistant_message}"},
            seps=["\n"]
        )
        assert hasattr(conv, 'tool_parser_instance')

    def test_tool_parser_instance_is_none_when_no_parser_specified(self):
        """tool_parser_instance should be None when no tool_parser is specified."""
        conv = Conversation(
            roles={"user": "{user_message}", "assistant": "{assistant_message}"},
            role_templates={"user": "{user_message}", "assistant": "{assistant_message}"},
            seps=["\n"]
        )
        assert conv.tool_parser_instance is None

    def test_tool_parser_hydration_with_registered_parser(self):
        """tool_parser_instance should be hydrated when tool_parser references a registered parser."""
        conv = Conversation(
            roles={"user": "{user_message}", "assistant": "{assistant_message}"},
            role_templates={"user": "{user_message}", "assistant": "{assistant_message}"},
            seps=["\n"],
            tool_parser="qwen3_coder"
        )
        # The hydration should happen automatically during initialization
        assert conv.tool_parser_instance is not None
        assert hasattr(conv.tool_parser_instance, 'parse')
        assert hasattr(conv.tool_parser_instance, 'parse_streaming')

    def test_tool_parser_hydration_with_unregistered_parser(self):
        """tool_parser_instance should remain None when tool_parser references an unregistered parser."""
        conv = Conversation(
            roles={"user": "{user_message}", "assistant": "{assistant_message}"},
            role_templates={"user": "{user_message}", "assistant": "{assistant_message}"},
            seps=["\n"],
            tool_parser="unregistered_parser"
        )
        assert conv.tool_parser_instance is None

    def test_tool_parser_hydration_with_empty_string(self):
        """tool_parser_instance should remain None when tool_parser is empty string."""
        conv = Conversation(
            roles={"user": "{user_message}", "assistant": "{assistant_message}"},
            role_templates={"user": "{user_message}", "assistant": "{assistant_message}"},
            seps=["\n"],
            tool_parser=""
        )
        assert conv.tool_parser_instance is None


class TestSerializationDeserialization:
    """Test serialization and deserialization cycle for Conversation with tool_parser."""

    def test_serialization_excludes_tool_parser_instance(self):
        """JSON serialization should exclude the hydrated tool_parser_instance."""
        conv = Conversation(
            roles={"user": "{user_message}", "assistant": "{assistant_message}"},
            role_templates={"user": "{user_message}", "assistant": "{assistant_message}"},
            seps=["\n"],
            tool_parser="qwen3_coder"
        )
        json_str = conv.model_dump_json()
        assert 'tool_parser_instance' not in json_str
        assert 'tool_parser' in json_str

    def test_deserialization_hydrates_tool_parser(self):
        """Deserialization should automatically hydrate the tool_parser_instance."""
        conv = Conversation(
            roles={"user": "{user_message}", "assistant": "{assistant_message}"},
            role_templates={"user": "{user_message}", "assistant": "{assistant_message}"},
            seps=["\n"],
            tool_parser="qwen3_coder"
        )
        
        # Serialize to JSON
        json_str = conv.model_dump_json()
        
        # Deserialize back to Conversation object
        import json
        json_dict = json.loads(json_str)
        new_conv = Conversation.model_validate(json_dict)
        
        # The tool_parser_instance should be hydrated again
        assert new_conv.tool_parser == "qwen3_coder"
        assert new_conv.tool_parser_instance is not None
        assert hasattr(new_conv.tool_parser_instance, 'parse')

    def test_serialization_with_no_tool_parser(self):
        """Serialization should work correctly when no tool_parser is specified."""
        conv = Conversation(
            roles={"user": "{user_message}", "assistant": "{assistant_message}"},
            role_templates={"user": "{user_message}", "assistant": "{assistant_message}"},
            seps=["\n"]
        )
        json_str = conv.model_dump_json()
        assert 'tool_parser' not in json_str or json_str.find('"tool_parser":null') != -1
        assert 'tool_parser_instance' not in json_str

    def test_from_json_dict_method(self):
        """from_json_dict method should create Conversation with hydrated tool_parser."""
        json_dict = {
            "roles": {"user": "{user_message}", "assistant": "{assistant_message}"},
            "role_templates": {"user": "{user_message}", "assistant": "{assistant_message}"},
            "seps": ["\n"],
            "tool_parser": "qwen3_coder"
        }
        conv = Conversation.from_json_dict(json_dict)
        assert conv.tool_parser == "qwen3_coder"
        assert conv.tool_parser_instance is not None
        assert hasattr(conv.tool_parser_instance, 'parse')

    def test_to_json_dict_method(self):
        """to_json_dict method should exclude tool_parser_instance."""
        conv = Conversation(
            roles={"user": "{user_message}", "assistant": "{assistant_message}"},
            role_templates={"user": "{user_message}", "assistant": "{assistant_message}"},
            seps=["\n"],
            tool_parser="qwen3_coder"
        )
        json_dict = conv.to_json_dict()
        assert 'tool_parser_instance' not in json_dict
        assert json_dict['tool_parser'] == "qwen3_coder"


class TestParserFunctionality:
    """Test basic functionality of the hydrated parser instance."""

    def test_hydrated_parser_has_parse_method(self):
        """Hydrated tool_parser_instance should have a parse method."""
        conv = Conversation(
            roles={"user": "{user_message}", "assistant": "{assistant_message}"},
            role_templates={"user": "{user_message}", "assistant": "{assistant_message}"},
            seps=["\n"],
            tool_parser="qwen3_coder"
        )
        assert callable(getattr(conv.tool_parser_instance, 'parse', None))

    def test_hydrated_parser_has_parse_streaming_method(self):
        """Hydrated tool_parser_instance should have a parse_streaming method."""
        conv = Conversation(
            roles={"user": "{user_message}", "assistant": "{assistant_message}"},
            role_templates={"user": "{user_message}", "assistant": "{assistant_message}"},
            seps=["\n"],
            tool_parser="qwen3_coder"
        )
        assert callable(getattr(conv.tool_parser_instance, 'parse_streaming', None))

    def test_hydrated_parser_parse_method_works(self):
        """Hydrated parser's parse method should work with XML tool call format."""
        conv = Conversation(
            roles={"user": "{user_message}", "assistant": "{assistant_message}"},
            role_templates={"user": "{user_message}", "assistant": "{assistant_message}"},
            seps=["\n"],
            tool_parser="qwen3_coder"
        )
        
        # Test with XML tool call format
        xml_text = '<tool_call><function=get_weather><parameter=location>Boston</parameter></function></tool_call>'
        content, tool_calls = conv.tool_parser_instance.parse(xml_text)
        
        assert isinstance(content, str)
        assert isinstance(tool_calls, list)

    def test_hydrated_parser_parse_streaming_method_works(self):
        """Hydrated parser's parse_streaming method should work with chunked input."""
        conv = Conversation(
            roles={"user": "{user_message}", "assistant": "{assistant_message}"},
            role_templates={"user": "{user_message}", "assistant": "{assistant_message}"},
            seps=["\n"],
            tool_parser="qwen3_coder"
        )
        
        # Test with partial chunk
        result = conv.tool_parser_instance.parse_streaming('<tool_call>')
        assert result is not None  # Should return some result for partial input
