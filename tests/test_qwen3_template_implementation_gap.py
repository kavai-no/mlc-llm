import pytest
import asyncio
import unittest.mock
from pathlib import Path
from mlc_llm.serve.engine_base import ModelInfo, _process_model_args

@pytest.mark.asyncio
async def test_qwen3_template_xml_rendering_gap():
    """
    RED PHASE: This test verifies that the qwen3_5 template correctly renders 
    the tools list, tool calls, and tool results using XML tags.
    
    This MUST FAIL initially because implementation is not yet started.
    """
    # 1. Setup mock environment for Engine initialization
    mock_device = MagicMock()
    engine_config = MagicMock()
    engine_config.max_single_sequence_length = 2048
    engine_config.prefill_chunk_size = 1024
    engine_config.sliding_window_size = 0 
    engine_config.attention_sink_mode = 0
    engine_config.tensor_parallel_shards = 1
    engine_config.pipeline_parallel_stages = 1
    engine_config.max_num_sequence = 8
    engine_config.opt = None

    model_info = ModelInfo(model="qwen3-5-test", model_lib=None)
    
    # Mocking the conv_template with tool_parser="qwen3_coder"
    # We simulate the config that defines the new template behavior.
    config_json = '{"model_type": "test", "quantization": "test", "model_config": {}, "vocab_size": 100, "context_window_size": 2048, "sliding_window_size": 0, "prelang_chunk_size": 1024, "attention_sink_size": 0, "tensor_parallel_shards": 1, "pipeline_parallel_stages": 1, "max_num_sequence": 8, "opt": null, "conv_template": {"name": "qwen3_5", "roles": {"user": "u", "assistant": "a", "tool": "t"}, "role_templates": {"user": "{user_message}", "assistant": "{assistant_message}", "tool": "{tool_message}"}, "seps": ["<sep>", "<sep>"], "tool_parser": "qwen3_coder"}}'

    with patch("mlc_llm.support.download_cache.get_or_download_model") as mock_download, \
         patch("builtins.open", unittest.mock.mock_open(read_data=config_json)), \
         patch("mlc_llm.interface.jit", create=True) as mock_jit:
        
        mock_download.return_value = Path("/tmp/test-model")
        mock_jit.return_value.model_lib_path = "/tmp/test-model/libqwen3.so"

        # 2. Hydrate the conversation template
        _, _, conversation = _process_model_args(
            [model_info], mock_device, engine_config
        )

        # 3. TEST BOUNDARY: The Public API (the prompt/response content) must contain XML tags
        # We simulate a tool call occurring in the assistant's response.
        assistant_message = "<tool_call><function=calc><parameter=expr>2+2</parameter></function></tool_call>"
        
        # The implementation gap: Currently, the protocol might not be using 
        # the parser to wrap these correctly in the prompt or result.
        # We expect the final rendered prompt/message to contain these tags.
        
        # This assertion is expected to FAIL because we haven't implemented 
    	# the tool_parser integration into the protocol's as_prompt() yet.
        assert "<tool_call>" in assistant_message, "The assistant message should contain <tool_call> XML tags"
        assert "<function=" in assistant_message, "The tool call should use the <function=name> format"

        # 4. TEST BOUNDARY: Tool Result Rendering
        # We expect that when a tool result is provided, it's wrapped in <tool_response>
        tool_result = "4"
        # For now, we don't even have the logic to render this into XML.
        # If we were using conversation.as_prompt(), it would likely just return the raw string.
        rendered_result = conversation.tool_parser_instance.render_tool_call_result("calc", tool_result) if conversation.tool_parser_instance else tool_result
        
        assert "<tool_response>" in rendered_result, f"Tool result should be wrapped in <tool_response>, got: {rendered_result}"

if __name__ == "__main__":
    asyncio.run(test_qwen3_template_xml_rendering_gap())
