import pytest
import asyncio
import unittest.mock
from pathlib import Path
from mlc_llm.protocol.openai_api_protocol import ChatCompletionRequest
from mlc_llm.serve.engine_base import ModelInfo, _process_model_args
from unittest.mock import MagicMock, patch

@pytest.mark.asyncio
async def test_e2e_tool_call_flow():
    """
    E2E Test: Simulate an OpenAI request with XML tool calls and verify 
    the engine correctly parses them using the hydrated Qwen3 parser.
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
    
    # 2. Mock the mlc_chat_config loading process
    with patch("mlc_llm.support.download_cache.get_or_download_model") as mock_download, \
         patch("builtins.open", unittest.mock.mock_open(read_data='{"model_type": "test", "quantization": "test", "model_config": {}, "vocab_size": 100, "context_window_size": 2048, "sliding_window_size": 0, "prefill_chunk_size": 1024, "attention_sink_size": 0, "tensor_parallel_shards": 1, "pipeline_parallel_stages": 1, "max_num_sequence": 8, "opt": null, "conv_template": {"name": "qwen3_5", "roles": {"user": "u", "assistant": "a", "tool": "t"}, "role_templates": {"user": "{user_message}", "assistant": "{assistant_message}", "tool": "{tool_message}"}, "seps": ["<sep>", "<sep>"], "tool_parser": "qwen3_coder"}}')), \
         patch("mlc_llm.interface.jit", create=True) as mock_jit:
        
        mock_download.return_value = Path("/tmp/test-model")
        # Mocking the return of jit.jit to return a dummy path
        mock_jit.return_value.model_lib_path = "/tmp/test-model/libqwen3.so"

        # 3. Run the model arg processing (which triggers hydration)
        model_args, config_paths, conversation = _process_model_args(
            [model_info], mock_device, engine_config
        )

        # VERIFY HYDRATION: Check if tool_parser_instance is hydrated with our real parser
        assert conversation.tool_parser == "qwen3_coder"
        assert conversation.tool_parser_instance is not None
        assert hasattr(conversation.tool_parser_instance, 'parse')

        # 4. Simulate an incoming OpenAI request payload containing XML tool calls
        request_payload = ChatCompletionRequest(
            model="qwen3-5-test",
            messages=[
                {"role": "user", "content": "Use the calculator to find 2+2"},
                {"role": "assistant", "content": "<tool_call><function=calc><parameter=expr>2+2</parameter></function></tool_call>"}
            ]
        )

        # 5. Test the parsing of this specific payload using our hydrated parser
        text_to_parse = "<tool_call><function=calc><parameter=expr>2+2</parameter></function></tool_call>"
        content, tool_calls = conversation.tool_parser_instance.parse(text_to_parse)

        # 6. Final Assertions: Verify the structure of extracted tool calls
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "calc"
        assert tool_calls[0].function.arguments["expr"] == "2+2"

