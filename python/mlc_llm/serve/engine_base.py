"""The MLC LLM Serving engine base class."""

# pylint: disable=too-many-lines

import ast
import asyncio
import json
import numbers
import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import tvm
from tvm.runtime import Device

from mlc_llm.protocol import openai_api_protocol
from mlc_llm.protocol.conversation_protocol import Conversation
from mlc_llm.protocol.generation_config import GenerationConfig
from mlc_llm.protocol.mlc_chat_config import MLCChatConfig
from mlc_llm.serve import data, engine_utils
from mlc_llm.serve.config import EngineConfig
from mlc_llm.serve.event_trace_recorder import EventTraceRecorder
from mlc_llm.serve.tool_parsers import ToolParserManager
from mlc_llm.support import download_cache, logging
from mlc_llm.support.auto_device import detect_device
from mlc_llm.support.style import green
from mlc_llm.tokenizers import TextStreamer, Tokenizer

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """The model info dataclass.

    Parameters
    ----------
    model : str
        The identifier of the input model.
        It may be a compiled model's id (e.g., "Llama-2-7b-chat-hf-q4f16_1"),
        or a full path to a model directory
        (e.g., "dist/prebuilt/mlc-chat-Llama-2-7b-chat-hf-q4f16_1")

    model_lib : Optional[str]
        The path to the compiled library of the model.
        E.g., "dist/prebuilt/lib/Llama-2-7b-chat-hf-q4f16_1-cuda.so"
    """

    model: str
    model_lib: Optional[str] = None


def _check_engine_config(
    model: str,
    model_lib: Optional[str],
    mode: Literal["local", "interactive", "server"],
    engine_config: EngineConfig,
) -> None:
    """Check if the given engine config is valid."""
    if engine_config.model is not None and engine_config.model != model:
        raise ValueError(
            f'The argument "model" of engine constructor is "{model}", while the "model" '
            f'field in argument "engine_config" is "{engine_config.model}". '
            'Please set the "engine_config.model" to None or set it to the same as the '
            'argument "model".'
        )
    if (
        engine_config.model_lib is not None
        and model_lib is not None
        and engine_config.model_lib != model_lib
    ):
        raise ValueError(
            f'The argument "model_lib" of engine constructor is "{model_lib}", while the '
            f'"model_lib" field in argument "engine_config" is "{engine_config.model_lib}". '
            'Please set the "engine_config.model_lib" to None or set it to the same as the '
            'argument "model_lib".'
        )
    if engine_config.mode is not None and engine_config.mode != mode:
        raise ValueError(
            f'The argument "mode" of engine constructor is "{mode}", while the '
            f'"mode" field in argument "engine_config" is "{engine_config.mode}". '
            'Please set the "engine_config.mode" to None or set it to the same as the '
            'argument "mode".'
        )
    if engine_config.kv_cache_page_size != 16:
        raise ValueError(
            'KV cache only supports page size 16, while the "kv_cache_page_size" field in '
            f'argument "engine_config" is "{engine_config.kv_cache_page_size}". '
            'Please set "engine_config.kv_cache_page_size" to 16.'
        )


def _parse_models(
    model: str,
    model_lib: Optional[str],
    additional_models: List[Union[str, Tuple[str, str]]],
) -> List[ModelInfo]:
    """Parse the specified model paths and model libs.
    Return a list of ModelInfo, which is a wrapper class of the model path + lib path.
    """
    models = [ModelInfo(model, model_lib)]
    for additional_model in additional_models:
        if isinstance(additional_model, str):
            models.append(ModelInfo(additional_model))
        else:
            models.append(ModelInfo(additional_model[0], additional_model[1]))
    return models


def _process_model_args(
    models: List[ModelInfo],
    device: tvm.runtime.Device,
    engine_config: EngineConfig,
) -> Tuple[List[Tuple[str, str]], List[str], Conversation]:
    """Process the input ModelInfo to get the engine initialization arguments."""
    conversation: Optional[Conversation] = None
    config_file_paths: List[str] = []

    def _convert_model_info(model: ModelInfo) -> Tuple[str, str]:
        nonlocal conversation

        model_path = download_cache.get_or_download_model(model.model)
        mlc_config_path = model_path / "mlc-chat-config.json"
        config_file_paths.append(str(mlc_config_path))

        with open(mlc_config_path, mode="rt", encoding="utf-8") as file:
            mlc_chat_config = MLCChatConfig.model_validate_json(file.read())

        if conversation is None:
            conversation = mlc_chat_config.conv_template

        if model.model_lib is not None:
            # do model lib search if the model lib is provided
            # error out if file not found
            if model.model_lib.startswith("mock://"):
                model_lib = model.model_lib
                logger.info("[DEBUG] mock test: %s", model_lib)
            elif Path(model.model_lib).is_file():
                model_lib = model.model_lib
                logger.info("Using library model: %s", model_lib)
            else:
                raise FileNotFoundError(
                    f"The `model_lib` you passed in is not a file: {model.model_lib}.\n"
                )
        else:
            # Run jit if model_lib is not provided
            # NOTE: we only import jit when necessary
            # so the engine do not have to depend on compilation
            from mlc_llm.interface import jit  # pylint: disable=import-outside-toplevel

            model_compile_overrides = {
                "context_window_size": engine_config.max_single_sequence_length,
                "prefill_chunk_size": engine_config.prefill_chunk_size,
                "sliding_window_size": engine_config.sliding_window_size,
                "attention_sink_size": engine_config.attention_sink_size,
                "tensor_parallel_shards": engine_config.tensor_parallel_shards,
                "pipeline_parallel_stages": engine_config.pipeline_parallel_stages,
                "max_batch_size": engine_config.max_num_sequence,
                "opt": engine_config.opt,
            }

            model_lib = jit.jit(
                model_path=model_path,
                overrides=model_compile_overrides,
                device=device,
            ).model_lib_path
        return str(model_path), model_lib

    model_args: List[Tuple[str, str]] = [_convert_model_info(model) for model in models]

    assert conversation is not None
    return model_args, config_file_paths, conversation


def _print_engine_mode_logging_msg(
    mode: Literal["local", "interactive", "server"],
) -> None:
    """Print the logging info for engine mode selection."""
    if mode == "local":
        logger.info(
            "The selected engine mode is %s. "
            "We choose small max batch size and KV cache capacity to use less GPU memory.",
            green(mode),
        )
    elif mode == "interactive":
        logger.info(
            "The selected engine mode is %s. "
            "We fix max batch size to 1 for interactive single sequence use.",
            green(mode),
        )
    else:
        logger.info(
            "The selected engine mode is %s. "
            "We use as much GPU memory as possible (within the limit "
            "of gpu_memory_utilization).",
            green(mode),
        )

    if mode != "local":
        logger.info(
            "If you have low concurrent requests and want to use less GPU memory, "
            'please select mode "local".'
        )
    if mode != "interactive":
        logger.info(
            "If you don't have concurrent requests and only use the engine interactively, "
            'please select mode "interactive".'
        )
    if mode != "server":
        logger.info(
            "If you have high concurrent requests and want to maximize the GPU memory utilization, "
            'please select mode "server".'
        )


class EngineMetrics:
    """Class to store the result returned by engine metrics"""

    metrics: dict

    def __init__(self, metrics):
        self.metrics = metrics

    def __str__(self):
        return self.metrics.__str__()

    def __repr__(self):
        return self.metrics.__repr__()

    def __getitem__(self, key):
        return self.metrics[key]

    def prometheus_text(self) -> str:
        """Convert engine metrics into prometheus text format

        Returns
        -------
        text: str
            The metrics in prometheus text format
        """
        output_lines = [
            "# NOTE: these metrics count token in the unit of serving model's tokenization",
            "# be careful when comparing them to client-side metrics that may use",
            "# different tokenization to standardize across models.\n",
        ]

        def traverse(comment_scope, key_prefix, curr_value):
            if isinstance(curr_value, dict):
                if comment_scope:
                    output_lines.append(f"\n# {comment_scope}")
                # first prioritize metrics in current scope
                for key, value in curr_value.items():
                    if isinstance(value, numbers.Number):
                        output_lines.append(f"{key_prefix}{key}\t{value}")
                # then look into nested scopes if any
                for key, value in curr_value.items():
                    if isinstance(value, dict) and len(value) != 0:
                        traverse(f"{comment_scope}/{key}", f"{key_prefix}{key}_", value)

        traverse("", "", self.metrics)
        return "\n".join(output_lines)


def _query_engine_metrics(engine):
    """Query engine metrics via debug options"""
    dummy_message = {"role": "user", "context": ""}
    for response in engine.chat.completions.create(
        messages=[dummy_message],
        model="model",
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"debug_config": {"special_request": "query_engine_metrics"}},
    ):
        if response.usage is not None:
            return EngineMetrics(response.usage.extra)
    raise RuntimeError("query_engine metrics did not get metrics back")


async def _async_query_engine_metrics(engine):
    """Query engine metrics via debug options"""
    dummy_message = {"role": "user", "context": ""}
    result = None
    async for response in await engine.chat.completions.create(
        messages=[dummy_message],
        model="model",
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"debug_config": {"special_request": "query_engine_metrics"}},
    ):
        if response.usage is not None:
            assert result is None
            result = EngineMetrics(response.usage.extra)

    if result is not None:
        return result
    raise RuntimeError("query_engine metrics did not get metrics back")


@dataclass
class CallbackStreamOutput:
    """The output of MLCEngine._generate and AsyncMLCEngine._generate

    Attributes
    ----------
    delta_text : str
        The delta text generated since the last output.

    delta_logprob_json_strs : Optional[List[str]]
        The list of logprob JSON strings since the last output,
        or None if the request does not require logprobs.

    finish_reason : Optional[str]
        The finish reason of the request, or None if unfinished.

    request_final_usage_json_str: Optional[str]
        The usage json which appears in last chunk,
        when it appears all other fields will be empty
    """

    delta_text: str
    delta_logprob_json_strs: Optional[List[str]]
    finish_reason: Optional[str]
    request_final_usage_json_str: Optional[str]


class AsyncRequestStream:
    """The asynchronous stream for requests in AsyncMLCEngine.

    Each request has its own unique stream.
    The stream exposes the method `push` for engine to push new generated
    delta text to the stream, and the method `finish` for engine to mark
    the finish of generation.

    The stream implements `__aiter__` and `__anext__`, which the engine
    can use to iterates all the generated tokens in order asynchronously.
    """

    # The asynchronous queue to hold elements of either a list of
    # CallbackStreamOutput or an exception.
    if sys.version_info >= (3, 9):
        _queue: asyncio.Queue[  # pylint: disable=unsubscriptable-object
            Union[List[CallbackStreamOutput], Exception]
        ]
    else:
        _queue: asyncio.Queue
    # The finish flag.
    _finished: bool

    def __init__(self) -> None:
        self._queue = asyncio.Queue()
        self._finished = False

    def push(self, item_or_exception: Union[List[CallbackStreamOutput], Exception]) -> None:
        """Push a new token to the stream."""
        if self._finished:
            # No new item is expected after finish.
            self._queue.put_nowait(
                RuntimeError(
                    "The request has already finished. "
                    "The stream is not supposed to accept new items."
                )
            )
            return
        self._queue.put_nowait(item_or_exception)

    def finish(self) -> None:
        """Mark the finish of the generation in the stream."""
        self._queue.put_nowait(StopIteration())
        self._finished = True

    def __aiter__(self):
        return self

    async def __anext__(self) -> List[CallbackStreamOutput]:
        result = await self._queue.get()
        if isinstance(result, StopIteration):
            raise StopAsyncIteration
        if isinstance(result, Exception):
            raise result
        return result


class EngineState:
    """The engine states that the request stream callback function may use.

    This class is used for both AsyncMLCEngine and MLCEngine.
    AsyncMLCEngine uses the fields and methods starting with "async",
    and MLCEngine uses the ones starting with "sync".

    - For AsyncMLCEngine, the state contains an asynchronous event loop,
    the streamers and the number of unfinished generations for each request
    being processed.
    - For MLCEngine, the state contains a callback output blocking queue,
    the text streamers and the number of unfinished requests.

    We use this state class to avoid the callback function from capturing
    the AsyncMLCEngine.

    The state also optionally maintains an event trace recorder, which can
    provide Chrome tracing when enabled.
    """

    trace_recorder = None
    # States used for AsyncMLCEngine
    async_event_loop: Optional[asyncio.AbstractEventLoop] = None
    async_streamers: Dict[str, Tuple[AsyncRequestStream, List[TextStreamer]]] = {}
    # States used for MLCEngine
    sync_output_queue: queue.Queue = queue.Queue()
    sync_text_streamers: List[TextStreamer] = []

    # Tool parser state tracking
    async_tool_parser_states: Dict[str, Any] = {}  # request_id -> tool_parser_state
    sync_tool_parser_states: Dict[str, Any] = {}  # request_id -> tool_parser_state

    def __init__(self, enable_tracing: bool) -> None:
        """Constructor."""
        if enable_tracing:
            self.trace_recorder = EventTraceRecorder()

    def record_event(self, request_id: str, event: str) -> None:
        """Record a event for the input request in the trace
        recorder when the recorder exists.

        Parameters
        ----------
        request_id : str
            The subject request of the event.

        event : str
            The event in a string name.
            It can have one of the following patterns:
            - "start xxx", which marks the start of event "xxx",
            - "finish xxx", which marks the finish of event "xxx",
            - "yyy", which marks the instant event "yyy".
            The "starts" and "finishes" will be automatically paired in the trace recorder.
        """
        if self.trace_recorder is None:
            return
        self.trace_recorder.add_event(request_id, event)

    def get_request_stream_callback(
        self, kind: Literal["async", "sync"]
    ) -> Callable[[List[data.RequestStreamOutput]], None]:
        """Construct a callback function and return.

        The callback function has signature
        "Callable[[List[data.RequestStreamOutput]], None]",
        whose input is a list of "data.RequestStreamOutput".
        Each "data.RequestStreamOutput" is the delta output of a request,
        generated from the engine.
        """

        f_callback = (
            self._async_request_stream_callback
            if kind == "async"
            else self._sync_request_stream_callback
        )

        def _callback(delta_outputs: List[data.RequestStreamOutput]) -> None:
            f_callback(delta_outputs)

        return _callback

    def async_lazy_init_event_loop(self) -> None:
        """Lazily set the asyncio event loop so that the event
        loop is the main driving event loop of the process.
        """
        if self.async_event_loop is None:
            self.async_event_loop = asyncio.get_event_loop()

    def _async_request_stream_callback(self, delta_outputs: List[data.RequestStreamOutput]) -> None:
        """The request stream callback function for AsyncMLCEngine to stream back
        the request generation results.

        Note
        ----
        This callback function uses `call_soon_threadsafe` in asyncio to
        schedule the invocation in the event loop, so that the underlying
        callback logic will be executed asynchronously in the future rather
        than right now.
        """

        # Schedule a callback run in the event loop without executing right now.
        # NOTE: This function causes GIL during execution.
        self.async_event_loop.call_soon_threadsafe(
            self._async_request_stream_callback_impl, delta_outputs
        )

    def _async_request_stream_callback_impl(
        self, delta_outputs: List[data.RequestStreamOutput]
    ) -> None:
        """The underlying implementation of request stream callback for AsyncMLCEngine."""
        for delta_output in delta_outputs:
            request_id, stream_outputs = delta_output.unpack()
            streamers = self.async_streamers.get(request_id, None)
            if streamers is None:
                continue

            self.record_event(request_id, event="start callback")
            stream, text_streamers = streamers

            # final chunk is now always indicated by a chunk
            # where usage json is present
            # the backend engine always streams back this chunk
            # regardless of include_usage option
            is_final_chunk = stream_outputs[0].request_final_usage_json_str is not None
            if is_final_chunk:
                # stream back this final usage chunk
                output = CallbackStreamOutput(
                    delta_text="",
                    delta_logprob_json_strs=None,
                    finish_reason=None,
                    request_final_usage_json_str=stream_outputs[0].request_final_usage_json_str,
                )
                stream.push([output])
                stream.finish()
                self.async_streamers.pop(request_id, None)
                continue

            outputs = []
            for stream_output, text_streamer in zip(stream_outputs, text_streamers):
                self.record_event(request_id, event="start detokenization")
                delta_text = stream_output.extra_prefix_string + (
                    text_streamer.put(stream_output.delta_token_ids)
                    if len(stream_output.delta_token_ids) > 0
                    else ""
                )
                if stream_output.finish_reason is not None:
                    delta_text += text_streamer.finish()
                self.record_event(request_id, event="finish detokenization")

                outputs.append(
                    CallbackStreamOutput(
                        delta_text=delta_text,
                        delta_logprob_json_strs=stream_output.delta_logprob_json_strs,
                        finish_reason=stream_output.finish_reason,
                        request_final_usage_json_str=None,
                    )
                )

            # Push new delta text to the stream.
            stream.push(outputs)
            self.record_event(request_id, event="finish callback")

    def _sync_request_stream_callback(self, delta_outputs: List[data.RequestStreamOutput]) -> None:
        """The request stream callback function for MLCEngine to stream back
        the request generation results.
        """
        # Put the delta outputs to the queue in the unblocking way.
        self.sync_output_queue.put_nowait(delta_outputs)


class MLCEngineBase:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """The base engine class, which implements common functions that
    are shared by MLCEngine and AsyncMLCEngine.

    This class wraps a threaded engine that runs on a standalone
    thread inside and streams back the delta generated results via
    callback functions. The internal threaded engine keeps running an
    loop that drives the engine.

    MLCEngine and AsyncMLCEngine inherits this MLCEngineBase class, and implements
    their own methods to process the delta generated results received
    from callback functions and yield the processed delta results in
    the forms of standard API protocols.

    Checkout subclasses AsyncMLCEngine/MLCEngine for the docstring of constructor parameters.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        kind: Literal["async", "sync"],
        model: str,
        device: Union[str, tvm.runtime.Device],
        model_lib: Optional[str],
        mode: Literal["local", "interactive", "server"],
        engine_config: Optional[EngineConfig],
        enable_tracing: bool,
    ) -> None:
        # - Check the fields fields of `engine_config`.
        if engine_config is None:
            engine_config = EngineConfig()
        _check_engine_config(model, model_lib, mode, engine_config)

        # - Initialize model loading info.
        models = _parse_models(model, model_lib, engine_config.additional_models)
        if isinstance(device, str):
            device = detect_device(device)
        assert isinstance(device, Device)
        (
            model_args,
            model_config_paths,
            self.conv_template,
        ) = _process_model_args(models, device, engine_config)

        # - Load the raw model config into dict
        self.model_config_dicts = []
        for i, model_info in enumerate(models):
            model_info.model_lib = model_args[i][1]
            with open(model_config_paths[i], "r", encoding="utf-8") as file:
                self.model_config_dicts.append(json.load(file))

        # - Print logging info for regarding the mode selection.
        if engine_config.verbose:
            _print_engine_mode_logging_msg(mode)

        # - Initialize engine state and engine.
        self.state = EngineState(enable_tracing)
        module = tvm.get_global_func("mlc.serve.create_threaded_engine", allow_missing=False)()
        self._ffi = {
            key: module[key]
            for key in [
                "add_request",
                "abort_request",
                "run_background_loop",
                "run_background_stream_back_loop",
                "reload",
                "init_threaded_engine",
                "exit_background_loop",
                "create_request",
                "get_complete_engine_config",
                "reset",
                "debug_call_func_on_all_worker",
            ]
        }
        self.tokenizer = Tokenizer(model_args[0][0])
        
        # Initialize tool parser if available
        self.tool_parser = None
        # Check for tool parser from conversation template first, then engine_config
        tool_parser_name = getattr(self.conv_template, 'tool_parser', None) or getattr(engine_config, 'tool_parser', None)
        if tool_parser_name and tool_parser_name != "default":
            logger.info(f"Creating tool parser from template/config: {tool_parser_name}")
            tool_parser_class = ToolParserManager.get_parser(tool_parser_name)
            if tool_parser_class:
                self.tool_parser = tool_parser_class(self.tokenizer)
                logger.info(f"Successfully initialized tool parser: {self.tool_parser}")
            else:
                logger.warning(f"Could not find tool parser class: {tool_parser_name}")
        else:
            logger.debug(f"No tool_parser configured (template: {getattr(self.conv_template, 'tool_parser', None)}, config: {getattr(engine_config, 'tool_parser', None)})")
        
        self._ffi["init_threaded_engine"](
            device,
            self.state.get_request_stream_callback(kind),
            self.state.trace_recorder,
        )

        background_loop = self._ffi["run_background_loop"]
        background_stream_back_loop = self._ffi["run_background_stream_back_loop"]

        # - Create the background engine-driving thread and start the loop.
        self._background_loop_thread: threading.Thread = threading.Thread(target=background_loop)
        self._background_stream_back_loop_thread: threading.Thread = threading.Thread(
            target=background_stream_back_loop
        )
        self._background_loop_thread.start()
        self._background_stream_back_loop_thread.start()
        self._terminated = False
        
        engine_config.model = model_args[0][0]
        engine_config.model_lib = model_args[0][1]
        engine_config.additional_models = model_args[1:]  # type: ignore
        engine_config.mode = mode
        self._ffi["reload"](engine_config.asjson())
        self.engine_config = EngineConfig.from_json(self._ffi["get_complete_engine_config"]())
        self.max_input_sequence_length = min(
            self.engine_config.max_single_sequence_length,
            self.engine_config.max_total_sequence_length,
        )

    def __del__(self):
        """deleter, auto terminate"""
        self.terminate()

    def terminate(self):
        """Terminate the engine."""
        if hasattr(self, "_terminated") and self._terminated:
            return
        self._terminated = True
        if not hasattr(self, "_ffi"):
            return
        self._ffi["exit_background_loop"]()
        if hasattr(self, "_background_loop_thread"):
            self._background_loop_thread.join()
        if hasattr(self, "_background_stream_back_loop_thread"):
            self._background_stream_back_loop_thread.join()

    def _debug_call_func_on_all_worker(
        self, func_name: str, func_args: Optional[str] = None
    ) -> None:
        """Call the given global function on all workers. Only for debug purpose."""
        self._ffi["debug_call_func_on_all_worker"](func_name, func_args)

    def reset(self):
        """Reset the engine, clear the running data and metrics."""
        return self._ffi["reset"]()


def process_chat_completion_request(  # pylint: disable=too-many-arguments
    request: openai_api_protocol.ChatCompletionRequest,
    request_id: str,
    engine_state: EngineState,
    model_config: Dict[str, Any],
    f_tokenize: Callable[[str], List[int]],
    max_input_sequence_length: int,
    conv_template: Conversation,
    tool_parser_instance: Optional[object] = None,
) -> Tuple[List[Union[List[int], data.Data]], GenerationConfig, bool, int]:
    """Process the given ChatCompletionRequest, apply request validity
    checks, and return the processed prompts, and other info.

    Parameters
    ----------
    request : openai_api_protocol.ChatCompletionRequest
        The request to be processed and checked.

    request_id : str
        The id of the request.

    engine_state : EngineState
        The state of the engine.

    model_config : Dict[str, Any]
        The model configuration dictionary.

    f_tokenize : Callable[[str], List[int]]
        The tokenizer encode function.

    max_input_sequence_length : int
        The maximum allowed total prompt length.

    conv_template : Conversation
        The conversation template of the model.

    Returns
    -------
    prompts : List[Union[List[int], data.Data]]
        The prompts, in a list.
        Each element is a list of token ids or a "data.Data" instance.

    generation_cfg : GenerationConfig
        The generation config of the request got from the input request.

    use_function_calling : bool
        A boolean flag indicating if the request uses function call.

    prompt_length : int
        The total prompt length.
    """
    engine_state.record_event(request_id, event="receive request")
    # - Check if unsupported arguments are specified.
    engine_utils.check_unsupported_fields(request)

    # - Process messages and update the conversation template in three steps:
    #   i. Check the message validity.
    #  ii. Add the input messages to the conversation template.
    # iii. Add the additional message for the assistant.
    request.check_message_validity()
    # - Check for function calling usage and update the conversation template
    request.check_function_call_usage(conv_template, tool_parser_instance=tool_parser_instance)

    for message in request.messages:
        role = message.role
        content = message.content
        if role == "system":
            assert isinstance(content, str)
            conv_template.system_message = content if content is not None else ""
            continue
        conv_template.messages.append((role, content))
    conv_template.messages.append(("assistant", None))

    # - Get the prompt from template, and encode to token ids.
    # - Check prompt length
    engine_state.record_event(request_id, event="start tokenization")
    prompts = engine_utils.process_prompts(  # type: ignore
        conv_template.as_prompt(model_config), f_tokenize
    )
    engine_state.record_event(request_id, event="finish tokenization")

    if conv_template.system_prefix_token_ids is not None:
        if isinstance(prompts[0], list):
            prompts[0] = conv_template.system_prefix_token_ids + prompts[0]
        else:
            prompts.insert(0, conv_template.system_prefix_token_ids)
    prompt_length = engine_utils.check_and_get_prompts_length(prompts, max_input_sequence_length)

    # Process generation config. Create request id.
    generation_cfg = engine_utils.get_generation_config(
        request,
        extra_stop_token_ids=conv_template.stop_token_ids,
        extra_stop_str=conv_template.stop_str,
    )
    return prompts, generation_cfg, conv_template.use_function_calling, prompt_length


def process_chat_completion_stream_output(  # pylint: disable=too-many-arguments
    delta_outputs: List[CallbackStreamOutput],
    request: openai_api_protocol.ChatCompletionRequest,
    request_id: str,
    engine_state: EngineState,
    use_function_calling: bool,
    finish_reasons: List[Optional[str]],
    tool_parser: Optional[Any] = None,
) -> Optional[openai_api_protocol.ChatCompletionStreamResponse]:
    """Process the delta outputs of a single request of ChatCompletion,
    convert the delta output to ChatCompletionStreamResponse and return.

    Parameters
    ----------
    delta_outputs : List[CallbackStreamOutput]
        The delta outputs of a request.
        The list length is the number of parallel generation specified by "n".
        Each element corresponds to a generation.

    request_id : str
        The id of the request.

    engine_state : EngineState
        The state of the engine.

    use_function_calling : bool
        A boolean flag indicating if the request uses function call.

    finish_reasons : List[Optional[str]]
        The list of finish reasons of each generation.
        The list length is the number of parallel generation specified by "n".
        This list is updated in place.

    Returns
    -------
    response : Optional[openai_api_protocol.ChatCompletionStreamResponse]
        The converted OpenAI API ChatCompletionStreamResponse instance.
        It can be none when there is no content.
    """
    # we always stream back the final chunk with usage
    is_final_chunk = delta_outputs[0].request_final_usage_json_str is not None
    if is_final_chunk:
        assert len(delta_outputs) == 1
        engine_state.record_event(request_id, event="yield final usage")
        response = openai_api_protocol.ChatCompletionStreamResponse(
            id=request_id,
            choices=[],
            model=request.model,
            system_fingerprint="",
            usage=openai_api_protocol.CompletionUsage.model_validate_json(
                delta_outputs[0].request_final_usage_json_str
            ),
        )
        # non streaming mode always comes with usage
        if not request.stream:
            return response
        # skip usage if stream option does not indicate include usage
        if request.stream_options is None:
            return None
        if not request.stream_options.include_usage:
            return None
        return response

    # normal chunk
    assert len(delta_outputs) == request.n
    choices = []
    
    # Track accumulated text for tool parsing - keyed by request_id + response_index
    if not hasattr(engine_state, '_tool_call_text_buffer'):
        engine_state._tool_call_text_buffer = {}
    
    for i, delta_output in enumerate(delta_outputs):
        # Initialize buffer for this specific response (request_id + index)
        buffer_key = f"{request_id}_{i}"
        if buffer_key not in engine_state._tool_call_text_buffer:
            engine_state._tool_call_text_buffer[buffer_key] = ""
        finish_reason_updated = False
        if delta_output.finish_reason is not None and finish_reasons[i] is None:
            # Only set finish_reason to tool_calls if we actually have function calling
            # The parser will determine this based on whether it finds tool calls in the XML
            finish_reasons[i] = delta_output.finish_reason
            
            # Check if tool parser has detected actual tool calls (for streaming mode)
            if tool_parser is not None and use_function_calling:
                # If TVM set finish_reason to "tool_calls" but we haven't found any in prev_tool_call_arr,
                # it means the XML was malformed or incomplete
                if finish_reasons[i] == "tool_calls" and (
                    not hasattr(tool_parser, 'prev_tool_call_arr') or 
                    len(tool_parser.prev_tool_call_arr) == 0
                ):
                    logger.info("Correcting streaming finish_reason from 'tool_calls' to 'stop'")
                    finish_reasons[i] = "stop"
            
            finish_reason_updated = True
        if not finish_reason_updated and delta_output.delta_text == "":
            # Ignore empty delta text when finish reason is not updated.
            engine_state.record_event(request_id, event="skip empty delta text")
            continue

        # Use tool parser if available for streaming
        if tool_parser is not None and use_function_calling:
            # Get accumulated text from our buffer
            current_text = engine_state._tool_call_text_buffer[buffer_key]
            delta_text = delta_output.delta_text
            
            # Update buffer with new delta text
            engine_state._tool_call_text_buffer[buffer_key] += delta_text
            
            # Calculate previous text
            if len(delta_text) > 0 and len(current_text) >= len(delta_text):
                previous_text = current_text[:-len(delta_text)]
            else:
                previous_text = ""
            
            # Call tool parser for streaming
            tool_parser_result = tool_parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=getattr(delta_output, "previous_token_ids", []) or [],
                current_token_ids=getattr(delta_output, "current_token_ids", []) or [],
                delta_token_ids=getattr(delta_output, "delta_token_ids", []) or [],
                request=request,
            )
            
            if tool_parser_result is not None:
                # In streaming mode, we need proper deltas for tool calls
                # The tool parser returns complete messages, so we construct a delta from changes
                if request.stream and hasattr(tool_parser_result, 'tool_calls') and tool_parser_result.tool_calls:
                    # Create proper streaming delta with incremental tool call info
                    first_tool_call = tool_parser_result.tool_calls[0]
                    
                    # Check what's changed from previous state
                    if hasattr(engine_state, '_prev_streaming_tool_state'):
                        prev_state = engine_state._prev_streaming_tool_state.get(request_id + f"_{i}", {})
                        
                        # Only include fields that have changed or are new
                        tool_delta = {}
                        if 'function' in first_tool_call and hasattr(first_tool_call.function, 'name'):
                            if prev_state.get('function', {}).get('name') != first_tool_call.function.name:
                                tool_delta['name'] = first_tool_call.function.name
                        
                        # Always include arguments as they change incrementally
                        if hasattr(first_tool_call.function, 'arguments'):
                            tool_delta['arguments'] = first_tool_call.function.arguments
                            
                        # Include id only once when starting
                        if not prev_state.get('sent_id'):
                            tool_delta['id'] = str(getattr(first_tool_call, 'id', ''))
                            engine_state._prev_streaming_tool_state.setdefault(request_id + f"_{i}", {})['sent_id'] = True
                            
                        delta_message = openai_api_protocol.ChatCompletionMessage(
                            content="",
                            role="assistant"
                        ) if (not tool_delta and 'arguments' not in tool_delta) else openai_api_protocol.ChatCompletionMessage(
                            content="",
                            role="assistant",
                            tool_calls=[openai_api_protocol.ChatToolCall(
                                type="function",
                                index=0,
                                function=openai_api_protocol.ChatFunctionCall(name=first_tool_call.function.name, **tool_delta) if tool_delta else None
                            )]
                        )
                    else:
                        # First chunk - include initial structure
                        delta_message = openai_api_protocol.ChatCompletionMessage(
                            content="",
                            role="assistant",
                            tool_calls=[openai_api_protocol.ChatToolCall(
                                type="function",
                                index=0,
                                function=openai_api_protocol.ChatFunctionCall(name=first_tool_call.function.name, arguments="{}")
                            )]
                        )
                    
                    # Track current state for next comparison
                    if not hasattr(engine_state, '_prev_streaming_tool_state'):
                        engine_state._prev_streaming_tool_state = {}
                    engine_state._prev_streaming_tool_state[request_id + f"_{i}"] = {
                        'function': {'name': first_tool_call.function.name} if hasattr(first_tool_call.function, 'name') else {},
                        'arguments': first_tool_call.function.arguments if hasattr(first_tool_call.function, 'arguments') else ""
                    }
                elif not hasattr(tool_parser_result, 'tool_calls') or len(tool_parser_result.tool_calls) == 0:
                    # Tool parser result with empty/no tool calls - remove tool_calls field
                    delta_message = openai_api_protocol.ChatCompletionMessage(
                        content=getattr(tool_parser_result, 'content', ''),
                        role="assistant",
                        name=getattr(tool_parser_result, 'name', None)
                    )
                else:
                    # Non-streaming or no streaming changes - use full message
                    delta_message = tool_parser_result
            elif tool_parser_result is not None and (not hasattr(tool_parser_result, 'tool_calls') or len(tool_parser_result.tool_calls) == 0):
                # Tool parser returned a result but with empty/no tool calls
                delta_message = openai_api_protocol.ChatCompletionMessage(
                    content=delta_output.delta_text, 
                    role="assistant"
                )
            else:
                # No tool parser or no function calling - default behavior
                delta_message = openai_api_protocol.ChatCompletionMessage(
                    content=delta_output.delta_text, 
                    role="assistant"
                )
        else:
            # Default behavior without tool parser or not using function calling
            delta_message = openai_api_protocol.ChatCompletionMessage(
                content=delta_output.delta_text, role="assistant"
            )
        
        choices.append(
            openai_api_protocol.ChatCompletionStreamResponseChoice(
                index=i,
                finish_reason=finish_reasons[i],
                delta=delta_message,
                logprobs=(
                    openai_api_protocol.LogProbs(
                        content=[
                            openai_api_protocol.LogProbsContent.model_validate_json(
                                logprob_json_str
                            )
                            for logprob_json_str in delta_output.delta_logprob_json_strs
                        ]
                    )
                    if delta_output.delta_logprob_json_strs is not None
                    else None
                ),
            )
        )

    if len(choices) == 0:
        # Skip return when there is no delta output and no number of completion tokens.
        return None
    response = openai_api_protocol.ChatCompletionStreamResponse(
        id=request_id, choices=choices, model=request.model, system_fingerprint=""
    )
    engine_state.record_event(request_id, event="yield delta output")
    return response


def process_completion_request(  # pylint: disable=too-many-arguments
    request: openai_api_protocol.CompletionRequest,
    request_id: str,
    engine_state: EngineState,
    tokenizer: Tokenizer,
    max_input_sequence_length: int,
    conv_template: Conversation,
) -> Tuple[List[int], GenerationConfig, int, Optional[openai_api_protocol.CompletionResponse]]:
    """Process the given CompletionRequest, apply request validity
    checks, and return the processed prompts, and other info.

    Parameters
    ----------
    request : openai_api_protocol.CompletionRequest
        The request to be processed and checked.

    request_id : str
        The id of the request.

    engine_state : EngineState
        The state of the engine.

    tokenizer : Tokenizer
        The tokenizer instance of the model.

    max_input_sequence_length : int
        The maximum allowed total prompt length.

    conv_template : Conversation
        The conversation template of the model.

    Returns
    -------
    prompt : List[int]
        The prompt in a list of token ids.

    generation_cfg : GenerationConfig
        The generation config of the request got from the input request.

    prompt_length : int
        The total prompt length.

    echo_response : Optional[openai_api_protocol.CompletionResponse]
        The CompletionResponse of the echoing part, when argument "echo"
        of the input request is specified.
    """
    engine_state.record_event(request_id, event="receive request")
    # - Check if unsupported arguments are specified.
    engine_utils.check_unsupported_fields(request)

    # - Process prompt and check validity.
    engine_state.record_event(request_id, event="start tokenization")
    prompts = engine_utils.process_prompts(request.prompt, tokenizer.encode)
    engine_state.record_event(request_id, event="finish tokenization")
    prompt_length = engine_utils.check_and_get_prompts_length(prompts, max_input_sequence_length)
    prompt = prompts[0]
    assert isinstance(prompt, list)

    # Process generation config. Create request id.
    generation_cfg = engine_utils.get_generation_config(
        request,
        extra_stop_token_ids=conv_template.stop_token_ids,
        extra_stop_str=conv_template.stop_str,
    )

    # - Echo back the prompt.
    echo_response = None
    if request.echo:
        text = tokenizer.decode(prompt)
        response = openai_api_protocol.CompletionResponse(
            id=request_id,
            choices=[
                openai_api_protocol.CompletionResponseChoice(index=i, text=text)
                for i in range(generation_cfg.n)
            ],
            model=request.model,
            usage=None,
        )
        echo_response = response
    return prompt, generation_cfg, prompt_length, echo_response


def get_logprobs_from_delta(
    delta_logprob_json_strs: List[str],
) -> openai_api_protocol.CompletionLogProbs:
    """Convert json strings containing logprobs information to
    completion response format (OpenAI API compatible)

    Parameters
    ----------
    delta_logprob_json_strs : List[str]
        Logprobs information packed in json strings and
        kept in the delta outputs of a request.

    Returns
    -------
    logprobs : openai_api_protocol.CompletionLogProbs
        Logprobs information extracted from json string and converted to completion response format
    """
    token_logprobs = []
    tokens = []
    top_logprobs = []
    for logprob_json_str in delta_logprob_json_strs:
        content = openai_api_protocol.LogProbsContent.model_validate_json(logprob_json_str)
        tokens.append(content.token)
        token_logprobs.append(content.logprob)
        top_logprob_dict = {}
        for top_logprob in content.top_logprobs:
            top_logprob_dict[top_logprob.token] = top_logprob.logprob
        top_logprobs.append(top_logprob_dict)
    return openai_api_protocol.CompletionLogProbs(
        # TODO(vvchernov): support text_offset
        text_offset=None,
        token_logprobs=token_logprobs,
        tokens=tokens,
        top_logprobs=top_logprobs,
    )


def process_completion_stream_output(  # pylint: disable=too-many-arguments
    delta_outputs: List[CallbackStreamOutput],
    request: openai_api_protocol.CompletionRequest,
    request_id: str,
    engine_state: EngineState,
    finish_reasons: List[Optional[str]],
) -> Optional[openai_api_protocol.ChatCompletionResponse]:
    """Process the delta outputs of a single request of Completion,
    convert the delta output to CompletionResponse and return.

    Parameters
    ----------
    delta_outputs : List[CallbackStreamOutput]
        The delta outputs of a request.
        The list length is the number of parallel generation specified by "n".
        Each element corresponds to a generation.

    request: openai_api_protocol.CompletionRequest
        Information about the request

    request_id : str
        The id of the request.

    engine_state : EngineState
        The state of the engine.

    finish_reasons : List[Optional[str]]
        The list of finish reasons of each generation.
        The list length is the number of parallel generation specified by "n".
        This list is updated in place.

    Returns
    -------
    response : Optional[openai_api_protocol.CompletionResponse]
        The converted OpenAI API CompletionResponse instance.
        It can be none when there is no content.
    """
    # we always stream back the final chunk with usage
    is_final_chunk = delta_outputs[0].request_final_usage_json_str is not None
    if is_final_chunk:
        assert len(delta_outputs) == 1
        engine_state.record_event(request_id, event="yield final usage")
        response = openai_api_protocol.ChatCompletionResponse(
            id=request_id,
            choices=[],
            model=request.model,
            system_fingerprint="",
            usage=openai_api_protocol.CompletionUsage.model_validate_json(
                delta_outputs[0].request_final_usage_json_str
            ),
        )
        # non streaming mode always comes with usage
        if not request.stream:
            return response
        if request.stream_options is None:
            return None
        if not request.stream_options.include_usage:
            return None
        return response

    # normal chunk
    assert len(delta_outputs) == request.n
    choices = []
    for i, delta_output in enumerate(delta_outputs):
        if delta_output.finish_reason is not None and finish_reasons[i] is None:
            finish_reasons[i] = delta_output.finish_reason
            # Ignore empty delta text when finish reason is not updated.
            continue

        if delta_output.delta_logprob_json_strs is not None:
            logprobs = get_logprobs_from_delta(delta_output.delta_logprob_json_strs)
        else:
            logprobs = None
        choices.append(
            openai_api_protocol.CompletionResponseChoice(
                index=i,
                finish_reason=finish_reasons[i],
                text=delta_output.delta_text,
                logprobs=logprobs,
            )
        )

    if len(choices) == 0:
        # Skip return when there is no delta output and no number of completion tokens.
        return None
    response = openai_api_protocol.CompletionResponse(
        id=request_id,
        choices=choices,
        model=request.model,
        usage=None,
    )
    engine_state.record_event(request_id, event="yield delta output")
    return response


def create_completion_suffix_response(
    request: openai_api_protocol.CompletionRequest,
    request_id: str,
    finish_reasons: List[Optional[str]],
) -> Optional[openai_api_protocol.CompletionResponse]:
    """Create the suffix response of Completion request
    when the request requires suffix.

    Parameters
    ----------
    request : openai_api_protocol.CompletionRequest
        The request whose suffix response if to be created.

    request_id : str
        The id of the request.

    finish_reasons : List[Optional[str]]
        The list of finish reasons of each generation.
        The list length is the number of parallel generation specified by "n".
        This list is updated in place.

    Returns
    -------
    suffix_response : Optional[openai_api_protocol.CompletionResponse]
        The created OpenAI API CompletionResponse instance for the suffix.
        Or None if the request does not require suffix.
    """
    # - Echo the suffix.
    if request.suffix is None:
        return None
    assert all(finish_reason is not None for finish_reason in finish_reasons)
    response = openai_api_protocol.CompletionResponse(
        id=request_id,
        choices=[
            openai_api_protocol.CompletionResponseChoice(
                index=i,
                finish_reason=finish_reason,
                text=request.suffix,
            )
            for i, finish_reason in enumerate(finish_reasons)
        ],
        model=request.model,
        usage=None,
    )
    return response


def convert_function_str_to_json(stringified_calls: str) -> List[Union[Dict, None]]:
    """Convert a (possibly list) of function call string to a list of json objects.
    Return None for invalid function call string."""

    # Handle empty input
    if not stringified_calls or not stringified_calls.strip():
        return []

    def parse_function_call(call_str: str):
        node = ast.parse(call_str, mode="eval")
        call_node = node.body
        if isinstance(call_node, ast.Call) and isinstance(call_node.func, ast.Name):
            name = call_node.func.id
            arguments = {}
            for keyword in call_node.keywords:
                arguments[keyword.arg] = ast.literal_eval(keyword.value)
            return {"name": name, "arguments": arguments}
        return None

    if (
        len(stringified_calls) > 0 and
        stringified_calls[0] == "[" and stringified_calls[-1] == "]"
    ):  # hacky way to check if string list
        calls = ast.literal_eval(stringified_calls)
    else:
        calls = [stringified_calls]
    function_calls_json = [parse_function_call(call_str) for call_str in calls]
    return function_calls_json


def process_function_call_output(
    output_texts: List[str],
    finish_reasons: List[str],
    tool_parser: Optional[Any] = None,
) -> Tuple[bool, List[List[openai_api_protocol.ChatToolCall]], List[str]]:
    """Process the potential function call results outputted by model,
    according to the finish reasons.
    Return whether the output has function call, and the list of tool calls.
    
    If a tool_parser is provided (e.g., Qwen3CoderToolParser), it will be used
    to extract tool calls from XML format output. Otherwise, falls back to
    the default AST-based parsing.
    """
    logger.info(f"process_function_call_output: tool_parser={tool_parser}, finish_reasons={finish_reasons}")
    logger.info(f"Output texts received: {[len(t) for t in output_texts]}")
    # Track if we've seen a valid tool call with arguments to prevent duplicate empty ones
    _seen_valid_tool_call = [False] * len(output_texts)
    # Log full text for debugging (may contain sensitive info)
    for i, text in enumerate(output_texts):
        logger.debug(f"Response {i} FULL raw output: {repr(text)}")
    n = len(output_texts)
    tool_calls_list: List[List[openai_api_protocol.ChatToolCall]] = [[] for _ in range(n)]
    content_list: List[str] = [""] * n  # Track content for each response
    
    # Log the raw model output (first few chars for privacy)
    for i, text in enumerate(output_texts):
        preview = text[:200] if len(text) > 200 else text
        logger.info(f"Response {i} raw output: {repr(preview)}... length={len(text)}")
    # Determine function calling per-response instead of globally
    # This fixes issues with multiple responses where only some have tool calls
    
    for i, output_text in enumerate(output_texts):
        try:
            # Use tool parser if available (for Qwen3Coder XML format)
            if tool_parser is not None and hasattr(tool_parser, 'extract_tool_calls'):
                logger.info(f"Using Qwen3CoderToolParser for output_text length {len(output_text)}")
                # Extract tool calls using the tool parser
                result = tool_parser.extract_tool_calls(output_text, request=None)
                logger.info(f"Parser result: tools_called={result.tools_called}, num_calls={len(result.tool_calls or [])}")
                if result.tools_called and result.tool_calls and len(result.tool_calls) > 0:
                    # Validate that required parameters are present before treating as valid tool call
                    has_valid_args = True
                    logger.info(f"Validating {len(result.tool_calls)} tool calls from parser - checking for empty arguments")
                    for idx, tool_call in enumerate(result.tool_calls):
                        if not hasattr(tool_call, 'function') or not tool_call.function:
                            logger.info(f"Tool call {idx} missing function attribute")
                            has_valid_args = False
                            break
                        args_str = getattr(tool_call.function, 'arguments', '{}')
                        logger.info(f"Tool call {idx}: name={getattr(tool_call.function, 'name', None)}, args={repr(args_str)}")
                        try:
                            args_dict = json.loads(args_str)
                            if not isinstance(args_dict, dict):
                                logger.info(f"Tool call {idx} arguments are not a valid dict: {type(args_dict)}")
                                has_valid_args = False
                                break
                            elif len(args_dict) == 0:
                                # Empty dict - treat as invalid for required parameters
                                logger.warning(f"Tool call {idx} has empty arguments dict - this causes streaming issues with multiple tool_calls!")
                                has_valid_args = False
                                break
                        except (json.JSONDecodeError, ValueError) as e:
                            # Invalid JSON or empty string
                            logger.info(f"Tool call {idx} has invalid JSON: {e}")
                            has_valid_args = False
                            break
                    
                    tool_calls_list[i] = result.tool_calls
                    if has_valid_args:
                        logger.info(f"Response {i}: Valid tool calls detected, setting finish_reason='tool_calls'")
                        finish_reasons[i] = "tool_calls"
                        _seen_valid_tool_call[i] = True
                    else:
                        # If we already saw a valid tool call, don't add empty ones during streaming
                        if _seen_valid_tool_call[i]:
                            logger.warning(f"Response {i}: Skipping duplicate empty tool call after valid one")
                            tool_calls_list[i] = []
                        logger.info(f"Response {i}: Invalid/empty arguments detected, setting finish_reason='stop'")
                        finish_reasons[i] = "stop"  # Treat as regular response
                    
                    # Log final state for this response
                    logger.info(f"Response {i} FINAL: tool_calls_count={len(tool_calls_list[i]) if i < len(tool_calls_list) else 0}, finish_reason={finish_reasons[i]}, content_length={len(content_list[i]) if i < len(content_list) else 0}")
                    # Clear tool calls when arguments are empty to prevent streaming incomplete tool calls
                    if not has_valid_args:
                        logger.info(f"Response {i}: Clearing tool_calls_list[{i}] due to invalid arguments - prevents empty {{}} tool calls from appearing in streaming")
                        tool_calls_list[i] = []
                    content_list[i] = ""
                    continue
                elif not result.tools_called or len(result.tool_calls or []) == 0:
                    # Qwen3Coder parser found NO valid tool calls, but TVM might have thought there were some.
                    # This happens when XML is malformed or doesn't match expected format.
                    if finish_reasons[i] == "tool_calls":
                        logger.info(f"Correcting finish_reason from 'tool_calls' to 'stop' for output_text length {len(output_text)}")
                        finish_reasons[i] = "stop"
                    continue
            
            # Fallback to default parsing for non-tool-parser cases or if tool parser fails
            fn_json_list = convert_function_str_to_json(output_text)
            
            # Convert parsed JSON to ChatToolCall objects
            tool_calls_list[i] = [
                openai_api_protocol.ChatToolCall(
                    type="function",
                    function=openai_api_protocol.ChatFunctionCall(
                        name=fn_json_obj["name"], arguments=fn_json_obj["arguments"]
                    ),
                )
                for fn_json_obj in fn_json_list
                if fn_json_obj is not None
            ]
            
            # Validate that we got valid tool calls
            if len(tool_calls_list[i]) == 0:
                output_texts[i] = "Got an invalid function call output from model"
                finish_reasons[i] = "error"
            else:
                finish_reasons[i] = "tool_calls"
        except (SyntaxError, ValueError, IndexError, KeyError) as e:
            output_texts[i] = f"Got an invalid function call output from model: {str(e)}"
            finish_reasons[i] = "error"
    
    # Determine if function calling was used for ANY response
    actual_use_function_calling = (tool_parser is not None) or any(len(calls) > 0 for calls in tool_calls_list)
    return actual_use_function_calling, tool_calls_list, content_list


def wrap_chat_completion_response(  # pylint: disable=too-many-arguments
    request_id: str,
    model: str,
    output_texts: List[str],
    finish_reasons: List[str],
    tool_calls_list: List[List[openai_api_protocol.ChatToolCall]],
    content_list: List[str],
    logprob_results: Optional[List[List[openai_api_protocol.LogProbsContent]]],
    use_function_calling: bool,
    usage: Optional[openai_api_protocol.CompletionUsage],
) -> openai_api_protocol.ChatCompletionResponse:
    """Wrap the non-streaming chat completion results to ChatCompletionResponse instance."""
    choices = []
    for i in range(len(output_texts)):
        output_text = output_texts[i]
        finish_reason = finish_reasons[i]
        tool_calls = tool_calls_list[i] if i < len(tool_calls_list) else None
        content_for_message = content_list[i] if i < len(content_list) else ""
        
        # Determine if function calling should be used for this specific response
        per_response_use_function_calling = (
            finish_reason == "tool_calls" and len(tool_calls) > 0
        )
        
        # When tool calls are present, always include both content and tool_calls
        message = (
            openai_api_protocol.ChatCompletionMessage(
                role="assistant", 
                content=content_for_message if content_for_message else "",
                tool_calls=tool_calls if len(tool_calls) > 0 else None
            )
            if per_response_use_function_calling or finish_reason == "error"
            else openai_api_protocol.ChatCompletionMessage(role="assistant", content=output_text)
        )
        
        choices.append(openai_api_protocol.ChatCompletionResponseChoice(
            index=i,
            finish_reason=finish_reason,
            message=message,
            logprobs=(
                openai_api_protocol.LogProbs(content=logprob_results[i])
                if logprob_results is not None and i < len(logprob_results)
                else None
            ),
        ))
    
    return openai_api_protocol.ChatCompletionResponse(
        id=request_id,
        choices=choices,
        model=model,
        system_fingerprint="",
        usage=usage,
    )

def wrap_completion_response(  # pylint: disable=too-many-arguments
    request_id: str,
    model: str,
    output_texts: List[str],
    finish_reasons: List[str],
    logprob_results: List[Optional[openai_api_protocol.CompletionLogProbs]],
    usage: openai_api_protocol.CompletionUsage,
) -> openai_api_protocol.CompletionResponse:
    """Wrap the non-streaming completion results to CompletionResponse instance."""
    return openai_api_protocol.CompletionResponse(
        id=request_id,
        choices=[
            openai_api_protocol.CompletionResponseChoice(
                index=i,
                finish_reason=finish_reason,
                text=output_text,
                logprobs=logprob_results[i],
            )
            for i, (output_text, finish_reason) in enumerate(zip(output_texts, finish_reasons))
        ],
        model=model,
        usage=usage,
    )
