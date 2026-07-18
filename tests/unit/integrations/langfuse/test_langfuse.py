import asyncio
import json
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Final
from uuid import uuid4

import httpx
import pytest
import respx
import yaml
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue
from pydantic import TypeAdapter

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.integrations.langfuse.langfuse import (
    LangFuseLogger,
    _extract_cache_read_input_tokens,
    _get_langfuse_token_usage,
)
from litellm.integrations.langfuse.langfuse_prompt_management import langfuse_client_init
from litellm.integrations.langfuse.langfuse_sdk import (
    flush_langfuse_tracing,
    resolve_observation_id,
    resolve_trace_id,
)
from litellm.litellm_core_utils import litellm_logging
from litellm.litellm_core_utils.litellm_logging import DynamicLoggingCache as LegacyDynamicLoggingCache
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.litellm_core_utils.thread_pool_executor import MAX_THREADS, executor
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from litellm.types.router import DeploymentTypedDict

_EXPORT_URL: Final = "https://langfuse.unit/api/public/otel/v1/traces"
_VOLATILE_ATTRIBUTES: Final = frozenset(
    {
        "langfuse.observation.completion_start_time",
        "langfuse.observation.metadata.applied_guardrails",
        "langfuse.observation.metadata.cache_hit",
        "langfuse.observation.metadata.hidden_params",
        "langfuse.observation.metadata.litellm_call_id",
        "langfuse.observation.metadata.litellm_response_cost",
        "langfuse.observation.metadata.requester_metadata",
        "langfuse.observation.metadata.response_id",
        "langfuse.observation.metadata.usage_object",
    }
)


def test_langfuse_token_usage_supports_responses_api_usage() -> None:
    usage: Final = ResponseAPIUsage(
        input_tokens=13,
        input_tokens_details={"cached_tokens": 3},
        output_tokens=21,
        total_tokens=34,
    )

    assert _get_langfuse_token_usage(usage) == (13, 21, 34)
    assert _extract_cache_read_input_tokens(usage) == 3


def _decode_attribute(value: AnyValue) -> object:
    match value.WhichOneof("value"):
        case "string_value":
            try:
                return json.loads(value.string_value)
            except json.JSONDecodeError:
                return value.string_value
        case "bool_value":
            return value.bool_value
        case "int_value":
            return value.int_value
        case "double_value":
            return value.double_value
        case "array_value":
            return [_decode_attribute(item) for item in value.array_value.values]
        case _:
            return None


def _exported_spans(route: respx.Route) -> tuple[dict[str, object], ...]:
    exported: Final = tuple(
        ExportTraceServiceRequest.FromString(call.request.content)
        for call in route.calls
    )
    return tuple(
        {
            "trace_id": span.trace_id.hex(),
            "span_id": span.span_id.hex(),
            "parent_span_id": span.parent_span_id.hex() or None,
            "name": span.name,
            "attributes": {
                attribute.key: _decode_attribute(attribute.value)
                for attribute in span.attributes
            },
        }
        for request in exported
        for resource in request.resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    )


def _comparable(span: Mapping[str, object]) -> dict[str, object]:
    attributes: Final = TypeAdapter(dict[str, object]).validate_python(span["attributes"])
    return {
        "name": span["name"],
        "parent_span_id": span["parent_span_id"],
        "attributes": {
            key: value for key, value in sorted(attributes.items()) if key not in _VOLATILE_ATTRIBUTES
        },
    }


def _expected_generation(route: respx.Route, fixture_name: str, trace_id: str) -> dict[str, object]:
    otel_trace_id: Final = resolve_trace_id(trace_id)
    generations: Final = tuple(
        span
        for span in _exported_spans(route)
        if span["trace_id"] == otel_trace_id
        and TypeAdapter(dict[str, object]).validate_python(span["attributes"]).get("langfuse.observation.type")
        == "generation"
    )
    assert len(generations) == 1
    expected_path: Final = Path(__file__).with_name("langfuse_expected_request_body") / fixture_name
    expected: Final = TypeAdapter(dict[str, object]).validate_python(json.loads(expected_path.read_text()))
    actual: Final = _comparable(generations[0])
    expected_comparable: Final = _comparable(expected)
    assert actual == expected_comparable, json.dumps(
        {"actual": actual, "expected": expected_comparable}, indent=2, sort_keys=True
    )
    return actual


def _drain_logging_executor() -> None:
    workers: Final = MAX_THREADS
    gate: Final = threading.Barrier(workers + 1)
    futures: Final = tuple(executor.submit(gate.wait) for _ in range(workers))
    gate.wait()
    for future in futures:
        future.result()


async def _flush_langfuse() -> None:
    loop: Final = asyncio.get_running_loop()
    scheduled_callbacks: Final[asyncio.Future[None]] = loop.create_future()
    loop.call_soon(scheduled_callbacks.set_result, None)
    await scheduled_callbacks
    await GLOBAL_LOGGING_WORKER.flush()
    _drain_logging_executor()
    assert flush_langfuse_tracing()


@pytest.fixture
def langfuse_export(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> Iterator[tuple[respx.Route, str]]:
    trace_id: Final = f"litellm-migration-{uuid4()}"
    route: Final = respx_mock.post(_EXPORT_URL).respond(status_code=200)
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.unit")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", f"{trace_id}-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", f"{trace_id}-secret")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    client_cache: Final = LLMClientCache()
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", client_cache)
    http_client: Final = httpx.Client(transport=httpx.MockTransport(respx_mock.handler))
    client_cache.set_cache(
        key="httpx_client",
        value=HTTPHandler(client=http_client),
        litellm_owned_client=False,
    )
    monkeypatch.setattr(litellm, "success_callback", ["langfuse"])
    monkeypatch.setattr(litellm_logging, "langFuseLogger", None)
    monkeypatch.setattr(litellm_logging, "in_memory_dynamic_logger_cache", LegacyDynamicLoggingCache())
    monkeypatch.setattr(litellm_logging, "_in_memory_loggers", [])
    langfuse_client_init.cache_clear()
    yield route, trace_id
    logger: Final = litellm_logging.in_memory_dynamic_logger_cache.get_cache(
        credentials={}, service_name="langfuse"
    )
    if isinstance(logger, LangFuseLogger):
        logger.stop()
    client_cache.flush_cache()
    http_client.close()


@pytest.mark.asyncio
async def test_langfuse_logging_completion(langfuse_export: tuple[respx.Route, str]) -> None:
    route, trace_id = langfuse_export
    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        mock_response="Hello! How can I assist you today?",
        metadata={"trace_id": trace_id},
    )
    await _flush_langfuse()
    assert route.call_count == 1
    _expected_generation(route, "completion.json", trace_id)


@pytest.mark.asyncio
async def test_langfuse_logging_completion_with_tags(langfuse_export: tuple[respx.Route, str]) -> None:
    route, trace_id = langfuse_export
    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        mock_response="Hello! How can I assist you today?",
        metadata={"trace_id": trace_id, "tags": ["test_tag", "test_tag_2"]},
    )
    await _flush_langfuse()
    assert route.call_count == 1
    _expected_generation(route, "completion_with_tags.json", trace_id)


@pytest.mark.asyncio
async def test_langfuse_logging_completion_with_tags_stream(langfuse_export: tuple[respx.Route, str]) -> None:
    route, trace_id = langfuse_export
    response: Final = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        mock_response="Hello! How can I assist you today?",
        metadata={"trace_id": trace_id, "tags": ["test_tag_stream", "test_tag_2_stream"]},
        stream=True,
    )
    chunks: Final = tuple([chunk async for chunk in response])
    assert chunks
    await _flush_langfuse()
    assert route.call_count == 1
    _expected_generation(route, "completion_with_tags_stream.json", trace_id)


@pytest.mark.asyncio
async def test_langfuse_generation_id_metadata_names_the_exported_observation(
    langfuse_export: tuple[respx.Route, str],
) -> None:
    route, trace_id = langfuse_export
    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        mock_response="Hello! How can I assist you today?",
        metadata={"trace_id": trace_id, "generation_id": "my-generation"},
    )
    await _flush_langfuse()
    generation: Final = next(span for span in _exported_spans(route) if span["trace_id"] == resolve_trace_id(trace_id))
    assert generation["span_id"] == resolve_observation_id("my-generation")
    _expected_generation(route, "completion.json", trace_id)


@pytest.mark.asyncio
async def test_langfuse_logging_completion_with_langfuse_metadata(
    langfuse_export: tuple[respx.Route, str],
) -> None:
    route, trace_id = langfuse_export
    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        mock_response="Hello! How can I assist you today?",
        metadata={
            "trace_id": trace_id,
            "tags": ["test_tag", "test_tag_2"],
            "generation_name": "test_generation_name",
            "parent_observation_id": "test_parent_observation_id",
            "version": "test_version",
            "trace_user_id": "test_user_id",
            "session_id": "test_session_id",
            "trace_name": "test_trace_name",
            "trace_metadata": {"test_key": "test_value"},
            "trace_version": "test_trace_version",
            "trace_release": "test_trace_release",
        },
    )
    await _flush_langfuse()
    assert route.call_count == 1
    _expected_generation(route, "completion_with_langfuse_metadata.json", trace_id)


@pytest.mark.asyncio
async def test_langfuse_logging_with_non_serializable_metadata(
    langfuse_export: tuple[respx.Route, str],
) -> None:
    from pydantic import BaseModel, ConfigDict

    class UserPreferences(BaseModel):
        model_config = ConfigDict(frozen=True)
        favorite_colors: frozenset[str]
        settings: dict[str, object]

    route, trace_id = langfuse_export
    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        mock_response="Hello! How can I assist you today?",
        metadata={
            "user_prefs": UserPreferences(favorite_colors=frozenset({"red", "blue"}), settings={"theme": "dark"}),
            "nested_set": {"inner_set": frozenset({1, 2, 3})},
            "trace_id": trace_id,
        },
    )
    await _flush_langfuse()
    assert route.call_count == 1
    _expected_generation(route, "completion_with_complex_metadata.json", trace_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "fixture_name"),
    [
        ({"a": 1, "b": 2, "c": 3}, "simple_metadata.json"),
        ({"a": {"nested_a": 1}, "b": {"nested_b": 2}}, "nested_metadata.json"),
        ({"a": [1, 2, 3], "b": {4, 5, 6}}, "simple_metadata2.json"),
        ({"a": (1, 2), "b": frozenset({3, 4}), "c": {"d": [5, 6]}}, "simple_metadata3.json"),
        ({"lock": threading.Lock()}, "metadata_with_lock.json"),
        ({"func": lambda x: x + 1}, "metadata_with_function.json"),
        (
            {
                "int": 42,
                "str": "hello",
                "list": [1, 2, 3],
                "set": {4, 5},
                "dict": {"nested": "value"},
                "non_copyable": threading.Lock(),
                "function": print,
            },
            "complex_metadata.json",
        ),
        ({"list": ["list", "not", "a", "dict"]}, "complex_metadata_2.json"),
        ({}, "empty_metadata.json"),
    ],
)
async def test_langfuse_logging_with_various_metadata_types(
    langfuse_export: tuple[respx.Route, str], metadata: dict[str, object], fixture_name: str
) -> None:
    route, trace_id = langfuse_export
    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        mock_response="Hello! How can I assist you today?",
        metadata={**metadata, "trace_id": trace_id},
    )
    await _flush_langfuse()
    assert route.call_count == 1
    _expected_generation(route, fixture_name, trace_id)


@pytest.mark.asyncio
async def test_langfuse_logging_completion_with_malformed_llm_response(
    langfuse_export: tuple[respx.Route, str],
) -> None:
    route, trace_id = langfuse_export
    response: Final = litellm.ModelResponse(
        choices=[],
        usage=litellm.Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        model="gpt-3.5-turbo",
        object="chat.completion",
        created=1723081200,
    ).model_dump()
    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        mock_response=response,
        metadata={"trace_id": trace_id},
    )
    await _flush_langfuse()
    assert route.call_count == 1
    _expected_generation(route, "completion_with_no_choices.json", trace_id)


@pytest.mark.asyncio
async def test_langfuse_logging_completion_with_bedrock_llm_response(
    langfuse_export: tuple[respx.Route, str],
) -> None:
    route, trace_id = langfuse_export
    response: Final = litellm.ModelResponse(
        choices=[],
        usage=litellm.Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        model="anthropic.claude-haiku-4-5-20251001-v1:0",
        object="chat.completion",
        created=1723081200,
    ).model_dump()
    await litellm.acompletion(
        model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=[{"role": "user", "content": "Hello!"}],
        mock_response=response,
        metadata={"trace_id": trace_id},
        aws_access_key_id="fake-key",
        aws_secret_access_key="fake-key",
        aws_region="us-east-1",
    )
    await _flush_langfuse()
    assert route.call_count == 1
    _expected_generation(route, "completion_with_bedrock_call.json", trace_id)


@pytest.mark.asyncio
async def test_langfuse_logging_completion_with_vertex_llm_response(
    langfuse_export: tuple[respx.Route, str],
) -> None:
    route, trace_id = langfuse_export
    response: Final = litellm.ModelResponse(
        choices=[],
        usage=litellm.Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        model="vertex/gemini-3-flash-preview",
        object="chat.completion",
        created=1723081200,
    ).model_dump()
    await litellm.acompletion(
        model="vertex_ai/gemini-3-flash-preview",
        messages=[{"role": "user", "content": "Hello!"}],
        mock_response=response,
        metadata={"trace_id": trace_id},
        vertex_credentials="my-mock-credentials",
        api_key="my-mock-credentials-2",
    )
    await _flush_langfuse()
    assert route.call_count == 1
    _expected_generation(route, "completion_with_vertex_call.json", trace_id)


@pytest.mark.asyncio
async def test_langfuse_logging_vllm_embedding(
    langfuse_export: tuple[respx.Route, str], respx_mock: respx.MockRouter
) -> None:
    route, trace_id = langfuse_export
    embedding_url: Final = "http://my-fake-vllm.com/v1/embeddings"
    upstream: Final = respx_mock.post(embedding_url).mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "model": "BAAI/bge-small-en-v1.5",
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            },
        )
    )
    await litellm.aembedding(
        model="hosted_vllm/BAAI/bge-small-en-v1.5",
        input=["Hello from litellm!"],
        api_base="http://my-fake-vllm.com/v1",
        metadata={"trace_id": trace_id},
    )
    await _flush_langfuse()
    assert upstream.call_count == 1
    expected_path: Final = Path(__file__).with_name("langfuse_expected_request_body") / "embedding_with_vllm.json"
    assert TypeAdapter(dict[str, object]).validate_json(upstream.calls.last.request.content) == json.loads(
        expected_path.read_text()
    )
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_langfuse_logging_with_router(langfuse_export: tuple[respx.Route, str]) -> None:
    route, trace_id = langfuse_export
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "mock_response": "Hello! How can I assist you today?",
                    "api_key": "test_api_key",
                },
            }
        ]
    )
    await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        mock_response=litellm.ModelResponse(
            choices=[],
            usage=litellm.Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            model="gpt-3.5-turbo",
            object="chat.completion",
            created=1723081200,
        ).model_dump(),
        metadata={"trace_id": trace_id},
    )
    await _flush_langfuse()
    assert route.call_count == 1
    _expected_generation(route, "completion_with_router.json", trace_id)


@pytest.mark.asyncio
async def test_langfuse_e2e_sync(langfuse_export: tuple[respx.Route, str]) -> None:
    route, trace_id = langfuse_export
    litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "sync callback route"}],
        mock_response="Hello from litellm",
        metadata={"trace_id": trace_id},
    )
    await _flush_langfuse()
    assert route.call_count == 1
    assert str(route.calls[0].request.url).endswith("/api/public/otel/v1/traces")
    assert _exported_spans(route)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ("openai_chat", "gemini_chat", "gemini_native"))
@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("scenario", ("primary", "fallback_request", "fallback_published", "fallback_free"))
@pytest.mark.parametrize("usage_multiplier", (1, 100))
async def test_configured_request_prices_reach_langfuse_exports(
    langfuse_export: tuple[respx.Route, str],
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    surface: str,
    stream: bool,
    scenario: str,
    usage_multiplier: int,
) -> None:
    route, trace_id = langfuse_export
    provider: Final = "openai" if surface == "openai_chat" else "gemini"
    model: Final = "unit-langfuse-request-price"
    input_rate: Final = 0.001
    output_rate: Final = 0.002
    primary_price: Final = 0.01
    backup_price: Final = 0.0 if scenario == "fallback_free" else 0.02
    fail_primary: Final = scenario != "primary"
    prompt_tokens: Final = 10 * usage_multiplier
    completion_tokens: Final = 20 * usage_multiplier
    litellm.register_model({f"{provider}/{model}": {
        "litellm_provider": provider, "mode": "chat",
        "input_cost_per_token": input_rate, "output_cost_per_token": output_rate,
    }})
    backup_config: Final = (
        "" if scenario == "fallback_published" else f"    output_cost_per_request: {backup_price}\n"
    )
    models: Final = TypeAdapter(
        list[DeploymentTypedDict], config={"arbitrary_types_allowed": True}
    ).validate_python(yaml.safe_load(f"""
- model_name: primary
  litellm_params:
    model: {provider}/{model}
    api_key: unused
    api_base: https://primary.test/v1
  model_info:
    id: langfuse-primary-price
    output_cost_per_request: {primary_price}
- model_name: backup
  litellm_params:
    model: {provider}/{model}
    api_key: unused
    api_base: https://backup.test/v1
  model_info:
    id: langfuse-backup-price
{backup_config}"""))
    router: Final = litellm.Router(
        model_list=models, fallbacks=[{"primary": ["backup"]}], num_retries=0,
        max_fallbacks=1, disable_cooldowns=True,
    )
    body: Final = (
        {
            "candidates": [{
                "content": {"role": "model", "parts": [
                    {"text": "done"}, {"inlineData": {"mimeType": "image/png", "data": "aW1hZ2U="}},
                ]},
                "finishReason": "STOP", "index": 0,
            }],
            "usageMetadata": {
                "promptTokenCount": prompt_tokens, "candidatesTokenCount": completion_tokens,
                "totalTokenCount": prompt_tokens + completion_tokens,
            },
        }
        if provider == "gemini"
        else {
            "id": "chatcmpl-priced", "object": "chat.completion", "created": 1, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )
    chunk: Final = (
        body if provider == "gemini" else {
            **body, "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
        }
    )
    success: Final = (
        httpx.Response(200, text=f"data: {json.dumps(chunk)}\n\n", headers={"content-type": "text/event-stream"})
        if stream else httpx.Response(200, json=body)
    )
    primary: Final = respx_mock.post(url__startswith="https://primary.test/").mock(return_value=(
        httpx.Response(500, json={"error": {"message": "primary failed", "type": "server_error", "status": "INTERNAL"}})
        if fail_primary else success
    ))
    backup: Final = (
        respx_mock.post(url__startswith="https://backup.test/").mock(return_value=success) if fail_primary else None
    )
    response: Final = (
        await router.agenerate_content_stream(
            model="primary", contents=[{"role": "user", "parts": [{"text": "hello"}]}],
            metadata={"trace_id": trace_id}, stream=True,
        )
        if surface == "gemini_native" and stream
        else await router.agenerate_content(
            model="primary", contents=[{"role": "user", "parts": [{"text": "hello"}]}],
            metadata={"trace_id": trace_id},
        )
        if surface == "gemini_native"
        else await router.acompletion(
            model="primary", messages=[{"role": "user", "content": "hello"}], stream=stream,
            metadata={"trace_id": trace_id},
        )
    )
    if stream:
        existing_tasks: Final = frozenset(asyncio.all_tasks())
        chunks: Final = tuple([chunk async for chunk in response])
        assert chunks
        if surface == "gemini_native":
            await asyncio.gather(*(task for task in asyncio.all_tasks() if task not in existing_tasks))
    else:
        assert response is not None
    await _flush_langfuse()
    assert primary.call_count == 1
    if backup is not None:
        assert backup.call_count == 1
    generations: Final = tuple(
        span for span in _exported_spans(route)
        if span["trace_id"] == resolve_trace_id(trace_id)
        and TypeAdapter(dict[str, object]).validate_python(span["attributes"]).get("langfuse.observation.type")
        == "generation"
    )
    assert len(generations) == 1, _exported_spans(route)
    attributes: Final = TypeAdapter(dict[str, object]).validate_python(generations[0]["attributes"])
    expected: Final = (
        primary_price if not fail_primary
        else prompt_tokens * input_rate + completion_tokens * output_rate
        if scenario == "fallback_published" else backup_price
    )
    assert attributes["langfuse.observation.cost_details"] == {"total": pytest.approx(expected)}
    assert attributes["langfuse.observation.metadata.litellm_response_cost"] == pytest.approx(expected)
    usage: Final = TypeAdapter(dict[str, int]).validate_python(attributes["langfuse.observation.usage_details"])
    assert usage["input"] == prompt_tokens
    assert usage["output"] == completion_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize("allowance", (0.0, 0.005, 0.02))
async def test_request_prices_keep_finalized_allowance_discounts_in_langfuse(
    langfuse_export: tuple[respx.Route, str], local_model_cost_map: None, allowance: float,
) -> None:
    from datetime import datetime

    from litellm.litellm_core_utils.llm_cost_calc.finalization import finalize_cost
    from litellm.types.utils import ModelResponse, Usage

    litellm.utils.add_custom_logger_callback_to_specific_event("langfuse", "success")
    route, trace_id = langfuse_export
    model: Final = "openai/unit-finalized-request-price"
    price: Final = 0.02
    router: Final = litellm.Router(model_list=[{
        "model_name": "priced", "litellm_params": {"model": model, "api_key": "unused"},
        "model_info": {"id": "finalized-request-price", "output_cost_per_request": price},
    }])
    now: Final = datetime(2026, 1, 1)
    logger: Final = litellm_logging.Logging(
        model=model, messages=[], stream=False, call_type="acompletion", start_time=now,
        litellm_call_id=trace_id, function_id="finalized-request-price",
    )
    logger.update_environment_variables(optional_params={}, litellm_params={"metadata": {
        "model_info": router.registry().all()[0]["model_info"], "trace_id": trace_id,
    }}, custom_llm_provider="openai")
    response: Final = ModelResponse(model=model, usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30))
    prepared: Final = logger.prepare_response_cost(response, now)
    assert prepared is not None
    assert prepared.amount == pytest.approx(price)
    logger.record_finalized_cost(finalize_cost(prepared.amount, logger.cost_breakdown, allowance))

    await logger.async_success_handler(response, now, now)
    await _flush_langfuse()

    generations: Final = tuple(
        span for span in _exported_spans(route) if span["trace_id"] == resolve_trace_id(trace_id)
    )
    assert len(generations) == 1, _exported_spans(route)
    attributes: Final = TypeAdapter(dict[str, object]).validate_python(generations[0]["attributes"])
    expected: Final = price - allowance
    assert logger.model_call_details["response_cost"] == pytest.approx(expected)
    assert attributes["langfuse.observation.cost_details"] == {"total": pytest.approx(expected)}
    assert attributes["langfuse.observation.metadata.litellm_response_cost"] == pytest.approx(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", ("aembedding", litellm.CallTypes.aresponses_websocket.value))
@pytest.mark.parametrize("request_price", (None, 0.0, 0.02))
async def test_request_prices_leave_other_langfuse_costs_usage_based(
    langfuse_export: tuple[respx.Route, str], local_model_cost_map: None,
    call_type: str, request_price: float | None,
) -> None:
    from datetime import datetime

    from litellm.types.utils import EmbeddingResponse, Usage

    litellm.utils.add_custom_logger_callback_to_specific_event("langfuse", "success")
    route, trace_id = langfuse_export
    model: Final = "openai/unit-langfuse-request-price-other-call"
    input_rate: Final = 0.001
    output_rate: Final = 0.002
    litellm.register_model({model: {
        "litellm_provider": "openai",
        "mode": "chat" if call_type == litellm.CallTypes.aresponses_websocket.value else "embedding",
        "input_cost_per_token": input_rate, "output_cost_per_token": output_rate,
    }})
    router: Final = litellm.Router(model_list=[{
        "model_name": "request-priced", "litellm_params": {"model": model, "api_key": "unused"},
        "model_info": {
            "id": "langfuse-request-price-other-call",
            **({"output_cost_per_request": request_price} if request_price is not None else {}),
        },
    }])
    now: Final = datetime(2026, 1, 1)
    logger: Final = litellm_logging.Logging(
        model=model, messages=[], stream=False, call_type=call_type, start_time=now,
        litellm_call_id=trace_id, function_id="langfuse-other-call-price",
    )
    logger.update_environment_variables(optional_params={}, litellm_params={
        "custom_llm_provider": "openai", "metadata": {
            "model_info": router.model_list[0]["model_info"], "trace_id": trace_id,
        },
    }, custom_llm_provider="openai")
    response: Final = (
        [{"type": "response.completed", "response": {
            "model": model, "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        }}] if call_type == litellm.CallTypes.aresponses_websocket.value else EmbeddingResponse(
            model=model, data=[], usage=Usage(prompt_tokens=10, total_tokens=10),
        )
    )
    await logger.async_success_handler(
        result=logger.normalize_logging_result(response), start_time=now, end_time=now,
    )
    await _flush_langfuse()
    generations: Final = tuple(
        span for span in _exported_spans(route) if span["trace_id"] == resolve_trace_id(trace_id)
    )
    assert len(generations) == 1, _exported_spans(route)
    attributes: Final = TypeAdapter(dict[str, object]).validate_python(generations[0]["attributes"])
    expected: Final = 10 * input_rate + (
        20 * output_rate if call_type == litellm.CallTypes.aresponses_websocket.value else 0.0
    )
    assert attributes["langfuse.observation.cost_details"] == {"total": pytest.approx(expected)}
    assert attributes["langfuse.observation.metadata.litellm_response_cost"] == pytest.approx(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("request_price", (0.0, 0.02))
@pytest.mark.parametrize("usage_multiplier", (1, 100))
async def test_background_request_price_reaches_langfuse_on_settlement(
    langfuse_export: tuple[respx.Route, str], local_model_cost_map: None,
    request_price: float, usage_multiplier: int,
) -> None:
    from datetime import datetime

    from litellm_enterprise.proxy.common_utils.check_responses_cost import (
        BilledBackgroundResponse,
        log_billed_background_response,
    )

    from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY, MANAGED_BACKGROUND_RESPONSE_METADATA_KEY
    from litellm.types.utils import BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN

    litellm.utils.add_custom_logger_callback_to_specific_event("langfuse", "success")
    route, trace_id = langfuse_export
    model: Final = "openai/unit-langfuse-background-price"
    router: Final = litellm.Router(model_list=[
        {
            "model_name": "primary", "litellm_params": {"model": model, "api_key": "unused"},
            "model_info": {"id": "background-primary-price", "output_cost_per_request": 0.01},
        },
        {
            "model_name": "backup", "litellm_params": {"model": model, "api_key": "unused"},
            "model_info": {"id": "background-backup-price", "output_cost_per_request": request_price},
        },
    ])
    metadata: Final = {
        "model_info": router.model_list[1]["model_info"], "trace_id": trace_id,
        MANAGED_BACKGROUND_RESPONSE_METADATA_KEY: True,
    }
    response: Final = ResponsesAPIResponse(
        id="resp_background_priced", object="response", created_at=1, status="completed", model=model,
        background=True, output=[], usage=ResponseAPIUsage(
            input_tokens=10 * usage_multiplier, output_tokens=20 * usage_multiplier,
            total_tokens=30 * usage_multiplier,
        ),
    )
    response.hidden_params = {"response_cost": 0.0}
    for call_type in ("aresponses", "aget_responses"):
        logger: Final = litellm_logging.Logging(
            model=model, messages=[], stream=False, call_type=call_type, start_time=datetime(2026, 1, 1),
            litellm_call_id=f"background-{call_type}", function_id="background-price",
        )
        logger.update_environment_variables(optional_params={}, litellm_params={"metadata": metadata})
        assert logger.response_cost_calculator(response) == 0.0
    billed: Final = await log_billed_background_response(BilledBackgroundResponse(
        response=response, model=model, custom_llm_provider="openai", litellm_params={},
        metadata={**metadata, INTERNAL_CALL_ORIGIN_METADATA_KEY: BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN},
        creator_logging={},
    ))
    await _flush_langfuse()
    assert billed.cost == pytest.approx(request_price)
    generations: Final = tuple(
        span for span in _exported_spans(route) if span["trace_id"] == resolve_trace_id(trace_id)
    )
    assert len(generations) == 1, _exported_spans(route)
    attributes: Final = TypeAdapter(dict[str, object]).validate_python(generations[0]["attributes"])
    assert attributes["langfuse.observation.cost_details"] == {"total": pytest.approx(request_price)}
    assert attributes["langfuse.observation.metadata.litellm_response_cost"] == pytest.approx(request_price)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_retrieval", (False, True))
@pytest.mark.parametrize("request_price", (0.0, 0.02))
@pytest.mark.parametrize("status", ("queued", "in_progress", "completed"))
async def test_background_retrieval_exports_configured_request_price(
    langfuse_export: tuple[respx.Route, str], local_model_cost_map: None,
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch,
    async_retrieval: bool, request_price: float, status: str,
) -> None:
    route, trace_id = langfuse_export
    model: Final = "openai/unit-sync-background-price"
    discount: Final = 0.25
    margin: Final = 0.5
    monkeypatch.setattr(litellm, "cost_discount_config", {model: discount})
    monkeypatch.setattr(litellm, "cost_margin_config", {"openai": margin})
    router: Final = litellm.Router(model_list=[{
        "model_name": "background", "litellm_params": {"model": model, "api_key": "unused"},
        "model_info": {"id": "sync-background-request-price", "output_cost_per_request": request_price},
    }])
    response_id: Final = "resp_sync_background_price"
    respx_mock.get(f"https://background.unit/v1/responses/{response_id}").respond(json={
        "id": response_id, "object": "response", "created_at": 1, "model": model,
        "background": True, "status": status, "output": [],
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
    })
    response: Final = (
        await litellm.aget_responses(
            response_id=response_id, custom_llm_provider="openai", api_key="unused",
            api_base="https://background.unit/v1",
            metadata={"model_info": router.model_list[0]["model_info"], "trace_id": trace_id},
        ) if async_retrieval else litellm.get_responses(
            response_id=response_id, custom_llm_provider="openai", api_key="unused",
            api_base="https://background.unit/v1",
            metadata={"model_info": router.model_list[0]["model_info"], "trace_id": trace_id},
        )
    )
    assert isinstance(response, ResponsesAPIResponse)
    assert response.status == status
    await _flush_langfuse()
    generations: Final = tuple(
        span for span in _exported_spans(route) if span["trace_id"] == resolve_trace_id(trace_id)
    )
    assert len(generations) == 1, _exported_spans(route)
    attributes: Final = TypeAdapter(dict[str, object]).validate_python(generations[0]["attributes"])
    expected: Final = request_price * (1 - discount) * (1 + margin) if status == "completed" else 0.0
    assert attributes["langfuse.observation.cost_details"] == {"total": pytest.approx(expected)}
    assert attributes["langfuse.observation.metadata.litellm_response_cost"] == pytest.approx(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("request_price", (None, 0.0, 0.02))
async def test_native_embeddings_keep_token_cost_in_langfuse(
    langfuse_export: tuple[respx.Route, str], local_model_cost_map: None,
    request_price: float | None,
) -> None:
    from datetime import datetime

    from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging

    litellm.utils.add_custom_logger_callback_to_specific_event("langfuse", "success")
    route, trace_id = langfuse_export
    model: Final = "openai/unit-native-embedding-fee"
    input_rate: Final = 0.001
    prompt_tokens: Final = 10
    litellm.register_model({model: {
        "litellm_provider": "openai", "mode": "embedding", "input_cost_per_token": input_rate,
        "output_cost_per_token": 0.0,
    }})
    router: Final = litellm.Router(model_list=[{
        "model_name": "embedding", "litellm_params": {"model": model, "api_key": "unused"},
        "model_info": {
            "id": "native-embedding-fee",
            **({"output_cost_per_request": request_price} if request_price is not None else {}),
        },
    }])
    now: Final = datetime(2026, 1, 1)
    logger: Final = litellm_logging.Logging(
        model=model, messages=[], stream=False, call_type="pass_through_endpoint",
        start_time=now, litellm_call_id=trace_id, function_id="native-embedding-price",
    )
    logger.update_environment_variables(optional_params={}, litellm_params={"metadata": {
        "model_info": router.model_list[0]["model_info"], "trace_id": trace_id,
    }}, custom_llm_provider="openai")
    url: Final = "https://api.openai.com/v1/embeddings"
    request: Final = {"model": model, "input": "embedding billing"}
    body: Final = {
        "model": model, "object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }
    await PassThroughEndpointLogging().pass_through_async_success_handler(
        httpx_response=httpx.Response(200, json=body, request=httpx.Request("POST", url)),
        response_body=body, logging_obj=logger,
        url_route=url, result=json.dumps(body), start_time=now, end_time=now, cache_hit=False,
        request_body=request, passthrough_logging_payload={"url": url, "request_body": request},
        custom_llm_provider="openai",
    )
    await _flush_langfuse()
    generations: Final = tuple(
        span for span in _exported_spans(route) if span["trace_id"] == resolve_trace_id(trace_id)
    )
    assert len(generations) == 1, _exported_spans(route)
    attributes: Final = TypeAdapter(dict[str, object]).validate_python(generations[0]["attributes"])
    expected: Final = prompt_tokens * input_rate
    assert logger.model_call_details["response_cost"] == pytest.approx(expected)
    assert attributes["langfuse.observation.cost_details"] == {"total": pytest.approx(expected)}
    assert attributes["langfuse.observation.metadata.litellm_response_cost"] == pytest.approx(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("request_price", (None, 0.0, 0.02))
async def test_request_prices_preserve_final_batch_settlement_in_langfuse(
    langfuse_export: tuple[respx.Route, str], local_model_cost_map: None,
    request_price: float | None,
) -> None:
    from datetime import datetime

    from litellm.types.utils import LiteLLMBatch, Usage

    litellm.utils.add_custom_logger_callback_to_specific_event("langfuse", "success")
    route, trace_id = langfuse_export
    model: Final = "openai/unit-final-batch-price"
    router: Final = litellm.Router(model_list=[{
        "model_name": "batch-model", "litellm_params": {"model": model, "api_key": "unused"},
        "model_info": {
            "id": "final-batch-price",
            **({"output_cost_per_request": request_price} if request_price is not None else {}),
        },
    }])
    now: Final = datetime(2026, 1, 1)
    logger: Final = litellm_logging.Logging(
        model=model, messages=[], stream=False, call_type="aretrieve_batch",
        start_time=now, litellm_call_id=trace_id, function_id="final-batch-price",
    )
    logger.update_environment_variables(optional_params={}, litellm_params={"metadata": {
        "model_info": router.model_list[0]["model_info"], "trace_id": trace_id,
    }}, custom_llm_provider="openai")
    batch: Final = LiteLLMBatch(
        id="batch_settled", object="batch", endpoint="/v1/chat/completions", completion_window="24h",
        input_file_id="file_input", status="completed", created_at=1, output_file_id="file_output",
        request_counts={"completed": 1, "failed": 0, "total": 1},
    )
    settled_cost: Final = 0.15
    await logger.async_success_handler(
        result=batch, start_time=now, end_time=now, batch_cost=settled_cost,
        batch_usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30), batch_models=[model],
        batch_successful_requests=1, batch_failed_requests=0,
    )
    await _flush_langfuse()
    generations: Final = tuple(
        span for span in _exported_spans(route) if span["trace_id"] == resolve_trace_id(trace_id)
    )
    assert len(generations) == 1, _exported_spans(route)
    attributes: Final = TypeAdapter(dict[str, object]).validate_python(generations[0]["attributes"])
    assert attributes["langfuse.observation.cost_details"] == {"total": pytest.approx(settled_cost)}
    assert attributes["langfuse.observation.metadata.litellm_response_cost"] == pytest.approx(settled_cost)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ("azure", "azure_ai"))
@pytest.mark.parametrize("native_path, surface", (
    ("openai/responses", "responses"), ("openai/v1/responses", "responses"),
    ("openai/deployments/priced/responses", "responses"), ("priced/openai/responses", "responses"),
    ("priced/openai/v1/responses", "responses"), ("priced/responses", "responses"),
    ("models/responses", "responses"), ("priced/models/responses", "responses"),
    ("models/chat/completions", "chat"), ("priced/models/chat/completions", "chat"),
))
@pytest.mark.parametrize("request_price", (None, 0.0, 0.02))
@pytest.mark.parametrize("stream", (False, True))
async def test_native_azure_reservation_covers_the_langfuse_charge(
    langfuse_export: tuple[respx.Route, str], local_model_cost_map: None,
    provider: str, native_path: str, surface: str, request_price: float | None, stream: bool,
) -> None:
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.proxy.spend_tracking.budget_reservation import estimate_request_max_cost

    route, trace_id = langfuse_export
    model: Final = f"{provider}/unit-native-responses"
    input_rate: Final = 0.00001
    output_rate: Final = 0.00002
    litellm.register_model({model: {
        "litellm_provider": provider, "mode": "chat", "input_cost_per_token": input_rate,
        "output_cost_per_token": output_rate,
    }})
    router: Final = litellm.Router(model_list=[{
        "model_name": "priced", "litellm_params": {
            "model": model, "api_key": "unused", "api_base": "https://native-azure.unit",
            "api_version": "unit-version",
        },
        "model_info": {
            "id": "native-azure-responses-price",
            **({"output_cost_per_request": request_price} if request_price is not None else {}),
        },
    }], num_retries=0)
    request: Final = {"model": "priced", "stream": stream, **(
        {"input": "hello", "max_output_tokens": 20} if surface == "responses"
        else {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 20}
    )}
    body: Final = {
        "id": "resp_unit", "object": "response", "created_at": 1, "model": model, "status": "completed",
        "output": [], "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
    } if surface == "responses" else {
        "id": "chatcmpl_unit", "object": "chat.completion", "created": 1, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    quote: Final = estimate_request_max_cost(
        request_body=request, route=f"/{provider}/{native_path}", llm_router=router,
        input_token_counts={"priced": 10},
    )
    event: Final = {"type": "response.completed", "response": body} if surface == "responses" else {
        **body, "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
    }
    upstream_response: Final = httpx.Response(
        200, text=f"data: {json.dumps(event)}\n\n",
        headers={"content-type": "text/event-stream"},
    ) if stream else httpx.Response(200, json=body)
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(lambda _: upstream_response))
    try:
        response: Final = await router.allm_passthrough_route(
            model="priced", method="POST", endpoint=native_path, data=request, stream=stream,
            metadata={"trace_id": trace_id}, client=client,
        )
        if stream:
            chunks: Final = tuple([chunk async for chunk in response])
            assert chunks
        else:
            assert response.status_code == 200
    finally:
        await client.close()
    await _flush_langfuse()
    generations: Final = tuple(
        span for span in _exported_spans(route) if span["trace_id"] == resolve_trace_id(trace_id)
    )
    assert len(generations) == 1, _exported_spans(route)
    attributes: Final = TypeAdapter(dict[str, object]).validate_python(generations[0]["attributes"])
    expected: Final = request_price if request_price is not None else 10 * input_rate + 20 * output_rate
    assert attributes["langfuse.observation.metadata.litellm_response_cost"] == pytest.approx(expected)
    assert quote == pytest.approx(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("request_price", (None, 0.0, 0.02))
@pytest.mark.parametrize("action", ("converse", "invoke"))
async def test_native_bedrock_reservation_covers_the_langfuse_charge(
    langfuse_export: tuple[respx.Route, str], local_model_cost_map: None,
    request_price: float | None, action: str,
) -> None:
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.proxy.spend_tracking.budget_reservation import estimate_request_max_cost

    route, trace_id = langfuse_export
    model: Final = "bedrock/anthropic.unit-native-price"
    input_rate: Final = 0.00001
    output_rate: Final = 0.00002
    litellm.register_model({model: {
        "litellm_provider": "bedrock", "mode": "chat", "input_cost_per_token": input_rate,
        "output_cost_per_token": output_rate,
    }})
    router: Final = litellm.Router(model_list=[{
        "model_name": "priced", "litellm_params": {
            "model": model, "aws_access_key_id": "unit-key", "aws_secret_access_key": "unit-secret",
            "aws_region_name": "us-west-2",
        },
        "model_info": {
            "id": "native-bedrock-price",
            **({"output_cost_per_request": request_price} if request_price is not None else {}),
        },
    }], num_retries=0)
    request: Final = {
        "messages": [{"role": "user", "content": [{"text": "hello"}]}], "inferenceConfig": {"maxTokens": 20},
    } if action == "converse" else {
        "messages": [{"role": "user", "content": "hello"}], "max_tokens": 20,
        "anthropic_version": "bedrock-2023-05-31",
    }
    body: Final = {
        "output": {"message": {"role": "assistant", "content": [{"text": "done"}]}}, "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30}, "metrics": {"latencyMs": 1},
    } if action == "converse" else {
        "id": "msg_unit", "type": "message", "role": "assistant", "model": model,
        "content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }
    quote: Final = estimate_request_max_cost(
        request_body=request, route=f"/bedrock/model/priced/{action}", llm_router=router,
        input_token_counts={"priced": 10},
    )
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(
        lambda upstream: httpx.Response(200, json=body, request=upstream),
    ))
    try:
        response: Final = await router.allm_passthrough_route(
            model="priced", method="POST", endpoint=f"/model/priced/{action}", data=request,
            metadata={"trace_id": trace_id}, client=client,
        )
    finally:
        await client.close()
    assert response.status_code == 200
    await _flush_langfuse()
    generations: Final = tuple(
        span for span in _exported_spans(route) if span["trace_id"] == resolve_trace_id(trace_id)
    )
    assert len(generations) == 1, _exported_spans(route)
    attributes: Final = TypeAdapter(dict[str, object]).validate_python(generations[0]["attributes"])
    expected: Final = request_price if request_price is not None else 10 * input_rate + 20 * output_rate
    assert attributes.get("langfuse.observation.cost_details", {"total": 0.0}) == {"total": pytest.approx(expected)}
    assert attributes["langfuse.observation.metadata.litellm_response_cost"] == pytest.approx(expected)
    assert quote == pytest.approx(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("request_price", (None, 0.0, 0.02))
@pytest.mark.parametrize("modality", ("audio", "image", "video", "output_image"))
async def test_request_prices_preserve_multimodal_batch_settlement_in_langfuse(
    langfuse_export: tuple[respx.Route, str], local_model_cost_map: None,
    respx_mock: respx.MockRouter, request_price: float | None, modality: str,
) -> None:
    from datetime import datetime, timezone

    from litellm.types.utils import LiteLLMBatch

    litellm.utils.add_custom_logger_callback_to_specific_event("langfuse", "success")
    route, trace_id = langfuse_export
    model: Final = "openai/unit-multimodal-batch"
    text_input_rate: Final = 0.001
    text_output_rate: Final = 0.002
    modality_rate: Final = 0.004
    litellm.register_model({model: {
        "litellm_provider": "openai", "mode": "chat",
        "input_cost_per_token": 0.01, "output_cost_per_token": 0.02,
        "input_cost_per_token_batches": text_input_rate, "output_cost_per_token_batches": text_output_rate,
        **({"output_cost_per_image_token_batches": modality_rate} if modality == "output_image"
           else {f"input_cost_per_{modality}_token_batches": modality_rate}),
    }})
    router: Final = litellm.Router(model_list=[{
        "model_name": "batch-model", "litellm_params": {"model": model, "api_key": "unused"},
        "model_info": {
            "id": "multimodal-batch-settlement",
            **({"output_cost_per_request": request_price} if request_price is not None else {}),
        },
    }])
    now: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    logger: Final = litellm_logging.Logging(
        model=model, messages=[], stream=False, call_type="aretrieve_batch",
        start_time=now, litellm_call_id=trace_id, function_id="multimodal-batch-settlement",
    )
    logger.update_environment_variables(optional_params={}, litellm_params={"api_key": "unused", "metadata": {
        "model_info": router.model_list[0]["model_info"], "trace_id": trace_id,
    }}, custom_llm_provider="openai")
    usage: Final = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, **(
        {"completion_tokens_details": {"image_tokens": 30}} if modality == "output_image"
        else {"prompt_tokens_details": {f"{modality}_tokens": 30}}
    )}
    row: Final = {"id": "request_multimodal", "custom_id": "request_1", "error": None, "response": {
        "status_code": 200, "body": {"model": model, "usage": usage},
    }}
    respx_mock.get("https://api.openai.com/v1/files/file_multimodal/content").respond(
        status_code=200, text=json.dumps(row) + "\n",
    )
    batch: Final = LiteLLMBatch(
        id="batch_multimodal", object="batch", endpoint="/v1/chat/completions", completion_window="24h",
        input_file_id="file_input", status="completed", created_at=1, output_file_id="file_multimodal",
        request_counts={"completed": 1, "failed": 0, "total": 1},
    )
    await logger.async_success_handler(result=batch, start_time=now, end_time=now)
    await _flush_langfuse()
    expected: Final = (
        100 * text_input_rate + 20 * text_output_rate + 30 * modality_rate if modality == "output_image"
        else 70 * text_input_rate + 30 * modality_rate + 50 * text_output_rate
    )
    generations: Final = tuple(
        span for span in _exported_spans(route) if span["trace_id"] == resolve_trace_id(trace_id)
    )
    assert len(generations) == 1, _exported_spans(route)
    attributes: Final = TypeAdapter(dict[str, object]).validate_python(generations[0]["attributes"])
    assert logger.model_call_details["response_cost"] == pytest.approx(expected)
    assert attributes["langfuse.observation.cost_details"] == {"total": pytest.approx(expected)}
    assert attributes["langfuse.observation.metadata.litellm_response_cost"] == pytest.approx(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ("anthropic", "openai", "vertex_ai"))
@pytest.mark.parametrize("operation", ("create", "poll", "cancel"))
@pytest.mark.parametrize("request_price", (None, 0.0, 0.02))
async def test_native_batch_management_stays_unbilled_in_langfuse(
    langfuse_export: tuple[respx.Route, str], local_model_cost_map: None,
    operation: str, request_price: float | None, provider: str,
) -> None:
    from datetime import datetime

    from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging

    litellm.utils.add_custom_logger_callback_to_specific_event("langfuse", "success")
    route, trace_id = langfuse_export
    model: Final = f"{provider}/unit-native-batch-price"
    router: Final = litellm.Router(model_list=[{
        "model_name": "batch-model", "litellm_params": {"model": model, "api_key": "unused"},
        "model_info": {
            "id": "native-batch-management-price",
            **({"output_cost_per_request": request_price} if request_price is not None else {}),
        },
    }])
    now: Final = datetime(2026, 1, 1)
    logger: Final = litellm_logging.Logging(
        model=model, messages=[], stream=False, call_type="pass_through_endpoint",
        start_time=now, litellm_call_id=trace_id, function_id="native-batch-management",
    )
    logger.update_environment_variables(optional_params={}, litellm_params={"metadata": {
        "model_info": router.model_list[0]["model_info"], "trace_id": trace_id,
    }}, custom_llm_provider=provider)
    urls: Final = {
        "anthropic": "https://api.anthropic.com/v1/messages/batches",
        "openai": "https://api.openai.com/v1/batches",
        "vertex_ai": "https://us-aiplatform.googleapis.com/v1/projects/p/locations/us/batchPredictionJobs",
    }
    url: Final = urls[provider] + (
        "" if operation == "create" else "/msgbatch_priced/cancel" if operation == "cancel" else "/msgbatch_priced"
    )
    request: Final = {"model": model, "requests": []} if operation == "create" else {}
    bodies: Final = {"anthropic": {
        "id": "msgbatch_priced", "type": "message_batch", "processing_status": "in_progress",
        "created_at": "2026-01-01T00:00:00Z", "expires_at": "2026-01-02T00:00:00Z",
        "ended_at": None, "cancel_initiated_at": None, "archived_at": None, "results_url": None,
        "request_counts": {"processing": 3, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0},
    }, "openai": {
        "id": "batch_priced", "object": "batch", "endpoint": "/v1/chat/completions", "created_at": 1,
        "input_file_id": "file_input", "status": "in_progress", "completion_window": "24h",
        "request_counts": {"total": 3, "completed": 0, "failed": 0},
    }, "vertex_ai": {
        "name": "projects/p/locations/us/batchPredictionJobs/batch_priced", "model": model,
        "state": "JOB_STATE_RUNNING", "createTime": "2026-01-01T00:00:00Z",
        "inputConfig": {"instancesFormat": "jsonl", "gcsSource": {"uris": ["gs://unit/input.jsonl"]}},
        "outputConfig": {"predictionsFormat": "jsonl", "gcsDestination": {"outputUriPrefix": "gs://unit/output"}},
    }}
    body: Final = bodies[provider]
    await PassThroughEndpointLogging().pass_through_async_success_handler(
        httpx_response=httpx.Response(200, json=body), response_body=body, logging_obj=logger,
        url_route=url, result=json.dumps(body), start_time=now, end_time=now, cache_hit=False,
        request_body=request, passthrough_logging_payload={"url": url, "request_body": request},
        custom_llm_provider=provider,
    )
    await _flush_langfuse()
    generations: Final = tuple(
        span for span in _exported_spans(route) if span["trace_id"] == resolve_trace_id(trace_id)
    )
    assert len(generations) == 1, _exported_spans(route)
    attributes: Final = TypeAdapter(dict[str, object]).validate_python(generations[0]["attributes"])
    assert logger.model_call_details["response_cost"] == 0.0
    assert attributes.get("langfuse.observation.cost_details", {"total": 0.0}) == {"total": 0.0}
    assert attributes["langfuse.observation.metadata.litellm_response_cost"] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ("gemini", "vertex_ai"))
@pytest.mark.parametrize("request_price", (0.0, 0.02))
@pytest.mark.parametrize("seconds", (4, 8))
async def test_native_video_keeps_duration_cost_in_langfuse(
    langfuse_export: tuple[respx.Route, str], local_model_cost_map: None,
    provider: str, request_price: float, seconds: int,
) -> None:
    from datetime import datetime, timezone

    from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging

    litellm.utils.add_custom_logger_callback_to_specific_event("langfuse", "success")
    route, trace_id = langfuse_export
    model: Final = f"{provider}/unit-native-video"
    rate: Final = 0.04
    litellm.register_model({model: {
        "litellm_provider": provider, "mode": "video", "output_cost_per_second": rate,
    }})
    router: Final = litellm.Router(model_list=[{
        "model_name": "priced", "litellm_params": {"model": model, "api_key": "unused"},
        "model_info": {"id": "native-video-request-price", "output_cost_per_request": request_price},
    }])
    now: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    logger: Final = litellm_logging.Logging(
        model=model, messages=[], stream=False, call_type="pass_through_endpoint",
        start_time=now, litellm_call_id=trace_id, function_id="video-request-price",
    )
    logger.update_environment_variables(optional_params={}, litellm_params={"metadata": {
        "model_info": router.model_list[0]["model_info"], "trace_id": trace_id,
    }}, custom_llm_provider=provider)
    base: Final = (
        "https://generativelanguage.googleapis.com/v1beta/models" if provider == "gemini"
        else "https://us-aiplatform.googleapis.com/v1/projects/p/locations/us/publishers/google/models"
    )
    url: Final = f"{base}/unit-native-video:predictLongRunning"
    body: Final = {"name": "operations/unit_video"}
    request: Final = {"instances": [{"prompt": "make a video"}], "parameters": {"durationSeconds": seconds}}
    await PassThroughEndpointLogging().pass_through_async_success_handler(
        httpx_response=httpx.Response(200, json=body), response_body=body, logging_obj=logger,
        url_route=url, result=json.dumps(body), start_time=now, end_time=now, cache_hit=False,
        request_body=request, passthrough_logging_payload={"url": url, "request_body": request},
        custom_llm_provider=provider,
    )
    await _flush_langfuse()
    generations: Final = tuple(
        span for span in _exported_spans(route) if span["trace_id"] == resolve_trace_id(trace_id)
    )
    assert len(generations) == 1, _exported_spans(route)
    attributes: Final = TypeAdapter(dict[str, object]).validate_python(generations[0]["attributes"])
    expected: Final = seconds * rate
    assert logger.model_call_details["response_cost"] == pytest.approx(expected)
    assert attributes["langfuse.observation.cost_details"] == {"total": pytest.approx(expected)}
    assert attributes["langfuse.observation.metadata.litellm_response_cost"] == pytest.approx(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider, url", (
    ("vertex_ai", "https://us-aiplatform.googleapis.com/v1/projects/p/locations/us/"
     "publishers/google/models/priced:countTokens"),
    ("gemini", "https://generativelanguage.googleapis.com/v1beta/models/priced:countTokens"),
    ("openai", "https://api.openai.com/v1/responses/input_tokens"),
))
async def test_token_counting_passthrough_stays_free_in_langfuse(
    langfuse_export: tuple[respx.Route, str], local_model_cost_map: None, provider: str, url: str,
) -> None:
    from datetime import datetime, timezone

    from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging

    litellm.utils.add_custom_logger_callback_to_specific_event("langfuse", "success")
    route, trace_id = langfuse_export
    model: Final = f"{provider}/unit-counting-price"
    router: Final = litellm.Router(model_list=[{
        "model_name": "priced", "litellm_params": {"model": model, "api_key": "unused"},
        "model_info": {"id": "token-counting-export-price", "output_cost_per_request": 0.02},
    }])
    now: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    logger: Final = litellm_logging.Logging(
        model=model, messages=[], stream=False, call_type="pass_through_endpoint",
        start_time=now, litellm_call_id=trace_id, function_id="token-counting-export",
    )
    logger.update_environment_variables(optional_params={}, litellm_params={"metadata": {
        "model_info": router.model_list[0]["model_info"], "trace_id": trace_id,
    }}, custom_llm_provider=provider)
    body: Final = {"totalTokens": 17}
    request: Final = {"model": model}
    await PassThroughEndpointLogging().pass_through_async_success_handler(
        httpx_response=httpx.Response(200, json=body, request=httpx.Request("POST", url)),
        response_body=body, logging_obj=logger,
        url_route=url, result=json.dumps(body), start_time=now, end_time=now, cache_hit=False,
        request_body=request, passthrough_logging_payload={"url": url, "request_body": request},
        custom_llm_provider=provider,
    )
    await _flush_langfuse()
    generations: Final = tuple(
        span for span in _exported_spans(route) if span["trace_id"] == resolve_trace_id(trace_id)
    )
    assert len(generations) == 1, _exported_spans(route)
    attributes: Final = TypeAdapter(dict[str, object]).validate_python(generations[0]["attributes"])
    assert logger.model_call_details["response_cost"] == 0.0
    assert attributes.get("langfuse.observation.cost_details", {"total": 0.0}) == {"total": 0.0}
    assert attributes["langfuse.observation.metadata.litellm_response_cost"] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("scenario", ("primary", "fallback_request", "fallback_free", "fallback_published"))
async def test_native_relay_request_prices_reach_langfuse_exports(
    langfuse_export: tuple[respx.Route, str], local_model_cost_map: None, respx_mock: respx.MockRouter,
    stream: bool, scenario: str,
) -> None:
    route, trace_id = langfuse_export
    model: Final = "hosted_vllm/unit-native-relay-price"
    primary_price: Final = 0.01
    backup_price: Final = 0.0 if scenario == "fallback_free" else 0.02
    fail_primary: Final = scenario != "primary"
    input_rate: Final = 0.001
    output_rate: Final = 0.002
    litellm.register_model({model: {
        "litellm_provider": "hosted_vllm", "mode": "chat",
        "input_cost_per_token": input_rate, "output_cost_per_token": output_rate,
    }})
    router: Final = litellm.Router(model_list=[
        {
            "model_name": "primary", "litellm_params": {
                "model": model, "api_key": "unused", "api_base": "https://primary.test",
            },
            "model_info": {"id": "native-relay-primary", "output_cost_per_request": primary_price},
        },
        {
            "model_name": "backup", "litellm_params": {
                "model": model, "api_key": "unused", "api_base": "https://backup.test",
            },
            "model_info": {
                "id": "native-relay-backup",
                **({"output_cost_per_request": backup_price} if scenario != "fallback_published" else {}),
            },
        },
    ], fallbacks=[{"primary": ["backup"]}], num_retries=0, max_fallbacks=1, disable_cooldowns=True)
    body: Final = {
        "id": "chatcmpl-native-relay", "object": "chat.completion", "model": model, "created": 1,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    chunk: Final = {
        **body, "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
    }
    success: Final = (
        httpx.Response(200, text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
                       headers={"content-type": "text/event-stream"})
        if stream else httpx.Response(200, json=body)
    )
    primary: Final = respx_mock.post(url__startswith="https://primary.test/").mock(return_value=(
        httpx.Response(500, json={"error": {"message": "primary failed", "type": "server_error"}})
        if fail_primary else success
    ))
    backup: Final = (
        respx_mock.post(url__startswith="https://backup.test/").mock(return_value=success) if fail_primary else None
    )
    response: Final = await router.allm_passthrough_route(
        model="primary", method="POST", endpoint="v1/chat/completions", metadata={"trace_id": trace_id},
        json={"model": "primary", "messages": [{"role": "user", "content": "hello"}],
              **({"stream": True} if stream else {})},
    )
    if stream:
        chunks: Final = tuple([chunk async for chunk in response])
        assert chunks
        await response.wait_for_spend_flush()
    else:
        assert response.status_code == 200
    await _flush_langfuse()
    assert primary.call_count == 1
    if backup is not None:
        assert backup.call_count == 1
    generations: Final = tuple(
        span for span in _exported_spans(route) if span["trace_id"] == resolve_trace_id(trace_id)
    )
    assert len(generations) == 1, _exported_spans(route)
    attributes: Final = TypeAdapter(dict[str, object]).validate_python(generations[0]["attributes"])
    expected: Final = (
        primary_price if not fail_primary
        else 10 * input_rate + 20 * output_rate if scenario == "fallback_published" else backup_price
    )
    assert attributes["langfuse.observation.cost_details"] == {"total": pytest.approx(expected)}
    assert attributes["langfuse.observation.metadata.litellm_response_cost"] == pytest.approx(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("request_price", (0.0, 0.02))
@pytest.mark.parametrize("reported_cost", (None, 0.0, 0.5))
async def test_native_openai_stream_exports_request_price_after_reassembly(
    langfuse_export: tuple[respx.Route, str], local_model_cost_map: None,
    request_price: float, reported_cost: float | None,
) -> None:
    from datetime import datetime

    from litellm.proxy.pass_through_endpoints.streaming_handler import PassThroughStreamingHandler
    from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging
    from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType

    litellm.utils.add_custom_logger_callback_to_specific_event("langfuse", "success")
    route, trace_id = langfuse_export
    model: Final = "openai/unit-native-stream-request-price"
    litellm.register_model({model: {
        "litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 0.001, "output_cost_per_token": 0.002,
    }})
    router: Final = litellm.Router(model_list=[{
        "model_name": "native-stream", "litellm_params": {"model": model, "api_key": "unused"},
        "model_info": {"id": "native-stream-price", "output_cost_per_request": request_price},
    }])
    now: Final = datetime(2026, 1, 1)
    logger: Final = litellm_logging.Logging(
        model=model, messages=[], stream=True, call_type="pass_through_endpoint",
        start_time=now, litellm_call_id=trace_id, function_id="native-stream-price",
    )
    logger.update_environment_variables(optional_params={}, litellm_params={"metadata": {
        "model_info": router.model_list[0]["model_info"], "trace_id": trace_id,
    }})
    logger.model_call_details["custom_llm_provider"] = "openai"
    logger.model_call_details["response_cost"] = 0.3
    chunk: Final = {
        "id": "chatcmpl-native-stream", "object": "chat.completion.chunk", "created": 1, "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30,
                  **({"cost": reported_cost} if reported_cost is not None else {})},
    }
    await PassThroughStreamingHandler.route_streaming_logging_to_handler(
        litellm_logging_obj=logger, passthrough_success_handler_obj=PassThroughEndpointLogging(),
        url_route="https://api.openai.com/v1/chat/completions", request_body={"model": model},
        endpoint_type=EndpointType.OPENAI, start_time=now, end_time=now,
        raw_bytes=[f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()], model=model,
    )
    await _flush_langfuse()
    generations: Final = tuple(
        span for span in _exported_spans(route) if span["trace_id"] == resolve_trace_id(trace_id)
    )
    assert len(generations) == 1, _exported_spans(route)
    attributes: Final = TypeAdapter(dict[str, object]).validate_python(generations[0]["attributes"])
    assert logger.model_call_details["response_cost"] == pytest.approx(request_price)
    assert attributes["langfuse.observation.cost_details"] == {"total": pytest.approx(request_price)}
    assert attributes["langfuse.observation.metadata.litellm_response_cost"] == pytest.approx(request_price)


@pytest.mark.asyncio
@pytest.mark.parametrize("request_price", (None, 0.0, 0.02))
async def test_sync_native_gemini_stream_exports_configured_request_price(
    langfuse_export: tuple[respx.Route, str], local_model_cost_map: None,
    respx_mock: respx.MockRouter, request_price: float | None,
) -> None:
    route, trace_id = langfuse_export
    model: Final = "gemini/unit-sync-native-stream-price"
    input_rate: Final = 0.001
    output_rate: Final = 0.002
    litellm.register_model({model: {
        "litellm_provider": "gemini", "mode": "chat",
        "input_cost_per_token": input_rate, "output_cost_per_token": output_rate,
    }})
    router: Final = litellm.Router(model_list=[{
        "model_name": "native-stream", "litellm_params": {
            "model": model, "api_key": "unused", "api_base": "https://google.test/v1beta",
        },
        "model_info": {
            "id": "sync-native-stream-price",
            **({"output_cost_per_request": request_price} if request_price is not None else {}),
        },
    }])
    body: Final = {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "done"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20, "totalTokenCount": 30},
    }
    vendor: Final = respx_mock.post(url__startswith="https://google.test/").respond(
        200, text=f"data: {json.dumps(body)}\n\n", headers={"content-type": "text/event-stream"},
    )
    response: Final = router.generate_content_stream(
        model="native-stream", contents=[{"role": "user", "parts": [{"text": "hello"}]}],
        metadata={"trace_id": trace_id}, stream=True,
    )
    chunks: Final = tuple(response)
    assert chunks
    assert next(response, None) is None
    await _flush_langfuse()
    assert vendor.call_count == 1
    generations: Final = tuple(
        span for span in _exported_spans(route) if span["trace_id"] == resolve_trace_id(trace_id)
    )
    assert len(generations) == 1, _exported_spans(route)
    attributes: Final = TypeAdapter(dict[str, object]).validate_python(generations[0]["attributes"])
    expected: Final = 10 * input_rate + 20 * output_rate if request_price is None else request_price
    assert attributes["langfuse.observation.cost_details"] == {"total": pytest.approx(expected)}
    assert attributes["langfuse.observation.metadata.litellm_response_cost"] == pytest.approx(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ("generations", "edits"))
@pytest.mark.parametrize("image_count", (1, 3))
@pytest.mark.parametrize("request_price", (0.0, 0.02))
async def test_native_image_request_price_reaches_langfuse_exports(
    langfuse_export: tuple[respx.Route, str], local_model_cost_map: None,
    endpoint: str, image_count: int, request_price: float,
) -> None:
    from datetime import datetime

    from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging

    litellm.utils.add_custom_logger_callback_to_specific_event("langfuse", "success")
    route, trace_id = langfuse_export
    model: Final = "openai/unit-langfuse-native-image-price"
    router: Final = litellm.Router(model_list=[
        {
            "model_name": "primary", "litellm_params": {"model": model, "api_key": "unused"},
            "model_info": {"id": "langfuse-image-primary", "output_cost_per_request": 0.01},
        },
        {
            "model_name": "backup", "litellm_params": {"model": model, "api_key": "unused"},
            "model_info": {"id": "langfuse-image-backup", "output_cost_per_request": request_price},
        },
    ])
    now: Final = datetime(2026, 1, 1)
    logger: Final = litellm_logging.Logging(
        model=model, messages=[], stream=False, call_type="pass_through_endpoint",
        start_time=now, litellm_call_id=trace_id, function_id="native-image-price",
    )
    logger.update_environment_variables(optional_params={}, litellm_params={"metadata": {
        "model_info": router.model_list[1]["model_info"], "trace_id": trace_id,
    }})
    url: Final = f"https://api.openai.com/v1/images/{endpoint}"
    request: Final = {"model": model, "n": image_count, "size": "1024x1024"}
    body: Final = {"data": [{"url": "https://images.test/generated.png"} for _ in range(image_count)]}
    await PassThroughEndpointLogging().pass_through_async_success_handler(
        httpx_response=httpx.Response(200, json=body, request=httpx.Request("POST", url)),
        response_body=body, logging_obj=logger,
        url_route=url, result=json.dumps(body), start_time=now, end_time=now, cache_hit=False,
        request_body=request, passthrough_logging_payload={"url": url, "request_body": request},
        custom_llm_provider="openai",
    )
    await _flush_langfuse()
    generations: Final = tuple(
        span for span in _exported_spans(route) if span["trace_id"] == resolve_trace_id(trace_id)
    )
    assert len(generations) == 1, _exported_spans(route)
    attributes: Final = TypeAdapter(dict[str, object]).validate_python(generations[0]["attributes"])
    assert attributes["langfuse.observation.cost_details"] == {"total": pytest.approx(request_price)}
    assert attributes["langfuse.observation.metadata.litellm_response_cost"] == pytest.approx(request_price)
