import asyncio
import base64
import json
import multiprocessing
import os
import threading
import uuid
from collections import Counter
from collections.abc import Iterator, Mapping
from typing import Final

import pytest
from integration._support.process import kill_stray_process
from integration._support.wire import Reply, Request, Wire, wire_server
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import Span
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExportResult
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.google_genai.main import agenerate_content_stream
from litellm.integrations.langfuse.langfuse_sdk import _build_span_exporter, resolve_trace_id

REDACTED: Final = "redacted-by-litellm"
_MEDIA_REQUEST: Final = TypeAdapter(Mapping[str, JsonValue])


def _accept(request: Request) -> Reply:
    return Reply()


def _spans(bodies: tuple[bytes, ...]) -> Iterator[Span]:
    for body in bodies:
        for resource_spans in ExportTraceServiceRequest.FromString(body).resource_spans:
            for scope_spans in resource_spans.scope_spans:
                yield from scope_spans.spans


def _generations(bodies: tuple[bytes, ...], trace_id: str) -> tuple[dict[str, str], ...]:
    attributes: Final = (
        {attribute.key: attribute.value.string_value for attribute in span.attributes}
        for span in _spans(bodies)
        if span.trace_id.hex() == trace_id
    )
    return tuple(span for span in attributes if span.get("langfuse.observation.type") == "generation")


def _media_span(name: str) -> ReadableSpan:
    image: Final = b"\x89PNG\r\n\x1a\n" + name.encode()
    span: Final = (
        TracerProvider()
        .get_tracer("langfuse-sdk")
        .start_span(
            name,
            attributes={"langfuse.observation.output": "data:image/png;base64," + base64.b64encode(image).decode()},
        )
    )
    span.end()
    return span


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
@pytest.mark.parametrize("pending", [False, True])
def test_forked_process_exports_media_without_replaying_or_waiting_for_its_parents_spans(
    monkeypatch: pytest.MonkeyPatch, pending: bool
) -> None:
    monkeypatch.setenv("LANGFUSE_MAX_RETRIES", "0")
    release: Final = threading.Event()
    entered: Final = threading.Event()
    initial: Final = _media_span("parent-initial")
    blocked: Final = _media_span("parent-blocked")
    blocked_context: Final = blocked.context
    assert blocked_context is not None
    queued: Final = _media_span("parent-queued")
    child_span: Final = _media_span("child")
    after: Final = _media_span("parent-after")

    def respond(request: Request) -> Reply:
        if request.target != "/api/public/media":
            return Reply()
        body: Final = _MEDIA_REQUEST.validate_json(request.body)
        trace_id: Final = body["traceId"]
        assert isinstance(trace_id, str)
        if trace_id == format(blocked_context.trace_id, "032x"):
            entered.set()
            assert release.wait(10), "parent media request was never released"
        return Reply(
            body=json.dumps({"mediaId": trace_id, "uploadUrl": f"{destination.url}/image/{trace_id}"}).encode()
        )

    with wire_server(respond) as destination:
        exporter: Final = _build_span_exporter(public_key="pk-fork", secret_key="sk-fork", base_url=destination.url)

        def upload_child() -> None:
            try:
                assert exporter.export((child_span,)) is SpanExportResult.SUCCESS
                assert exporter.force_flush(timeout_millis=2000), "child media span was not delivered"
            finally:
                exporter.shutdown()

        child: Final = multiprocessing.get_context("fork").Process(target=upload_child, daemon=True)
        try:
            assert exporter.export((initial,)) is SpanExportResult.SUCCESS
            assert exporter.force_flush(timeout_millis=5000)
            if pending:
                assert exporter.export((blocked, queued)) is SpanExportResult.SUCCESS
                assert entered.wait(5), "parent media request never started"
            child.start()
            child.join(5)
            assert child.exitcode == 0, f"forked media exporter failed or hung, exit code {child.exitcode}"
            release.set()
            assert exporter.export((after,)) is SpanExportResult.SUCCESS
            assert exporter.force_flush(timeout_millis=5000)
        finally:
            release.set()
            if child.pid is not None:
                if child.is_alive():
                    kill_stray_process(child.pid)
                child.join(5)
                assert not child.is_alive(), "forked media exporter survived cleanup"
            exporter.shutdown()
        requests: Final = destination.drain()

    bodies: Final = tuple(request.body for request in requests if request.target.endswith("/traces"))
    expected: Final = (initial, blocked, queued, child_span, after) if pending else (initial, child_span, after)
    assert Counter(span.name for span in _spans(bodies)) == Counter(span.name for span in expected)
    assert Counter(request.body for request in requests if request.method == "PUT") == Counter(
        b"\x89PNG\r\n\x1a\n" + span.name.encode() for span in expected
    )


async def _trace_exports(wire: Wire, trace_id: str, seen: tuple[bytes, ...], attempts: int) -> tuple[bytes, ...]:
    bodies: Final = seen + tuple(
        request.body for request in wire.drain() if request.method == "POST" and request.target.endswith("/traces")
    )
    if attempts == 0 or _generations(bodies, trace_id):
        return bodies
    await asyncio.sleep(0.25)
    return await _trace_exports(wire, trace_id, bodies, attempts - 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_native_google_truncated_output_keeps_images_attached_to_the_trace_preview(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    monkeypatch.setenv("LANGFUSE_MAX_EVENT_SIZE_BYTES", "4096")
    monkeypatch.setattr(litellm, "success_callback", ["langfuse"])
    monkeypatch.setitem(
        litellm.model_cost,
        "gemini/test-image-model",
        {"input_cost_per_token": 0, "output_cost_per_token": 0, "litellm_provider": "gemini", "mode": "chat"},
    )
    image: Final = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    encoded: Final = base64.b64encode(image).decode()
    trace_name: Final = f"native-google-media-{uuid.uuid4()}"
    trace_id: Final = resolve_trace_id(trace_name)
    received: Final[list[Request]] = []
    uploaded: Final = threading.Event()
    response: Final = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "t" * 8192}, {"inlineData": {"mimeType": "image/png", "data": encoded}}],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
    }

    def respond(request: Request) -> Reply:
        received.append(request)
        if request.target.startswith("/models/"):
            return (
                Reply(content_type="text/event-stream", chunks=(b"data: " + json.dumps(response).encode() + b"\n\n",))
                if stream
                else Reply(body=json.dumps(response).encode())
            )
        if request.target == "/api/public/media":
            return Reply(
                body=json.dumps(
                    {"mediaId": "server-image", "uploadUrl": None if uploaded.is_set() else wire.url + "/image"}
                ).encode()
            )
        if request.method == "PUT":
            uploaded.set()
        return Reply()

    with wire_server(respond) as wire:
        kwargs: Final = {
            "model": "gemini/test-image-model",
            "contents": [{"role": "user", "parts": [{"text": "generate an image"}]}],
            "api_base": wire.url,
            "api_key": "local-test-key",
            "metadata": {"trace_id": trace_name},
            "langfuse_public_key": f"pk-{trace_name}",
            "langfuse_secret_key": "sk-local",
            "langfuse_host": wire.url,
        }
        if stream:
            chunks: Final = await agenerate_content_stream(**kwargs)
            _ = [chunk async for chunk in chunks]
        else:
            await litellm.agenerate_content(**kwargs)
        bodies: Final = await _trace_exports(wire, trace_id, (), 120)

    (generation,) = _generations(bodies, trace_id)
    assert generation["langfuse.observation.output"] == "<truncated due to size exceeding limit>"
    registrations: Final = tuple(
        _MEDIA_REQUEST.validate_json(request.body) for request in received if request.target == "/api/public/media"
    )
    trace_attachments: Final = tuple(
        registration for registration in registrations if not registration.get("observationId")
    )
    assert tuple((item["traceId"], item["field"]) for item in trace_attachments) == ((trace_id, "output"),), (
        "Langfuse trace previews read trace_media independently, checked 2026-10-21. "
        "https://github.com/langfuse/langfuse/blob/6343bd819089b5fb34795e8e4bd0f095068d5385/"
        "web/src/server/api/routers/media.ts#L119"
    )
    (upload,) = tuple(request for request in received if request.method == "PUT")
    assert upload.body == image


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_langfuse_export_carries_no_raw_prompt_or_answer_when_message_logging_is_off(
    stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt: Final = f"prompt-{uuid.uuid4()}"
    answer: Final = f"answer-{uuid.uuid4()}"
    trace_name: Final = f"litellm-test-{uuid.uuid4()}"
    monkeypatch.setattr(litellm, "turn_off_message_logging", True)
    monkeypatch.setattr(litellm, "success_callback", ["langfuse"])
    with wire_server(_accept) as wire:
        response: Final = await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            mock_response=answer,
            stream=stream,
            metadata={"trace_id": trace_name},
            langfuse_public_key=f"pk-lf-{trace_name}",
            langfuse_secret_key="sk-lf-local",
            langfuse_host=wire.url,
        )
        if stream:
            _ = [chunk async for chunk in response]
        bodies: Final = await _trace_exports(wire, resolve_trace_id(trace_name), (), 120)

    generations: Final = _generations(bodies, resolve_trace_id(trace_name))
    assert len(generations) == 1, generations
    assert json.loads(generations[0]["langfuse.observation.input"]) == {
        "messages": [{"content": REDACTED, "role": "user"}]
    }
    assert json.loads(generations[0]["langfuse.observation.output"])["content"] == REDACTED
    assert all(prompt.encode() not in body and answer.encode() not in body for body in bodies)
