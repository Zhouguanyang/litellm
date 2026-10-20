from __future__ import annotations

import logging
import os
import re
import threading
from base64 import b64encode
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import partial, reduce
from hashlib import sha256
from importlib.metadata import version
from itertools import chain, count
from json import dumps
from queue import Empty, Full, Queue
from time import monotonic, sleep
from types import MappingProxyType
from typing import Final, Literal, ParamSpec, TypeVar
from urllib.parse import quote
from weakref import ref

import backoff
import httpx
import opentelemetry.trace as otel_trace
from langfuse import LangfuseOtelSpanAttributes
from langfuse._utils.serializer import EventSerializer
from langfuse.api import GetMediaUploadUrlResponse, LangfuseAPI, MediaContentType, Prompt, Prompt_Chat
from langfuse.api.core import http_client as langfuse_http_client
from langfuse.api.core.api_error import ApiError
from langfuse.api.core.request_options import RequestOptions
from langfuse.media import LangfuseMedia
from langfuse.model import BasePromptClient, ChatPromptClient, PromptClient, TextPromptClient
from opentelemetry import context as otel_context
from opentelemetry.context import (
    _SUPPRESS_INSTRUMENTATION_KEY,  # pyright: ignore[reportPrivateUsage]  # SDK exporter suppression key
    Context,
)
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.sdk.trace.sampling import ALWAYS_ON, Decision, Sampler, SamplingResult
from opentelemetry.trace import Link, NonRecordingSpan, Span, SpanContext, SpanKind, TraceFlags, Tracer, TraceState
from opentelemetry.util.types import Attributes, AttributeValue
from pydantic import ConfigDict, JsonValue, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.langfuse.langfuse import PROMPT_CACHE_TTL_ENV, parse_langfuse_debug, whole_number
from litellm.litellm_core_utils.prompt_templates.common_utils import base64_media_data_uri
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.llms.base import LiteLLMBaseModel

__all__ = (
    "AuthCheckFailure",
    "DiscardingSpanExporter",
    "LangfuseApiClient",
    "LangfuseObservation",
    "LangfusePromptError",
    "LangfuseSpanExporter",
    "LangfuseTracing",
    "TraceIdHashSampler",
    "acquire_langfuse_tracing",
    "build_langfuse_client",
    "build_langfuse_tracing",
    "configured_flush_at",
    "configured_max_retries",
    "configured_release",
    "configured_sample_rate",
    "configured_timeout",
    "contains_langfuse_media",
    "enable_langfuse_debug_logging",
    "flush_langfuse_tracing",
    "observation_attributes",
    "release_langfuse_tracing",
    "resolve_observation_id",
    "resolve_trace_id",
    "start_child_span",
    "start_generation",
    "to_unix_nanos",
    "trace_attributes",
)

_TRACE_ID_PATTERN: Final = re.compile(r"^(?=.*[1-9a-f])[0-9a-f]{32}$")
_OBSERVATION_ID_PATTERN: Final = re.compile(r"^(?=.*[1-9a-f])[0-9a-f]{16}$")
_TRACER_NAME: Final = "langfuse-sdk"
_LANGFUSE_INGESTION_VERSION_HEADER: Final = "x-langfuse-ingestion-version"
_LANGFUSE_INGESTION_VERSION: Final = "4"
_NO_REST_RETRIES: Final = RequestOptions(max_retries=0)
_TRUNCATION_MARKER: Final = "<truncated due to size exceeding limit>"
_METADATA_PREFIXES: Final = (LangfuseOtelSpanAttributes.OBSERVATION_METADATA, LangfuseOtelSpanAttributes.TRACE_METADATA)
_OUTPUT_PREFIXES: Final = (LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT, LangfuseOtelSpanAttributes.TRACE_OUTPUT)
_TRUNCATION_GROUPS: Final = (
    _OUTPUT_PREFIXES,
    _METADATA_PREFIXES,
)
_MEDIA_FIELDS: Final = (
    (LangfuseOtelSpanAttributes.OBSERVATION_INPUT, "input"),
    (LangfuseOtelSpanAttributes.TRACE_INPUT, "input"),
    (LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT, "output"),
    (LangfuseOtelSpanAttributes.TRACE_OUTPUT, "output"),
    (LangfuseOtelSpanAttributes.OBSERVATION_METADATA, "metadata"),
    (LangfuseOtelSpanAttributes.TRACE_METADATA, "metadata"),
)
_MEDIA_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_MediaParams: Final = ParamSpec("_MediaParams")
_MediaResult: Final = TypeVar("_MediaResult")
_SERVER_FLOOR_HINT: Final = (
    "; the OTLP traces route needs a self-hosted Langfuse server on 3.63.0 or newer "
    "(https://langfuse.com/self-hosting/upgrade/versioning#sdk-server)"
)
_langfuse_logger: Final = logging.getLogger("langfuse")
_MAX_QUEUE_SIZE: Final = 100_000
_DEFAULT_FLUSH_AT: Final = 512
_CHANNEL_RETIRE_GRACE_SECONDS: Final = 60.0
_DEFAULT_TIMEOUT_SECONDS: Final = 20.0
_DEFAULT_MAX_RETRIES: Final = 3
_DEFAULT_MAX_EVENT_SIZE_BYTES: Final = 1_000_000
_MAX_RETRIES: Final = 1_000
_MAX_BACKOFF_EXPONENT: Final = 6
_DEFAULT_PROMPT_CACHE_TTL_SECONDS: Final = 60.0
_JSON_SAFE_INT: Final = 2**53 - 1
_COMMON_RELEASE_ENVS: Final = (
    "RENDER_GIT_COMMIT",
    "CI_COMMIT_SHA",
    "CIRCLE_SHA1",
    "SOURCE_VERSION",
    "TRAVIS_COMMIT",
    "GIT_COMMIT",
    "GITHUB_SHA",
    "BITBUCKET_COMMIT",
    "BUILD_SOURCEVERSION",
    "DRONE_COMMIT_SHA",
)
_SPAN_LIMITS: Final = SpanLimits(
    max_attributes=SpanLimits.UNSET,
    max_events=128,
    max_links=128,
    max_span_attributes=SpanLimits.UNSET,
    max_event_attributes=128,
    max_link_attributes=128,
    max_attribute_length=SpanLimits.UNSET,
    max_span_attribute_length=SpanLimits.UNSET,
)


def to_unix_nanos(value: datetime | float | None) -> int | None:
    """Langfuse v4 takes OTel timestamps, which are integer nanoseconds since the epoch.

    Guardrail entries carry unix seconds as floats rather than datetimes, so both
    shapes have to convert; the v2 SDK accepted either through a pydantic model.
    """
    if value is None:
        return None
    seconds: Final = value.timestamp() if isinstance(value, datetime) else float(value)
    return int(seconds * 1_000_000_000)


def resolve_trace_id(trace_id: object | None) -> str:
    """Map a caller's trace id onto the 32 lowercase hex characters v4 requires."""
    serialized: Final = "" if trace_id is None else str(trace_id)
    normalized: Final = serialized.lower().replace("-", "")
    if _TRACE_ID_PATTERN.fullmatch(normalized):
        return normalized
    if not serialized:
        return format(RandomIdGenerator().generate_trace_id(), "032x")
    return sha256(serialized.encode("utf-8")).digest()[:16].hex()


def resolve_observation_id(observation_id: object | None) -> str | None:
    """Map a caller's parent observation id onto v4's 16 lowercase hex characters."""
    serialized: Final = "" if observation_id is None else str(observation_id)
    normalized: Final = serialized.lower().replace("-", "")
    if _OBSERVATION_ID_PATTERN.fullmatch(normalized):
        return normalized
    if not serialized:
        return None
    return sha256(serialized.encode("utf-8")).digest()[:8].hex()


def _serialize(value: object) -> str | None:
    return value if value is None or isinstance(value, str) else safe_dumps(value)


def _string_or_none(value: object) -> str | None:
    return None if value is None else str(value)


def _serialize_datetime(value: object) -> str | None:
    """A datetime the way the SDK's ``EventSerializer`` sends one: a JSON string, naive values read as local time."""
    if isinstance(value, datetime):
        return safe_dumps(value.astimezone().isoformat())
    return _serialize(value)


def _strings(items: Iterable[object]) -> tuple[str, ...]:
    return tuple(str(item) for item in items)


def _string_sequence(value: object) -> Sequence[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set, frozenset)):
        return _strings(value) or None
    return (str(value),)


def _present(entries: Iterable[tuple[str, AttributeValue | None]]) -> Mapping[str, AttributeValue]:
    return MappingProxyType({key: value for key, value in entries if value is not None})


def _metadata_value(value: object) -> AttributeValue | None:
    """A metadata value as it survives the trip: OTLP drops ints past int64 and a JSON reader rounds ints past
    2**53, so those go as strings, which is how v2's readback showed them."""
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and -_JSON_SAFE_INT <= value <= _JSON_SAFE_INT:
        return value
    return _serialize(value)


def _flattened_metadata(prefix: str, metadata: object) -> Mapping[str, AttributeValue]:
    """Mirror the SDK's wire shape: one ``<prefix>.<key>`` attribute per key, or ``<prefix>`` for a non-dict."""
    if metadata is None:
        return _present(())
    if not isinstance(metadata, Mapping):
        return _present(((prefix, _serialize(metadata)),))
    return _present((f"{prefix}.{key}", _metadata_value(value)) for key, value in metadata.items())


def trace_attributes(
    *,
    name: object = None,
    user_id: object = None,
    session_id: object = None,
    version: object = None,
    release: object = None,
    tags: object = None,
    metadata: object = None,
    public: bool | None = None,
    input: object = None,
    output: object = None,
) -> Mapping[str, AttributeValue]:
    """Trace-level fields ride on an observation's span as ``langfuse.trace.*`` style attributes in v4.

    On the root observation they define the trace; on a continuation they update it, which is
    how v2's ``trace(...)`` and ``update_trace_keys`` contracts map onto the OTLP ingestion.
    """
    scalar: Final[tuple[tuple[str, str | bool | None], ...]] = (
        (LangfuseOtelSpanAttributes.TRACE_NAME, _string_or_none(name)),
        (LangfuseOtelSpanAttributes.TRACE_USER_ID, _string_or_none(user_id)),
        (LangfuseOtelSpanAttributes.TRACE_SESSION_ID, _string_or_none(session_id)),
        (LangfuseOtelSpanAttributes.VERSION, _string_or_none(version)),
        (LangfuseOtelSpanAttributes.RELEASE, _string_or_none(release)),
        (LangfuseOtelSpanAttributes.TRACE_PUBLIC, public),
        (LangfuseOtelSpanAttributes.TRACE_INPUT, _serialize(input)),
        (LangfuseOtelSpanAttributes.TRACE_OUTPUT, _serialize(output)),
    )
    tags_entry: Final[tuple[str, Sequence[str] | None]] = (
        LangfuseOtelSpanAttributes.TRACE_TAGS,
        _string_sequence(tags),
    )
    return _present(
        chain(scalar, (tags_entry,), _flattened_metadata(LangfuseOtelSpanAttributes.TRACE_METADATA, metadata).items())
    )


def observation_attributes(
    *,
    observation_type: Literal["generation", "span"],
    input: object = None,
    output: object = None,
    metadata: object = None,
    level: object = None,
    status_message: object = None,
    version: object = None,
    model: object = None,
    model_parameters: object = None,
    usage_details: object = None,
    cost_details: object = None,
    completion_start_time: object = None,
    prompt: object = None,
) -> Mapping[str, AttributeValue]:
    """The observation's own fields, serialized the way the SDK's ``create_generation_attributes`` does.

    ``prompt`` links the generation to a managed prompt only when it is a real prompt client;
    v2 dropped anything else, and a fallback prompt has no server-side version to link.
    """
    linked_prompt: Final = prompt if isinstance(prompt, BasePromptClient) and not prompt.is_fallback else None
    scalar: Final[tuple[tuple[str, str | int | None], ...]] = (
        (LangfuseOtelSpanAttributes.OBSERVATION_TYPE, observation_type),
        (LangfuseOtelSpanAttributes.OBSERVATION_LEVEL, _string_or_none(level)),
        (LangfuseOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE, _string_or_none(status_message)),
        (LangfuseOtelSpanAttributes.VERSION, _string_or_none(version)),
        (LangfuseOtelSpanAttributes.OBSERVATION_INPUT, _serialize(input)),
        (LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT, _serialize(output)),
        (LangfuseOtelSpanAttributes.OBSERVATION_MODEL, _string_or_none(model)),
        (LangfuseOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS, _serialize(model_parameters)),
        (LangfuseOtelSpanAttributes.OBSERVATION_USAGE_DETAILS, _serialize(usage_details)),
        (LangfuseOtelSpanAttributes.OBSERVATION_COST_DETAILS, _serialize(cost_details)),
        (LangfuseOtelSpanAttributes.OBSERVATION_COMPLETION_START_TIME, _serialize_datetime(completion_start_time)),
        (LangfuseOtelSpanAttributes.OBSERVATION_PROMPT_NAME, linked_prompt.name if linked_prompt else None),
        (LangfuseOtelSpanAttributes.OBSERVATION_PROMPT_VERSION, linked_prompt.version if linked_prompt else None),
    )
    return _present(
        chain(scalar, _flattened_metadata(LangfuseOtelSpanAttributes.OBSERVATION_METADATA, metadata).items())
    )


@dataclass(frozen=True, slots=True)
class LangfuseObservation:
    """A Langfuse observation as the OTel span litellm exports for it."""

    span: Span
    public: bool | None

    @property
    def id(self) -> str:
        return format(self.span.get_span_context().span_id, "016x")

    @property
    def trace_id(self) -> str:
        return format(self.span.get_span_context().trace_id, "032x")

    def end(self, end_time: datetime | float | None = None) -> None:
        self.span.end(end_time=to_unix_nanos(end_time))


_requested_trace_id: Final[ContextVar[int | None]] = ContextVar("litellm_langfuse_requested_trace_id", default=None)
_requested_span_id: Final[ContextVar[int | None]] = ContextVar("litellm_langfuse_requested_span_id", default=None)


class _RequestedIdGenerator(RandomIdGenerator):
    """Hand out the ids the calling context asked for, random otherwise.

    v2 took caller trace and generation ids as plain fields; OTel derives both from
    the tracer's id generator, so the request rides on a context variable instead.
    """

    def generate_trace_id(self) -> int:
        requested: Final = _requested_trace_id.get()
        return super().generate_trace_id() if requested is None else requested

    def generate_span_id(self) -> int:
        requested: Final = _requested_span_id.get()
        return super().generate_span_id() if requested is None else requested


def _parent_context(*, trace_id: str, parent_observation_id: str | None, existing_trace: bool) -> Context:
    """Where a new observation hangs: nowhere for a fresh trace, under a remote parent when continuing one.

    ``existing_trace`` is the v2 ``existing_trace_id`` contract: the trace is appended to, never
    rewritten. The server takes a root observation's name and I/O as the trace's, so a continuation
    without a known parent hangs under a parent id that is never exported instead of claiming root.
    An explicitly empty context also keeps the caller's own active span out of the picture.
    """
    if parent_observation_id is None and not existing_trace:
        return Context()
    parent_span_id: Final = (
        int(parent_observation_id, 16) if parent_observation_id is not None else RandomIdGenerator().generate_span_id()
    )
    remote_parent: Final = NonRecordingSpan(
        SpanContext(
            trace_id=int(trace_id, 16),
            span_id=parent_span_id,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
    )
    return otel_trace.set_span_in_context(remote_parent)


def _start_span(
    tracer: Tracer,
    *,
    name: str,
    context: Context,
    start_time: datetime | float | None,
    trace_id: str | None,
    observation_id: str | None,
    attributes: Mapping[str, AttributeValue],
) -> Span:
    trace_token: Final = _requested_trace_id.set(int(trace_id, 16) if trace_id is not None else None)
    span_token: Final = _requested_span_id.set(int(observation_id, 16) if observation_id is not None else None)
    try:
        return tracer.start_span(
            name=name, context=context, start_time=to_unix_nanos(start_time), attributes=attributes
        )
    finally:
        _requested_span_id.reset(span_token)
        _requested_trace_id.reset(trace_token)


def start_generation(
    *,
    tracing: LangfuseTracing,
    trace_id: str,
    parent_observation_id: str | None,
    existing_trace: bool,
    observation_id: str | None,
    name: str,
    start_time: datetime | float | None,
    public: bool | None,
    attributes: Mapping[str, AttributeValue],
) -> LangfuseObservation:
    """Create the generation for one model call, timed from when that call began.

    ``trace_id``, ``parent_observation_id`` and ``observation_id`` are the v2 ``trace(id=...)``,
    ``generation(parent_observation_id=...)`` and ``generation(id=...)`` arguments, already
    normalized by ``resolve_trace_id`` and ``resolve_observation_id``.
    """
    span: Final = _start_span(
        tracing.tracer,
        name=name,
        context=_parent_context(
            trace_id=trace_id, parent_observation_id=parent_observation_id, existing_trace=existing_trace
        ),
        start_time=start_time,
        trace_id=trace_id,
        observation_id=observation_id,
        attributes=attributes,
    )
    return LangfuseObservation(span=span, public=public)


def start_child_span(
    *,
    tracing: LangfuseTracing,
    parent: LangfuseObservation,
    name: str,
    start_time: datetime | float | None,
    attributes: Mapping[str, AttributeValue],
) -> LangfuseObservation:
    """Create an observation under the generation, keeping its own time window.

    The server folds the trace's ``public`` flag across every observation, with a missing
    attribute read as ``False``, so the child repeats the generation's value.
    """
    public_entry: Final[tuple[str, bool | None]] = (LangfuseOtelSpanAttributes.TRACE_PUBLIC, parent.public)
    span: Final = _start_span(
        tracing.tracer,
        name=name,
        context=otel_trace.set_span_in_context(parent.span),
        start_time=start_time,
        trace_id=None,
        observation_id=None,
        attributes=_present(chain((public_entry,), attributes.items())),
    )
    return LangfuseObservation(span=span, public=parent.public)


@dataclass(frozen=True, slots=True)
class TraceIdHashSampler(Sampler):
    """Sample on a SHA-256 of the trace id rather than its low 64 bits.

    litellm trace ids are UUIDs, whose variant bits pin the top of that low word, so
    ``TraceIdRatioBased`` drops every trace at rates up to 0.5 and skews above it.
    """

    rate: float

    def should_sample(
        self,
        parent_context: Context | None,
        trace_id: int,
        name: str,
        kind: SpanKind | None = None,
        attributes: Attributes = None,
        links: Sequence[Link] | None = None,
        trace_state: TraceState | None = None,
    ) -> SamplingResult:
        digest: Final = sha256(trace_id.to_bytes(16, "big")).digest()
        sampled: Final = int.from_bytes(digest[:8], "big") < round(self.rate * 2**64)
        parent: Final = otel_trace.get_current_span(parent_context).get_span_context()
        return SamplingResult(
            Decision.RECORD_AND_SAMPLE if sampled else Decision.DROP,
            attributes if sampled else None,
            parent.trace_state if parent.is_valid else None,
        )

    def get_description(self) -> str:
        return f"TraceIdHashSampler{{{self.rate}}}"


def _parse_float(raw: str) -> float | None:
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_sample_rate(raw: str) -> float | None:
    rate: Final = _parse_float(raw)
    return rate if rate is not None and 0.0 <= rate <= 1.0 else None


def configured_sample_rate() -> float:
    """``LANGFUSE_SAMPLE_RATE`` as a fraction, exporting everything when it is unset or unusable."""
    raw: Final = os.environ.get("LANGFUSE_SAMPLE_RATE")
    if raw is None:
        return 1.0
    parsed: Final = _parse_sample_rate(raw)
    if parsed is None:
        verbose_logger.warning(
            "LANGFUSE_SAMPLE_RATE=%r is not a number between 0.0 and 1.0; ignoring it and exporting every trace", raw
        )
        return 1.0
    return parsed


def configured_timeout() -> float:
    """``LANGFUSE_TIMEOUT`` in seconds for every export and REST call, the v2 SDK's 20 s when unset.

    A value that is not a number raises, as the v2 client did at construction, so a typo is not silently ignored.
    """
    return float(os.environ.get("LANGFUSE_TIMEOUT", _DEFAULT_TIMEOUT_SECONDS))


def configured_max_retries() -> int:
    """``LANGFUSE_MAX_RETRIES`` as the number of re-sends after a failed export, the v2 SDK's knob and default.

    Capped at ``_MAX_RETRIES``: with the backoff ceiling that is already hours per batch, and the exporter holds
    one delay per re-send.
    """
    raw: Final = os.environ.get("LANGFUSE_MAX_RETRIES")
    if raw is None:
        return _DEFAULT_MAX_RETRIES
    if not raw.strip().isdigit():
        verbose_logger.warning(
            "LANGFUSE_MAX_RETRIES=%r is not a whole number; retrying %d times", raw, _DEFAULT_MAX_RETRIES
        )
        return _DEFAULT_MAX_RETRIES
    requested: Final = int(raw)
    if requested > _MAX_RETRIES:
        verbose_logger.warning(
            "LANGFUSE_MAX_RETRIES=%d is above the ceiling; retrying %d times", requested, _MAX_RETRIES
        )
    return min(requested, _MAX_RETRIES)


def configured_release() -> str | None:
    """``LANGFUSE_RELEASE``, else the commit variable of the CI or deploy platform, as both SDK generations resolve it."""
    return os.environ.get("LANGFUSE_RELEASE") or next(
        (os.environ[name] for name in _COMMON_RELEASE_ENVS if name in os.environ), None
    )


def configured_prompt_cache_ttl() -> float:
    """``LANGFUSE_PROMPT_CACHE_DEFAULT_TTL_SECONDS`` in whole seconds as the SDK reads it, its 60 s default when unset
    or unusable; ``raise_if_unusable_prompt_cache_ttl`` has already named a value that is not a whole number."""
    raw: Final = os.environ.get(PROMPT_CACHE_TTL_ENV)
    if raw is None:
        return _DEFAULT_PROMPT_CACHE_TTL_SECONDS
    parsed: Final = whole_number(raw)
    if parsed is None or parsed < 0:
        verbose_logger.warning(
            "%s=%r is not a whole number of seconds at or above 0; caching prompts for %.0f s",
            PROMPT_CACHE_TTL_ENV,
            raw,
            _DEFAULT_PROMPT_CACHE_TTL_SECONDS,
        )
        return _DEFAULT_PROMPT_CACHE_TTL_SECONDS
    return float(parsed)


def configured_flush_at() -> int:
    """``LANGFUSE_FLUSH_AT`` as the export batch size, the SDK's own knob, with its default when unset or unusable."""
    raw: Final = os.environ.get("LANGFUSE_FLUSH_AT")
    if raw is None:
        return _DEFAULT_FLUSH_AT
    parsed: Final = int(raw) if raw.strip().isdigit() else None
    if parsed is None or not 0 < parsed <= _MAX_QUEUE_SIZE:
        verbose_logger.warning(
            "LANGFUSE_FLUSH_AT=%r is not a whole number between 1 and %d; exporting batches of %d",
            raw,
            _MAX_QUEUE_SIZE,
            _DEFAULT_FLUSH_AT,
        )
        return _DEFAULT_FLUSH_AT
    return parsed


class DiscardingSpanExporter(SpanExporter):
    """Accept and drop every span, for mock mode.

    The mock intercepts the httpx client behind the REST API, but observations
    travel over OTLP, so without this the "no network calls" contract silently sends
    real traces to the configured host.
    """

    def export(self, spans: object) -> SpanExportResult:
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


_ExportOutcome = Literal["delivered", "retry", "rejected", "too_large"]
_Batch = tuple[ReadableSpan, ...]


@dataclass(frozen=True, slots=True)
class _Halving:
    """One round of a 413 split: the batches still to send and the results of the ones already settled."""

    pending: tuple[_Batch, ...]
    settled: tuple[SpanExportResult, ...] = ()


def _smaller(batch: _Batch) -> tuple[_Batch, ...]:
    """What to send after a 413: the two halves of a batch, or a single span with its largest field truncated."""
    if len(batch) != 1:
        return batch[: len(batch) // 2], batch[len(batch) // 2 :]
    (only,) = batch
    truncated: Final = _truncated(only)
    return () if truncated is None else ((truncated,),)


def _in_group(key: str, group: tuple[str, ...]) -> bool:
    return any(key == prefix or key.startswith(prefix + ".") for prefix in group)


def _group_size(attributes: Mapping[str, AttributeValue], group: tuple[str, ...]) -> int:
    return sum(
        len(str(value).encode("utf-8", errors="replace"))
        for key, value in attributes.items()
        if _in_group(key, group) and value != _TRUNCATION_MARKER
    )


def _marker_key(prefix: str) -> str:
    """Langfuse reads input and output as one string but metadata only as flattened keys, so the marker gets one."""
    return f"{prefix}.truncated" if prefix in _METADATA_PREFIXES else prefix


def _truncated(span: ReadableSpan, groups: tuple[tuple[str, ...], ...] = _TRUNCATION_GROUPS) -> ReadableSpan | None:
    attributes: Final = span.attributes or MappingProxyType({})
    largest: Final = max(groups, key=lambda group: _group_size(attributes, group))
    if _group_size(attributes, largest) == 0:
        return None
    kept: Final = {key: value for key, value in attributes.items() if not _in_group(key, largest)}
    marked: Final = {
        _marker_key(prefix): _TRUNCATION_MARKER
        for prefix in largest
        if any(_in_group(key, (prefix,)) for key in attributes)
    }
    return _with_attributes(span, MappingProxyType({**kept, **marked}))


def _with_attributes(span: ReadableSpan, attributes: Mapping[str, AttributeValue]) -> ReadableSpan:
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=attributes,
        events=span.events,
        links=span.links,
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


def _limited_span(span: ReadableSpan, max_bytes: int) -> ReadableSpan:
    attributes: Final = span.attributes or MappingProxyType({})
    oversized: Final = tuple(prefix for prefix in _OUTPUT_PREFIXES if _group_size(attributes, (prefix,)) > max_bytes)
    if not oversized:
        return span
    verbose_logger.warning(
        "Langfuse output exceeds LANGFUSE_MAX_EVENT_SIZE_BYTES=%d, replacing it with %r",
        max_bytes,
        _TRUNCATION_MARKER,
    )
    return _truncated(span, (oversized,)) or span


def _media_data(value: str) -> JsonValue:
    try:
        return _MEDIA_JSON.validate_json(value)
    except ValidationError:
        return value


@dataclass(frozen=True, slots=True)
class _MediaEndpoints:
    get_upload_url: partial[GetMediaUploadUrlResponse]
    patch: partial[None]


@dataclass(frozen=True, slots=True)
class _MediaApi:
    media: _MediaEndpoints


@dataclass(frozen=True, slots=True)
class _MediaStorageClient:
    client: httpx.Client
    timeout: float

    def put(self, url: str, *, headers: Mapping[str, str], content: bytes) -> httpx.Response:
        return self.client.put(url, headers=headers, content=content, timeout=self.timeout)


class _MediaContent(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True, alias_generator=lambda key: "_" + key)

    content_bytes: bytes
    content_length: int
    content_type: MediaContentType
    content_sha256_hash: str


def _media_error_response(error: Exception) -> httpx.Response | None:
    if isinstance(error, httpx.HTTPStatusError):
        return error.response
    if isinstance(error, ApiError) and error.status_code is not None:
        return httpx.Response(error.status_code, headers=httpx.Headers(error.headers, encoding="utf-8"))
    return None


def _media_retry_giveup(error: Exception) -> bool:
    response: Final = _media_error_response(error)
    return response is not None and response.status_code < 500 and response.status_code not in (408, 409, 429)


@dataclass(frozen=True, slots=True)
class _RetryingMediaManager:
    api: _MediaApi
    storage: _MediaStorageClient
    max_attempts: int

    def upload(self, *, media: LangfuseMedia, trace_id: str, observation_id: str | None, field: str) -> None:
        content: Final = _MediaContent.model_validate(media)
        registration: Final = self._request_with_backoff(
            self.api.media.get_upload_url,
            content_length=content.content_length,
            content_type=content.content_type,
            sha256hash=content.content_sha256_hash,
            trace_id=trace_id,
            observation_id=observation_id,
            field=field,
        )
        upload_url: Final = registration.upload_url
        if upload_url is not None:
            headers: Final = {
                "Content-Type": content.content_type,
                **(
                    {}
                    if "storage.googleapis.com" in upload_url
                    else {"x-ms-blob-type": "BlockBlob", "x-amz-checksum-sha256": content.content_sha256_hash}
                ),
            }

            def upload() -> httpx.Response:
                return self.storage.put(upload_url, headers=headers, content=content.content_bytes).raise_for_status()

            started: Final = monotonic()
            response: Final = self._upload_response(upload)
            self._request_with_backoff(
                self.api.media.patch,
                media_id=registration.media_id,
                uploaded_at=datetime.now(timezone.utc),
                upload_http_status=response.status_code,
                upload_http_error=response.text,
                upload_time_ms=int((monotonic() - started) * 1000),
            )
            response.raise_for_status()
        media._media_id = registration.media_id  # pyright: ignore[reportPrivateUsage]  # rebind-ok: SDK reference ID

    def _upload_response(self, upload: Callable[[], httpx.Response]) -> httpx.Response:
        try:
            return self._request_with_backoff(upload)
        except httpx.HTTPStatusError as error:
            return error.response

    def _request_with_backoff(
        self,
        func: Callable[_MediaParams, _MediaResult],
        *args: _MediaParams.args,
        **kwargs: _MediaParams.kwargs,  # kwargs-ok: SDK retry hook forwards calls with a ParamSpec-preserved contract
    ) -> _MediaResult:
        attempts: Final = count()

        def delay(error: Exception) -> float:
            response: Final = _media_error_response(error) or httpx.Response(500)
            return langfuse_http_client._retry_timeout(  # pyright: ignore[reportPrivateUsage]  # SDK owns wait policy
                response=response, retries=next(attempts)
            )

        retry: Final = backoff.on_exception(
            backoff.runtime,
            Exception,
            max_tries=self.max_attempts,
            giveup=_media_retry_giveup,
            value=delay,
            jitter=None,
            logger=None,
        )
        return retry(partial(func, *args, **kwargs))()


def _media_candidate(data: JsonValue) -> tuple[tuple[str, ...], str] | None:
    candidate: Final = base64_media_data_uri(data)
    if candidate is None or candidate[1].startswith("@@@langfuseMedia:", candidate[1].find(",") + 1):
        return None
    return candidate


def _contains_media(data: JsonValue, depth: int = 0) -> bool:
    if depth > 20:
        return False
    if _media_candidate(data) is not None:
        return True
    if isinstance(data, list):
        return any(_contains_media(item, depth + 1) for item in data)
    if isinstance(data, dict):
        return any(_contains_media(item, depth + 1) for item in data.values())
    return False


def contains_langfuse_media(value: object) -> bool:
    serialized: Final = _serialize(value)
    return serialized is not None and _contains_media(_media_data(serialized))


def _with_media_reference(data: JsonValue, path: tuple[str, ...], reference: str) -> JsonValue:
    if not path:
        return reference
    if not isinstance(data, dict):
        return data
    return {
        key: _with_media_reference(value, path[1:], reference) if key == path[0] else value
        for key, value in data.items()
    }


@dataclass(frozen=True, slots=True)
class _MediaUploads:
    manager: _RetryingMediaManager
    enabled: bool = field(
        default_factory=lambda: os.environ.get("LANGFUSE_MEDIA_UPLOAD_ENABLED", "True").lower() not in ("false", "0")
    )

    def has_media(self, span: ReadableSpan) -> bool:
        return self.enabled and any(
            isinstance(value, str) and _contains_media(_media_data(value))
            for key, value in (span.attributes or {}).items()
            if any(_in_group(key, (prefix,)) for prefix, _ in _MEDIA_FIELDS)
        )

    def process(self, span: ReadableSpan) -> ReadableSpan:
        context: Final = span.context
        if context is None or not context.is_valid:
            return span
        attributes: Final = MappingProxyType(
            {key: self._attribute(key, value, context) for key, value in (span.attributes or {}).items()}
        )
        return _with_attributes(span, attributes)

    def _attribute(self, key: str, value: AttributeValue, context: SpanContext) -> AttributeValue:
        field_name: Final = next((field for prefix, field in _MEDIA_FIELDS if _in_group(key, (prefix,))), None)
        if field_name is None or not isinstance(value, str):
            return value
        data: Final = _media_data(value)
        processed: Final = self._value(data, context, field_name, key)
        return value if processed == data else dumps(processed)

    def _value(self, data: JsonValue, context: SpanContext, field_name: str, key: str, depth: int = 0) -> JsonValue:
        if depth > 20:
            return data
        candidate: Final = _media_candidate(data)
        if candidate is None:
            if isinstance(data, list):
                return [self._value(item, context, field_name, key, depth + 1) for item in data]
            if isinstance(data, dict):
                return {name: self._value(item, context, field_name, key, depth + 1) for name, item in data.items()}
            return data
        try:
            media: Final = LangfuseMedia(base64_data_uri=candidate[1])
            self.manager.upload(
                media=media,
                trace_id=format(context.trace_id, "032x"),
                observation_id=format(context.span_id, "016x") if key.startswith("langfuse.observation.") else None,
                field=field_name,
            )
            reference: Final = _MEDIA_JSON.validate_json(dumps(media, cls=EventSerializer))
            return _with_media_reference(data, candidate[0], reference) if isinstance(reference, str) else data
        except Exception as error:  # noqa: BLE001  # SDK conversion and serialization can raise arbitrary exceptions
            verbose_logger.warning(
                "Langfuse media conversion failed for an item in %s (%s), keeping the item", key, type(error).__name__
            )
            return data


class _MediaSpanQueue:
    def __init__(
        self,
        media: _MediaUploads,
        send: Callable[[Sequence[ReadableSpan]], SpanExportResult],
        close: Callable[[], None],
    ) -> None:
        self.media: Final = media
        self.send: Final = send
        self.close: Final = close
        self.closed = False
        self._reset()
        if hasattr(os, "register_at_fork"):
            queue_ref: Final = ref(self)

            def reinitialize() -> None:
                if (queue := queue_ref()) is not None:
                    queue._reset()

            os.register_at_fork(after_in_child=reinitialize)

    def _reset(self) -> None:
        self.queue: Queue[ReadableSpan | None] = Queue(_MAX_QUEUE_SIZE)
        self.condition: threading.Condition = threading.Condition()
        self.worker: threading.Thread = threading.Thread(target=self._run, name="langfuse-media", daemon=True)
        self.submitted = 0
        self.completed = 0
        self.failed = False

    def submit(self, span: ReadableSpan) -> bool:
        with self.condition:
            if self.closed:
                return False
            try:
                self.queue.put_nowait(span)
            except Full:
                verbose_logger.error("Langfuse media span queue is full, exporting the span without media conversion")
                return False
            self.submitted += 1
            if self.submitted == 1:
                self.worker.start()
            return True

    def flush(self, timeout_millis: int) -> bool:
        with self.condition:
            target: Final = self.submitted
            if not self.condition.wait_for(lambda: self.completed >= target, timeout_millis / 1000):
                return False
            delivered: Final = not self.failed
            self.failed = False
            return delivered

    def shutdown(self) -> None:
        with self.condition:
            if self.closed:
                return
            self.closed = True
            if self.submitted == 0:
                self.close()
                return
            try:
                self.queue.put_nowait(None)
            except Full:
                pass
        self.worker.join(30)
        if self.worker.is_alive():
            verbose_logger.warning("Langfuse media is still uploading after the shutdown deadline")

    def _run(self) -> None:
        try:
            while self._consume():
                pass
        finally:
            self.close()

    def _consume(self) -> bool:
        try:
            span: Final = self.queue.get(timeout=1)
        except Empty:
            return not self.closed
        if span is None:
            self.queue.task_done()
            return False
        delivered: Final = self._deliver(span)
        with self.condition:
            self.failed = self.failed or not delivered
            self.completed += 1
            self.condition.notify_all()
        self.queue.task_done()
        return True

    def _deliver(self, span: ReadableSpan) -> bool:
        token: Final = otel_context.attach(otel_context.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        try:
            return self.send((self.media.process(span),)) is SpanExportResult.SUCCESS
        except Exception:  # noqa: BLE001  # a failed SDK upload must not stop the independent worker
            verbose_logger.exception("Langfuse media span export failed")
            return False
        finally:
            otel_context.detach(token)


def enable_langfuse_debug_logging() -> None:
    """What ``Langfuse(debug=True)`` does: a root handler if none exists, and the ``langfuse`` logger at DEBUG."""
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    _langfuse_logger.setLevel(logging.DEBUG)


def _retryable_status(status: int) -> bool:
    """Any 5xx, a timeout or a rate limit: what the v2 consumer re-sent, plus the 408 the OTLP exporter retries."""
    return status in (408, 429) or 500 <= status <= 599


@dataclass(frozen=True, slots=True)
class LangfuseSpanExporter(SpanExporter):
    """OTLP/HTTP protobuf export through litellm's own HTTP handler.

    The handler carries litellm's TLS material (``ssl_verify``, CA bundle, client certificate) exactly
    as v2's injected httpx client did. A connect or read failure and a retryable status are re-sent after
    each delay, matching the v2 ingestion consumer; ``BatchSpanProcessor`` would otherwise drop the whole
    batch on the first exception. A 413 splits the batch in halves until each body fits or a single span
    is left; that span is re-sent with its output and metadata replaced by the truncation marker,
    largest first, and dropped only when the fully truncated span is still refused. Input is preserved.
    """

    handler: HTTPHandler
    endpoint: str
    headers: Mapping[str, str]
    timeout: float
    owns_handler: bool = False
    delays: Sequence[float] = (1.0, 2.0, 4.0)
    max_event_size_bytes: int = field(
        default_factory=lambda: int(os.environ.get("LANGFUSE_MAX_EVENT_SIZE_BYTES", _DEFAULT_MAX_EVENT_SIZE_BYTES))
    )
    media: _MediaUploads | None = None
    media_queue: _MediaSpanQueue | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.media is not None:
            object.__setattr__(
                self,
                "media_queue",
                _MediaSpanQueue(
                    self.media, self._export_spans, self.handler.close if self.owns_handler else lambda: None
                ),
            )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        immediate: Final = tuple(
            span
            for span in spans
            if self.media is None
            or self.media_queue is None
            or not self.media.has_media(span)
            or not self.media_queue.submit(span)
        )
        return self._export_spans(immediate) if immediate else SpanExportResult.SUCCESS

    def _export_spans(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Halving a batch of n spans settles every span within ``n.bit_length()`` rounds plus one per truncation
        step, so the rounds are a fixed fold rather than a recursion."""
        rounds: Final = range(len(spans).bit_length() + 1 + len(_TRUNCATION_GROUPS))
        limited: Final = tuple(_limited_span(span, self.max_event_size_bytes) for span in spans)
        final: Final = reduce(lambda halving, _: self._round(halving), rounds, _Halving(pending=(limited,)))
        return (
            SpanExportResult.SUCCESS
            if all(result is SpanExportResult.SUCCESS for result in final.settled)
            else SpanExportResult.FAILURE
        )

    def _round(self, halving: _Halving) -> _Halving:
        sent: Final = tuple((batch, self._send_batch(batch)) for batch in halving.pending)
        return _Halving(
            pending=tuple(chain.from_iterable(_smaller(batch) for batch, outcome in sent if outcome == "too_large")),
            settled=halving.settled
            + tuple(
                SpanExportResult.SUCCESS if outcome == "delivered" else SpanExportResult.FAILURE
                for _, outcome in sent
                if outcome != "too_large"
            ),
        )

    def _send_batch(self, batch: _Batch) -> _ExportOutcome:
        """A 413 on more than one span asks for halves; on a single span it asks for a truncation, and the span is
        dropped and reported once nothing is left to truncate."""
        body: Final = _encode(batch)
        if body is None:
            return "rejected"
        outcome: Final = self._send(body)
        if outcome != "too_large":
            return outcome
        match batch:
            case (only,) if _truncated(only) is None:
                verbose_logger.error(
                    "Langfuse rejected a single %d byte span export to %s as too large, dropping it",
                    len(body),
                    self.endpoint,
                )
                return "rejected"
            case (_,):
                verbose_logger.warning(
                    "Langfuse rejected a single %d byte span export to %s as too large, resending it with its "
                    "largest field replaced by %r",
                    len(body),
                    self.endpoint,
                    _TRUNCATION_MARKER,
                )
            case _:
                verbose_logger.warning(
                    "Langfuse rejected a %d byte export of %d spans as too large, resending in halves",
                    len(body),
                    len(batch),
                )
        return "too_large"

    def _send(self, body: bytes) -> _ExportOutcome:
        for delay in self.delays:
            outcome: _ExportOutcome = self._post(body)
            if outcome != "retry":
                return outcome
            verbose_logger.warning("Langfuse export to %s failed, retrying in %ss", self.endpoint, delay)
            sleep(delay)
        last: Final = self._post(body)
        if last == "retry":
            verbose_logger.error("Langfuse export to %s failed after %d retries", self.endpoint, len(self.delays))
        return last

    def _post(self, body: bytes) -> _ExportOutcome:
        try:
            self.handler.post(self.endpoint, data=body, headers=dict(self.headers), timeout=self.timeout)
        except httpx.HTTPStatusError as error:
            status: Final = error.response.status_code
            if _retryable_status(status):
                return "retry"
            if status == 413:
                return "too_large"
            verbose_logger.error(
                "Langfuse rejected an export to %s with HTTP %d%s",
                self.endpoint,
                status,
                _SERVER_FLOOR_HINT if status == 404 else "",
            )
            return "rejected"
        except (httpx.TransportError, litellm.Timeout) as error:
            verbose_logger.warning("Langfuse export to %s raised %s", self.endpoint, error)
            return "retry"
        _langfuse_logger.debug("Exported %d bytes of spans to %s", len(body), self.endpoint)
        return "delivered"

    def shutdown(self) -> None:
        if self.media_queue is not None:
            self.media_queue.shutdown()
        elif self.owns_handler:
            self.handler.close()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self.media_queue is None or self.media_queue.flush(timeout_millis)


def _encode(spans: Sequence[ReadableSpan]) -> bytes | None:
    """The OTLP body, or ``None`` when nothing survived: a span the encoder rejects is dropped, not the whole batch."""
    try:
        return encode_spans(spans).SerializeToString()
    except Exception:  # noqa: BLE001  # protobuf raises TypeError or ValueError depending on the field
        kept: Final = tuple(span for span in spans if _encodes(span))
        verbose_logger.error("Langfuse export dropped %d span(s) the OTLP encoder rejected", len(spans) - len(kept))
        return encode_spans(kept).SerializeToString() if kept else None


def _encodes(span: ReadableSpan) -> bool:
    try:
        encode_spans((span,))
    except Exception:  # noqa: BLE001  # same encoder failure modes as above
        return False
    return True


def _build_span_exporter(
    *, public_key: str, secret_key: str, base_url: str, handler: HTTPHandler | None = None
) -> LangfuseSpanExporter:
    """Endpoint, headers and export path are the v4 SDK span processor's, so the server treats the spans as SDK
    traffic; the 20 s timeout and the retry count are what the v2 consumer used. The ingestion-version header is
    the one Langfuse's compatibility matrix asks a v4 producer to send."""
    export_path: Final = os.getenv("LANGFUSE_OTEL_TRACES_EXPORT_PATH") or "/api/public/otel/v1/traces"
    encoded_auth: Final = b64encode(f"{public_key}:{secret_key}".encode()).decode("ascii")
    http_handler: Final = HTTPHandler(timeout=configured_timeout()) if handler is None else handler
    media_api: Final = LangfuseAPI(
        base_url=base_url,
        username=public_key,
        password=secret_key,
        httpx_client=http_handler.client,
        timeout=configured_timeout(),
    )
    media_requests: Final = _MediaApi(
        media=_MediaEndpoints(
            get_upload_url=partial(media_api.media.get_upload_url, request_options=_NO_REST_RETRIES),
            patch=partial(media_api.media.patch, request_options=_NO_REST_RETRIES),
        )
    )
    media_storage: Final = _MediaStorageClient(client=http_handler.client, timeout=configured_timeout())
    return LangfuseSpanExporter(
        handler=http_handler,
        endpoint=f"{base_url.rstrip('/')}/{export_path.lstrip('/')}",
        headers=MappingProxyType(
            {
                "Authorization": "Basic " + encoded_auth,
                "Content-Type": "application/x-protobuf",
                "x-langfuse-sdk-name": "python",
                "x-langfuse-sdk-version": version("langfuse"),
                "x-langfuse-public-key": public_key,
                _LANGFUSE_INGESTION_VERSION_HEADER: _LANGFUSE_INGESTION_VERSION,
            }
        ),
        timeout=configured_timeout(),
        owns_handler=handler is None,
        delays=tuple(2.0 ** min(attempt, _MAX_BACKOFF_EXPONENT) for attempt in range(configured_max_retries())),
        media=_MediaUploads(
            manager=_RetryingMediaManager(
                api=media_requests,
                storage=media_storage,
                max_attempts=configured_max_retries() + 1,
            ),
        ),
    )


def _resource(*, environment: str | None, release: str | None) -> Resource:
    """Only litellm's own attributes: ``Resource.create`` would merge the host's ``OTEL_RESOURCE_ATTRIBUTES``."""
    return Resource(
        _present(
            (
                (LangfuseOtelSpanAttributes.ENVIRONMENT, environment),
                (LangfuseOtelSpanAttributes.RELEASE, release),
            )
        )
    )


class _ExportLedger(SpanExporter):
    """Counts the batches the exporter gave up on, so a flush can report delivery rather than a drained queue."""

    def __init__(self, exporter: SpanExporter) -> None:
        self.exporter: Final = exporter
        self.failed_batches = 0

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        result: Final = self.exporter.export(spans)
        if result is not SpanExportResult.SUCCESS:
            self.failed_batches += 1
        return result

    def shutdown(self) -> None:
        self.exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self.exporter.force_flush(timeout_millis) is not False


@dataclass(frozen=True, slots=True)
class LangfuseTracing:
    """litellm's own export channel to one Langfuse project: a provider, its tracer and the exporter behind them.

    The channel is litellm's rather than the SDK's so that the process-global OTel provider stays
    untouched, historical timestamps and caller ids are honoured, and no SDK internals are needed.
    """

    provider: TracerProvider
    tracer: Tracer
    ledger: _ExportLedger

    def flush(self, timeout_millis: int = 30_000) -> bool:
        """``True`` only when the queue drained in time and every batch it held was accepted by the destination."""
        failed_before: Final = self.ledger.failed_batches
        deadline: Final = monotonic() + timeout_millis / 1000
        return (
            self.provider.force_flush(timeout_millis)
            and self.ledger.force_flush(max(0, int((deadline - monotonic()) * 1000)))
            and self.ledger.failed_batches == failed_before
        )

    def shutdown(self) -> None:
        self.provider.shutdown()


@dataclass(frozen=True, slots=True)
class _TracingKey:
    public_key: str
    secret_key: str
    base_url: str
    environment: str | None
    release: str | None
    sample_rate: float
    flush_at: int
    flush_interval_millis: int
    mock_mode: bool


@dataclass(frozen=True, slots=True)
class _Lease:
    tracing: LangfuseTracing
    holders: int
    retire: threading.Timer | None = None


_TRACING_LOCK: Final = threading.Lock()
_TRACING: Final[dict[_TracingKey, _Lease]] = {}  # mutable-ok: process-wide channel cache, guarded by _TRACING_LOCK


def acquire_langfuse_tracing(
    *,
    public_key: str,
    secret_key: str,
    base_url: str,
    environment: str | None,
    release: str | None,
    flush_interval: float,
    mock_mode: bool,
) -> LangfuseTracing:
    """One export channel per credential set, shared by every logger built for it.

    A provider owns a batch export thread, so a channel lives while any logger holds it and is
    retired through ``release_langfuse_tracing`` once the last holder lets go.
    """
    if parse_langfuse_debug(os.getenv("LANGFUSE_DEBUG")):
        enable_langfuse_debug_logging()
    key: Final = _TracingKey(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        environment=environment,
        release=release,
        sample_rate=configured_sample_rate(),
        flush_at=configured_flush_at(),
        flush_interval_millis=int(flush_interval * 1000),
        mock_mode=mock_mode,
    )
    with _TRACING_LOCK:
        cached: Final = _TRACING.get(key)
        if cached is not None:
            if cached.retire is not None:
                cached.retire.cancel()
            _TRACING[key] = replace(cached, holders=cached.holders + 1, retire=None)
            return cached.tracing
        created: Final = build_langfuse_tracing(
            exporter=DiscardingSpanExporter()
            if mock_mode
            else _build_span_exporter(public_key=public_key, secret_key=secret_key, base_url=base_url),
            environment=environment,
            release=release,
            sample_rate=key.sample_rate,
            flush_at=key.flush_at,
            flush_interval_millis=key.flush_interval_millis,
        )
        _TRACING[key] = _Lease(tracing=created, holders=1)
        return created


def release_langfuse_tracing(tracing: LangfuseTracing, *, grace_seconds: float = _CHANNEL_RETIRE_GRACE_SECONDS) -> None:
    """Let go of one logger's hold on its channel; a channel nobody holds is retired ``grace_seconds`` later.

    The grace covers a callback that fetched its logger from the cache just before the entry expired,
    and a logger rebuilt for the same credentials in the meantime picks the channel back up instead.
    """
    with _TRACING_LOCK:
        held: Final = next(((key, lease) for key, lease in _TRACING.items() if lease.tracing is tracing), None)
        if held is None:
            return
        key, lease = held
        if lease.holders <= 0:
            return
        if lease.holders > 1:
            _TRACING[key] = replace(lease, holders=lease.holders - 1)
            return
        if grace_seconds > 0:
            retire: Final = threading.Timer(grace_seconds, lambda: _retire_unless_reacquired(key, retire))
            retire.name = "langfuse-retire"
            retire.daemon = True
            _TRACING[key] = _Lease(tracing=tracing, holders=0, retire=retire)
            retire.start()
            return
        del _TRACING[key]
    tracing.shutdown()


def _retire_unless_reacquired(key: _TracingKey, timer: threading.Timer) -> None:
    """Only the timer the lease still points at may retire it; a re-acquire cancels and clears the pending one."""
    with _TRACING_LOCK:
        lease: Final = _TRACING.get(key)
        if lease is None or lease.retire is not timer:
            return
        del _TRACING[key]
    lease.tracing.shutdown()


class _FlushWorker(threading.Thread):
    """Daemon, so a channel still blocked at the deadline cannot hold up interpreter exit."""

    def __init__(self, channel: LangfuseTracing, timeout_millis: int) -> None:
        super().__init__(name="langfuse-flush", daemon=True)
        self.channel: Final = channel
        self.timeout_millis: Final = timeout_millis
        self.flushed = False

    def run(self) -> None:
        self.flushed = self.channel.flush(self.timeout_millis)


def flush_langfuse_tracing(timeout_millis: int = 30_000) -> bool:
    """Force-flush every export channel this process acquired, all within one ``timeout_millis`` deadline.

    ``True`` only when every channel flushed in time; a channel still blocked at the deadline is left to
    finish in the background rather than pushing the deadline out for the channels after it.
    """
    with _TRACING_LOCK:
        channels: Final = tuple(lease.tracing for lease in _TRACING.values())
    workers: Final = tuple(_FlushWorker(channel, timeout_millis) for channel in channels)
    deadline: Final = monotonic() + timeout_millis / 1000
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(max(0.0, deadline - monotonic()))
    return all(not worker.is_alive() and worker.flushed for worker in workers)


def build_langfuse_tracing(
    *,
    exporter: SpanExporter,
    environment: str | None,
    release: str | None,
    sample_rate: float,
    flush_interval_millis: int,
    flush_at: int = _DEFAULT_FLUSH_AT,
) -> LangfuseTracing:
    """Wire the provider from litellm's own settings so a host's ``OTEL_*`` variables do not steer it.

    An unset sampler or span limit falls back to ``OTEL_TRACES_SAMPLER`` and
    ``OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT`` style variables, which are meant for the
    host application's own tracing. ``OTEL_SDK_DISABLED`` still applies, as it does to the SDK.

    The tracer carries the SDK's scope name because Langfuse keys on it: spans from any other
    scope are treated as foreign OTel traffic and get their raw attributes echoed into metadata.
    """
    if os.environ.get("OTEL_SDK_DISABLED", "").strip().lower() == "true":
        verbose_logger.warning("OTEL_SDK_DISABLED=true also disables the langfuse callback's export channel")
    provider: Final = TracerProvider(
        resource=_resource(environment=environment, release=release),
        sampler=ALWAYS_ON if sample_rate >= 1 else TraceIdHashSampler(sample_rate),
        id_generator=_RequestedIdGenerator(),
        span_limits=_SPAN_LIMITS,
    )
    ledger: Final = _ExportLedger(exporter)
    provider.add_span_processor(
        BatchSpanProcessor(
            ledger,
            max_queue_size=_MAX_QUEUE_SIZE,
            max_export_batch_size=flush_at,
            schedule_delay_millis=flush_interval_millis,
        )
    )
    return LangfuseTracing(provider=provider, tracer=provider.get_tracer(_TRACER_NAME), ledger=ledger)


@dataclass(frozen=True, slots=True)
class _CachedPrompt:
    prompt: PromptClient
    fetched_at: float


_PromptKey = tuple[str, int | None, str | None]


def _prompt_client(prompt: Prompt) -> PromptClient:
    return ChatPromptClient(prompt) if isinstance(prompt, Prompt_Chat) else TextPromptClient(prompt)


@dataclass(frozen=True, slots=True)
class AuthCheckFailure:
    reason: str


def _auth_check_failure(reason: str) -> AuthCheckFailure:
    verbose_logger.warning("Langfuse auth check failed: %s", reason)
    return AuthCheckFailure(reason)


class _ApiErrorDetail(LiteLLMBaseModel):
    """The status and body of an ``ApiError``, whose own ``str`` also dumps every response header."""

    model_config = ConfigDict(frozen=True, from_attributes=True)
    status_code: int | None
    body: object


def _api_error_reason(error: ApiError) -> str:
    detail: Final = _ApiErrorDetail.model_validate(error)
    return f"status_code: {detail.status_code}, body: {detail.body}"


class LangfusePromptError(Exception):
    """An ``ApiError`` without its ``headers``, which the proxy would otherwise forward to its own client."""

    def __init__(self, error: ApiError) -> None:
        detail: Final = _ApiErrorDetail.model_validate(error)
        super().__init__(f"status_code: {detail.status_code}, body: {detail.body}")
        self.status_code: Final = detail.status_code
        self.body: Final = detail.body


def _is_server_error(error: ApiError) -> bool:
    return error.status_code is not None and error.status_code >= 500


class LangfuseApiClient:
    """litellm's handle on one Langfuse project over its REST API: prompts, ``auth_check`` and the project id.

    The SDK's ``Langfuse`` client is deliberately not constructed. It keeps one tracing bundle per
    public key and hands it to every ``Langfuse()`` a host application builds for the same key, so
    litellm's exporter, host and masking would leak into that application. Observations travel
    over ``LangfuseTracing``; nothing here exports spans.

    Prompts are cached for ``LANGFUSE_PROMPT_CACHE_DEFAULT_TTL_SECONDS`` (60 by default) as the SDK
    does. A stale prompt is served at once and refreshed on a background thread, so the request
    that finds it stale, and the event loop it runs on, never wait for the REST round trip; a
    refresh that fails keeps serving the stale prompt rather than failing the request, again like the SDK.
    """

    def __init__(self, api: LangfuseAPI, *, prompt_cache_ttl_seconds: float) -> None:
        self.api: Final = api
        self.prompt_cache_ttl_seconds: Final = prompt_cache_ttl_seconds
        # mutable-ok: per-client prompt cache, guarded by _lock
        self._prompts: Final[dict[_PromptKey, _CachedPrompt]] = {}
        # mutable-ok: keys with a refresh in flight, guarded by _lock
        self._refreshing: Final[set[_PromptKey]] = set()
        self._lock: Final = threading.Lock()

    def auth_check(self) -> AuthCheckFailure | None:
        """``None`` when the keys reach a project; otherwise the reason, which is also logged.

        Mirrors the SDK's ``Langfuse.auth_check``: a 200 with no project is a failure too, and a server
        error or a transport failure is reported as itself rather than as bad credentials.
        """
        try:
            projects: Final = self.api.projects.get(request_options=_NO_REST_RETRIES).data
        except ApiError as error:
            return _auth_check_failure(_api_error_reason(error))
        except Exception as error:  # noqa: BLE001  # httpx transport errors or a body the response model rejects
            return _auth_check_failure(str(error) or type(error).__name__)
        if not projects:
            return _auth_check_failure("no project found for the keys provided")
        return None

    def project_id(self) -> str | None:
        projects: Final = self.api.projects.get(request_options=_NO_REST_RETRIES).data
        return projects[0].id if projects else None

    def get_prompt(self, name: str, *, label: str | None = None, version: int | None = None) -> PromptClient:
        key: Final[_PromptKey] = (name, version, label)
        with self._lock:
            cached: Final = self._prompts.get(key)
        if cached is None:
            return self._fetch(key)
        if monotonic() - cached.fetched_at >= self.prompt_cache_ttl_seconds:
            self._refresh_in_background(key)
        return cached.prompt

    def _fetch(self, key: _PromptKey) -> PromptClient:
        fetched: Final = _prompt_client(self._request_prompt(key))
        with self._lock:
            self._prompts[key] = _CachedPrompt(prompt=fetched, fetched_at=monotonic())
        return fetched

    def _request_prompt(self, key: _PromptKey) -> Prompt:
        """Retried once, at once, after a 5xx or a transport failure: a cold miss runs on the caller's event
        loop, so the generated client's sleeping retries stay off."""
        name, version, label = key
        request: Final = partial(
            self.api.prompts.get, quote(name, safe=""), version=version, label=label, request_options=_NO_REST_RETRIES
        )
        try:
            return request()
        except ApiError as error:
            if not _is_server_error(error):
                raise LangfusePromptError(error) from None
            verbose_logger.debug("Langfuse prompt %r fetch failed (%s), retrying once", name, _api_error_reason(error))
        except httpx.TransportError as error:
            verbose_logger.debug("Langfuse prompt %r fetch failed (%s), retrying once", name, error)
        try:
            return request()
        except ApiError as error:
            raise LangfusePromptError(error) from None

    def _refresh_in_background(self, key: _PromptKey) -> None:
        with self._lock:
            if key in self._refreshing:
                return
            self._refreshing.add(key)
        threading.Thread(target=self._refresh, args=(key,), name="langfuse-prompt-refresh", daemon=True).start()

    def _refresh(self, key: _PromptKey) -> None:
        try:
            self._fetch(key)
        except Exception as error:  # noqa: BLE001  # a failed refresh keeps the stale prompt in service
            verbose_logger.warning("Langfuse prompt %r refresh failed, serving the cached version: %s", key[0], error)
        finally:
            with self._lock:
                self._refreshing.discard(key)


def build_langfuse_client(
    *,
    public_key: str | None,
    secret_key: str | None,
    base_url: str,
    httpx_client: httpx.Client | None,
) -> LangfuseApiClient:
    """The REST client for prompt management, ``auth_check`` and the Slack project link.

    Missing keys are passed through as absent credentials: the server answers 401, which
    ``auth_check`` reports as a failure rather than raising at construction.
    """
    return LangfuseApiClient(
        LangfuseAPI(
            base_url=base_url,
            username=public_key,
            password=secret_key,
            x_langfuse_sdk_name="python",
            x_langfuse_sdk_version=version("langfuse"),
            x_langfuse_public_key=public_key,
            httpx_client=httpx_client,
            timeout=configured_timeout(),
        ),
        prompt_cache_ttl_seconds=configured_prompt_cache_ttl(),
    )
