"""
DeepSeek Anthropic-compatible messages transformation config.
"""

from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.reasoning_effort_utils import warn_non_string_reasoning_effort
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.anthropic.chat.transformation import REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT
from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.llms.anthropic.messages.native.transformation import (
    AnthropicMessagesConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams

_TOOL_LIST: Final = TypeAdapter(list[object])
_TOOL_FIELDS: Final = TypeAdapter(dict[object, object])
_REQUEST_PARAMS: Final = TypeAdapter(dict[str, object])


def _without_output_config_effort(params: Mapping[str, object]) -> Mapping[str, object]:
    output_config: Final = as_str_mapping(params.get("output_config"))
    if output_config is None or "effort" not in output_config:
        return params
    residual: Final = {key: value for key, value in output_config.items() if key != "effort"}
    without_output_config: Final = {key: value for key, value in params.items() if key != "output_config"}
    return {**without_output_config, "output_config": residual} if residual else without_output_config


def _with_default_output_config_effort(params: Mapping[str, object], effort: str) -> Mapping[str, object]:
    existing: Final = as_str_mapping(params.get("output_config"))
    merged_output_config: Final = {"effort": effort, **(existing or {})}
    return {**params, "output_config": merged_output_config}


def _normalize_deepseek_reasoning_effort(
    params: Mapping[str, object],
    model: str,
    *,
    report_reasoning_drops: bool,
) -> Mapping[str, object]:
    if "reasoning_effort" not in params:
        return params
    reasoning_effort: Final = params["reasoning_effort"]
    without_effort: Final = {key: value for key, value in params.items() if key != "reasoning_effort"}
    if not isinstance(reasoning_effort, str):
        if report_reasoning_drops:
            warn_non_string_reasoning_effort(reasoning_effort, model)
        return without_effort
    if reasoning_effort == "none":
        return {**_without_output_config_effort(without_effort), "thinking": {"type": "disabled"}}
    mapped_effort: Final = REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT.get(reasoning_effort)
    if mapped_effort is None:
        raise AnthropicError(
            message=(
                f"Invalid reasoning_effort: {reasoning_effort!r}. "
                f"Must be one of: 'minimal', 'low', 'medium', 'high', "
                f"'xhigh', 'max', 'none'"
            ),
            status_code=400,
        )
    thinking: Final = without_effort.get("thinking")
    thinking_dict: Final = as_str_mapping(thinking)
    if thinking_dict is not None and thinking_dict.get("type") == "disabled":
        return without_effort
    with_thinking: Final = (
        without_effort if thinking is not None else {**without_effort, "thinking": {"type": "enabled"}}
    )
    return _with_default_output_config_effort(with_thinking, mapped_effort)


def _custom_tool_without_type(tool: object) -> Mapping[object, object] | None:
    if not isinstance(tool, dict):
        return None
    fields: Final = _TOOL_FIELDS.validate_python(tool)
    if fields.get("type") == "custom":
        fields.pop("type", None)
        return fields
    return None


def _sanitized_tool(tool: object) -> object:
    stripped: Final = _custom_tool_without_type(tool)
    return tool if stripped is None else stripped


class DeepSeekAnthropicMessagesConfig(AnthropicMessagesConfig):
    """
    DeepSeek exposes an Anthropic-compatible Messages API at
    https://api.deepseek.com/anthropic.

    It accepts the native Anthropic Messages conversation shape, including
    thinking blocks in assistant history, but rejects Anthropic's explicit
    custom-tool discriminator (`{"type": "custom"}`).
    """

    @property
    def custom_llm_provider(self) -> str | None:
        return "deepseek"

    def should_strip_billing_metadata(self) -> bool:
        return True

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or get_secret_str("DEEPSEEK_API_KEY") or litellm.api_key

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str:
        return (
            api_base
            or get_secret_str("DEEPSEEK_ANTHROPIC_API_BASE")
            or get_secret_str("DEEPSEEK_API_BASE")
            or "https://api.deepseek.com/anthropic"
        )

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: Sequence[object],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict, str | None]:
        dynamic_api_key: Final = self.get_api_key(api_key=api_key)

        if "x-api-key" not in headers and "authorization" not in headers and dynamic_api_key is not None:
            headers["x-api-key"] = dynamic_api_key

        if "anthropic-version" not in headers:
            headers["anthropic-version"] = "2023-06-01"
        if "content-type" not in headers:
            headers["content-type"] = "application/json"

        headers = self._update_headers_with_anthropic_beta(
            headers=headers,
            optional_params=optional_params,
            custom_llm_provider=self.custom_llm_provider or "deepseek",
            messages=messages,
        )

        return headers, api_base

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        base_url = self.get_api_base(api_base=api_base).rstrip("/")

        if base_url.endswith("/v1/messages") and "/anthropic/" in base_url:
            return base_url
        base_url = base_url.removesuffix("/v1/messages")
        base_url = base_url.removesuffix("/v1")
        base_url = base_url.removesuffix("/beta")

        if not base_url.endswith("/anthropic") and "/anthropic/" not in base_url:
            base_url = join_api_base(base_url, "anthropic")

        return join_api_base(base_url, "v1/messages")

    @staticmethod
    def _translate_reasoning_effort_to_anthropic(
        model: str,
        optional_params: Mapping[str, object],
        max_tokens: int | None,
        custom_llm_provider: str,
    ) -> None:
        return None

    @staticmethod
    def _translate_adaptive_effort_for_non_adaptive_model(
        model: str,
        optional_params: Mapping[str, object],
        max_tokens: int | None,
        custom_llm_provider: str,
    ) -> None:
        return None

    @staticmethod
    def _drop_incompatible_temperature_for_thinking(
        model: str,
        optional_params: Mapping[str, object],
        custom_llm_provider: str,
    ) -> None:
        return None

    @staticmethod
    def _sanitize_tools_for_deepseek(tools: object) -> object:
        if not isinstance(tools, list):
            return tools

        return [_sanitized_tool(tool) for tool in _TOOL_LIST.validate_python(tools)]

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: list[dict],
        anthropic_messages_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        anthropic_messages_request: Final = super().transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=(
                dict(
                    _normalize_deepseek_reasoning_effort(
                        _REQUEST_PARAMS.validate_python(anthropic_messages_optional_request_params),
                        model=model,
                        report_reasoning_drops=self._report_reasoning_drops,
                    )
                )
                if "reasoning_effort" in anthropic_messages_optional_request_params
                else anthropic_messages_optional_request_params
            ),
            litellm_params=litellm_params,
            headers=headers,
        )
        if "tools" in anthropic_messages_request:
            anthropic_messages_request["tools"] = self._sanitize_tools_for_deepseek(anthropic_messages_request["tools"])
        return anthropic_messages_request
