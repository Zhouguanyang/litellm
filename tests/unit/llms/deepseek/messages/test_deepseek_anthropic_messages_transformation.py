import pytest

import litellm
from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.llms.anthropic.messages.native.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.deepseek.messages.transformation import (
    DeepSeekAnthropicMessagesConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager


def test_deepseek_provider_uses_anthropic_messages_config():
    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="deepseek-v4-pro",
        provider=litellm.LlmProviders.DEEPSEEK,
    )

    assert isinstance(config, DeepSeekAnthropicMessagesConfig)
    assert config.custom_llm_provider == "deepseek"


def test_deepseek_anthropic_messages_config_defaults():
    config = DeepSeekAnthropicMessagesConfig()

    assert config.custom_llm_provider == "deepseek"
    assert config.get_api_base() == "https://api.deepseek.com/anthropic"


def test_anthropic_provider_keeps_default_config_for_deepseek_named_model():
    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="deepseek-v4-pro",
        provider=litellm.LlmProviders.ANTHROPIC,
    )

    assert isinstance(config, AnthropicMessagesConfig)
    assert not isinstance(config, DeepSeekAnthropicMessagesConfig)


def test_deepseek_anthropic_messages_url_defaults_to_anthropic_endpoint():
    config = DeepSeekAnthropicMessagesConfig()

    assert (
        config.get_complete_url(
            api_base=None,
            api_key=None,
            model="deepseek-v4-pro",
            optional_params={},
            litellm_params={},
        )
        == "https://api.deepseek.com/anthropic/v1/messages"
    )
    assert (
        config.get_complete_url(
            api_base="https://api.deepseek.com/anthropic/v1",
            api_key=None,
            model="deepseek-v4-pro",
            optional_params={},
            litellm_params={},
        )
        == "https://api.deepseek.com/anthropic/v1/messages"
    )
    assert (
        config.get_complete_url(
            api_base="https://api.deepseek.com/anthropic",
            api_key=None,
            model="deepseek-v4-pro",
            optional_params={},
            litellm_params={},
        )
        == "https://api.deepseek.com/anthropic/v1/messages"
    )
    assert (
        config.get_complete_url(
            api_base="https://api.deepseek.com",
            api_key=None,
            model="deepseek-v4-pro",
            optional_params={},
            litellm_params={},
        )
        == "https://api.deepseek.com/anthropic/v1/messages"
    )
    assert (
        config.get_complete_url(
            api_base="https://api.deepseek.com/v1",
            api_key=None,
            model="deepseek-v4-pro",
            optional_params={},
            litellm_params={},
        )
        == "https://api.deepseek.com/anthropic/v1/messages"
    )
    assert (
        config.get_complete_url(
            api_base="https://api.deepseek.com/v1/messages",
            api_key=None,
            model="deepseek-v4-pro",
            optional_params={},
            litellm_params={},
        )
        == "https://api.deepseek.com/anthropic/v1/messages"
    )


def test_deepseek_anthropic_messages_headers_use_deepseek_key():
    config = DeepSeekAnthropicMessagesConfig()

    headers, api_base = config.validate_anthropic_messages_environment(
        headers={},
        model="deepseek-v4-pro",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="sk-deepseek",
        api_base="https://example.test/anthropic",
    )

    assert api_base == "https://example.test/anthropic"
    assert headers["x-api-key"] == "sk-deepseek"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["content-type"] == "application/json"


def test_deepseek_anthropic_messages_preserves_thinking_and_sanitizes_custom_tools():
    config = DeepSeekAnthropicMessagesConfig()
    messages = [
        {
            "role": "user",
            "content": "Use the tool.",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "I should call the tool.",
                    "signature": "sig",
                },
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "get_weather",
                    "input": {"city": "Sao Paulo"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_123",
                    "content": "Sunny",
                }
            ],
        },
    ]

    request = config.transform_anthropic_messages_request(
        model="deepseek-v4-pro",
        messages=messages,
        anthropic_messages_optional_request_params={
            "max_tokens": 100,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tools": [
                {
                    "type": "custom",
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {"type": "object"},
                },
                {
                    "type": "web_search_20260209",
                    "name": "web_search",
                    "max_uses": 1,
                },
            ],
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert request["messages"] == messages
    assert request["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert request["tools"][0] == {
        "name": "get_weather",
        "description": "Get weather",
        "input_schema": {"type": "object"},
    }
    assert request["tools"][1]["type"] == "web_search_20260209"


def test_deepseek_anthropic_messages_preserves_native_web_tools():
    config = DeepSeekAnthropicMessagesConfig()
    tools = [
        {"type": "deepseek_web_search", "name": "deepseek_web_search"},
        {"type": "deepseek_read_url", "name": "deepseek_read_url"},
        {
            "type": "custom",
            "name": "get_weather",
            "input_schema": {"type": "object"},
        },
    ]

    request = config.transform_anthropic_messages_request(
        model="deepseek-flash",
        messages=[{"role": "user", "content": "Check latest updates."}],
        anthropic_messages_optional_request_params={
            "max_tokens": 256,
            "tools": tools,
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert request["tools"] == [
        {"type": "deepseek_web_search", "name": "deepseek_web_search"},
        {"type": "deepseek_read_url", "name": "deepseek_read_url"},
        {"name": "get_weather", "input_schema": {"type": "object"}},
    ]


def test_deepseek_server_tool_use_adds_no_per_call_surcharge():
    from litellm.types.utils import Choices, Message, ModelResponse, ServerToolUse, Usage

    base_response = ModelResponse(
        model="deepseek-flash",
        choices=[Choices(finish_reason="stop", index=0, message=Message(role="assistant", content="done"))],
        usage=Usage(prompt_tokens=2912, completion_tokens=52, total_tokens=2964),
    )
    with_web_tools = ModelResponse(
        model="deepseek-flash",
        choices=[Choices(finish_reason="stop", index=0, message=Message(role="assistant", content="done"))],
        usage=Usage(
            prompt_tokens=2912,
            completion_tokens=52,
            total_tokens=2964,
            server_tool_use=ServerToolUse(
                web_search_requests=2,
                tool_search=2,
                web_fetch_requests=1,
                tool_fetch=1,
                tool_image_search=1,
            ),
        ),
    )

    base_cost = litellm.completion_cost(
        completion_response=base_response,
        model="deepseek/deepseek-flash",
        custom_llm_provider="deepseek",
    )
    web_tools_cost = litellm.completion_cost(
        completion_response=with_web_tools,
        model="deepseek/deepseek-flash",
        custom_llm_provider="deepseek",
    )

    assert base_cost > 0
    assert web_tools_cost == base_cost


@pytest.mark.parametrize(
    ("optional_params", "expected_thinking", "expected_output_config"),
    [
        ({"output_config": {"effort": "low"}}, None, {"effort": "low"}),
        (
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}},
            {"type": "adaptive"},
            {"effort": "low"},
        ),
        (
            {"thinking": {"type": "enabled"}, "output_config": {"effort": "high"}},
            {"type": "enabled"},
            {"effort": "high"},
        ),
        (
            {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}},
            {"type": "disabled"},
            {"effort": "low"},
        ),
        ({"thinking": {"type": "adaptive"}}, {"type": "adaptive"}, None),
    ],
)
def test_deepseek_anthropic_messages_preserves_native_thinking_and_effort(
    optional_params,
    expected_thinking,
    expected_output_config,
):
    config = DeepSeekAnthropicMessagesConfig()

    request = config.transform_anthropic_messages_request(
        model="deepseek-flash",
        messages=[{"role": "user", "content": "hi"}],
        anthropic_messages_optional_request_params={
            "max_tokens": 4096,
            "temperature": 0.2,
            **optional_params,
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert request.get("thinking") == expected_thinking
    assert request.get("output_config") == expected_output_config
    assert request["temperature"] == 0.2


def test_deepseek_anthropic_messages_maps_reasoning_effort_without_budget_tokens():
    config = DeepSeekAnthropicMessagesConfig()

    low_request = config.transform_anthropic_messages_request(
        model="deepseek-flash",
        messages=[{"role": "user", "content": "hi"}],
        anthropic_messages_optional_request_params={
            "max_tokens": 4096,
            "reasoning_effort": "low",
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert low_request["thinking"] == {"type": "enabled"}
    assert low_request["output_config"] == {"effort": "low"}
    assert "reasoning_effort" not in low_request

    minimal_request = config.transform_anthropic_messages_request(
        model="deepseek-flash",
        messages=[{"role": "user", "content": "hi"}],
        anthropic_messages_optional_request_params={
            "max_tokens": 4096,
            "reasoning_effort": "minimal",
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert minimal_request["thinking"] == {"type": "enabled"}
    assert minimal_request["output_config"] == {"effort": "low"}
    assert "reasoning_effort" not in minimal_request

    none_request = config.transform_anthropic_messages_request(
        model="deepseek-flash",
        messages=[{"role": "user", "content": "hi"}],
        anthropic_messages_optional_request_params={
            "max_tokens": 4096,
            "reasoning_effort": "none",
            "output_config": {"effort": "high", "format": {"type": "json_schema"}},
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert none_request["thinking"] == {"type": "disabled"}
    assert none_request["output_config"] == {"format": {"type": "json_schema"}}
    assert "reasoning_effort" not in none_request

    non_string_request = config.transform_anthropic_messages_request(
        model="deepseek-flash",
        messages=[{"role": "user", "content": "hi"}],
        anthropic_messages_optional_request_params={
            "max_tokens": 4096,
            "reasoning_effort": 123,
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert "reasoning_effort" not in non_string_request
    assert "thinking" not in non_string_request

    explicit_override_request = config.transform_anthropic_messages_request(
        model="deepseek-flash",
        messages=[{"role": "user", "content": "hi"}],
        anthropic_messages_optional_request_params={
            "max_tokens": 4096,
            "reasoning_effort": "low",
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "max"},
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert explicit_override_request["thinking"] == {"type": "adaptive"}
    assert explicit_override_request["output_config"] == {"effort": "max"}

    disabled_thinking_request = config.transform_anthropic_messages_request(
        model="deepseek-flash",
        messages=[{"role": "user", "content": "hi"}],
        anthropic_messages_optional_request_params={
            "max_tokens": 4096,
            "reasoning_effort": "high",
            "thinking": {"type": "disabled"},
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert disabled_thinking_request["thinking"] == {"type": "disabled"}
    assert "output_config" not in disabled_thinking_request

    with pytest.raises(AnthropicError) as exc_info:
        config.transform_anthropic_messages_request(
            model="deepseek-flash",
            messages=[{"role": "user", "content": "hi"}],
            anthropic_messages_optional_request_params={
                "max_tokens": 4096,
                "reasoning_effort": "bogus",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
    assert exc_info.value.status_code == 400
