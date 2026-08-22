from typing import Final
from unittest.mock import AsyncMock, patch

import litellm
from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig
from litellm.types.utils import ModelInfoBase
from litellm.utils import _build_model_info, is_vision_explicitly_disabled


def _function_tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object"}},
    }


def test_drop_unsupported_tools_keeps_function_tools_only():
    optional_params = {
        "tools": [
            _function_tool("shell"),
            {"type": "namespace", "name": "container.exec"},
            _function_tool("apply_patch"),
        ],
        "tool_choice": "auto",
    }

    result = DeepSeekChatConfig._drop_unsupported_tools(optional_params)

    assert [tool["function"]["name"] for tool in result["tools"]] == [
        "shell",
        "apply_patch",
    ]
    assert all(tool["type"] == "function" for tool in result["tools"])
    assert result["tool_choice"] == "auto"


def test_drop_unsupported_tools_drops_dangling_tool_choice_when_none_survive():
    optional_params = {
        "tools": [{"type": "namespace", "name": "container.exec"}],
        "tool_choice": "required",
        "parallel_tool_calls": True,
        "temperature": 0.2,
    }

    result = DeepSeekChatConfig._drop_unsupported_tools(optional_params)

    assert "tools" not in result
    assert "tool_choice" not in result
    assert "parallel_tool_calls" not in result
    assert result["temperature"] == 0.2


def test_drop_unsupported_tools_is_noop_for_function_only():
    optional_params = {
        "tools": [_function_tool("shell")],
        "tool_choice": "auto",
    }

    result = DeepSeekChatConfig._drop_unsupported_tools(optional_params)

    assert result is optional_params


def test_drop_unsupported_tools_is_noop_without_tools():
    optional_params = {"temperature": 0.7}

    result = DeepSeekChatConfig._drop_unsupported_tools(optional_params)

    assert result is optional_params


def test_transform_request_strips_unsupported_tools_from_body():
    config = DeepSeekChatConfig()
    body = config.transform_request(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={
            "tools": [
                _function_tool("shell"),
                {"type": "namespace", "name": "container.exec"},
            ],
            "tool_choice": "auto",
        },
        litellm_params={},
        headers={},
    )

    assert [tool["type"] for tool in body["tools"]] == ["function"]
    assert body["tools"][0]["function"]["name"] == "shell"


async def test_async_transform_request_strips_unsupported_tools_from_body():
    config = DeepSeekChatConfig()
    body = await config.async_transform_request(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={
            "tools": [
                _function_tool("shell"),
                {"type": "namespace", "name": "container.exec"},
            ],
            "tool_choice": "auto",
        },
        litellm_params={},
        headers={},
    )

    assert [tool["type"] for tool in body["tools"]] == ["function"]
    assert body["tools"][0]["function"]["name"] == "shell"


def test_thinking_mode_active_bool_thinking_returns_false_without_crashing():
    config = DeepSeekChatConfig()
    assert config._thinking_mode_active(model="deepseek-reasoner", optional_params={"thinking": True}) is False


def test_deepseek_flash_vision_capability_survives_missing_remote_model_entry():
    with patch.dict(litellm.model_cost, {}, clear=True):
        assert litellm.supports_vision(model="deepseek-flash", custom_llm_provider="deepseek") is True
        assert litellm.supports_vision(model="deepseek/deepseek-flash", custom_llm_provider="deepseek") is True
        assert litellm.supports_function_calling(model="deepseek-flash", custom_llm_provider="deepseek") is False
        result: Final = DeepSeekChatConfig()._transform_messages(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this image"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                        },
                    ],
                }
            ],
            model="deepseek-flash",
        )

    assert isinstance(result[0]["content"], list)


def test_deepseek_flash_only_overrides_remote_vision_capability():
    remote_entry: Final[ModelInfoBase] = {
        "litellm_provider": "deepseek",
        "mode": "chat",
        "input_cost_per_token": 9.1e-7,
        "max_input_tokens": 123456,
        "supports_function_calling": False,
        "supports_vision": False,
    }
    with patch.dict(
        litellm.model_cost,
        {"deepseek-flash": remote_entry, "deepseek/deepseek-flash": remote_entry},
        clear=True,
    ):
        assert litellm.supports_vision(model="deepseek-flash", custom_llm_provider="deepseek") is True
        assert litellm.supports_vision(model="deepseek/deepseek-flash") is True
        assert is_vision_explicitly_disabled(model="deepseek-flash", custom_llm_provider="deepseek") is False
        assert is_vision_explicitly_disabled(model="deepseek/deepseek-flash") is False
        model_info: Final = _build_model_info(model="deepseek/deepseek-flash")

    assert model_info["supports_vision"] is True
    assert model_info["input_cost_per_token"] == 9.1e-7
    assert model_info["max_input_tokens"] == 123456
    assert model_info["supports_function_calling"] is False


def test_deepseek_capability_override_does_not_send_api_key_to_openai_model_discovery():
    with patch.object(litellm.module_level_client, "get") as mock_get:
        litellm.get_valid_models(
            check_provider_endpoint=True,
            custom_llm_provider="deepseek",
            api_key="deepseek-key-must-not-reach-openai",
        )

    openai_calls: Final = tuple(
        call for call in mock_get.call_args_list if str(call.kwargs.get("url", "")).startswith("https://api.openai.com")
    )
    assert openai_calls == ()


class TestDeepSeekVisionMultimodalContent:
    """Image content lists are forwarded only on vision models."""

    VISION_MODEL = "deepseek/deepseek-flash"
    LEGACY_VISION_MODEL = "deepseek/deepseek-v4-flash-vision-exp"
    NON_VISION_MODEL = "deepseek/deepseek-v4-pro"

    def setup_method(self):
        self.config = DeepSeekChatConfig()
        prior_entry = litellm.model_cost.get(self.VISION_MODEL)
        self._prior_registry_entry = dict(prior_entry) if prior_entry is not None else None
        litellm.register_model(
            {
                self.VISION_MODEL: {
                    "litellm_provider": "deepseek",
                    "mode": "chat",
                    "input_cost_per_token": 3e-07,
                    "output_cost_per_token": 1.2e-06,
                    "supports_vision": True,
                }
            }
        )

    def teardown_method(self):
        if self._prior_registry_entry is None:
            litellm.model_cost.pop(self.VISION_MODEL, None)
        else:
            litellm.model_cost[self.VISION_MODEL] = self._prior_registry_entry

    @staticmethod
    def _image_message(role="user", url="https://example.com/image.jpg"):
        return {
            "role": role,
            "content": [
                {"type": "text", "text": "what is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": url, "detail": "auto"},
                },
            ],
        }

    def test_user_image_list_forwarded_on_vision_model(self):
        result = self.config._transform_messages([self._image_message()], model=self.VISION_MODEL)

        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "image_url"
        assert result[0]["content"][1]["image_url"]["url"] == "https://example.com/image.jpg"

    def test_base64_image_list_forwarded_on_current_model(self):
        data_url = "data:image/png;base64,iVBORw0KGgo="

        result = self.config._transform_messages([self._image_message(url=data_url)], model=self.VISION_MODEL)

        assert result[0]["content"][1]["image_url"]["url"] == data_url

    def test_legacy_vision_alias_still_forwards_images(self):
        result = self.config._transform_messages([self._image_message()], model=self.LEGACY_VISION_MODEL)

        assert isinstance(result[0]["content"], list)

    def test_image_list_collapsed_on_non_vision_model(self):
        result = self.config._transform_messages([self._image_message()], model=self.NON_VISION_MODEL)

        assert result[0]["content"] == "what is in this image?"

    def test_image_list_collapsed_on_non_user_roles_even_on_vision_model(self):
        for role in ("assistant", "system"):
            result = self.config._transform_messages([self._image_message(role=role)], model=self.VISION_MODEL)

            assert result[0]["content"] == "what is in this image?"

    def test_tool_image_list_forwarded_on_vision_model(self):
        message = {**self._image_message(role="tool"), "tool_call_id": "call_1"}

        result = self.config._transform_messages([message], model=self.VISION_MODEL)

        assert result[0]["role"] == "tool"
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][1]["type"] == "image_url"

    def test_tool_reference_removed_without_dropping_base64_image(self):
        image_message = self._image_message(role="tool", url="data:image/png;base64,iVBORw0KGgo=")
        message = {
            **image_message,
            "content": [*image_message["content"], {"type": "tool_reference", "tool_name": "vision_tool"}],
            "tool_call_id": "call_1",
        }

        result = self.config._transform_messages([message], model=self.VISION_MODEL)

        assert result[0]["content"] == message["content"][:-1]

    def test_nested_file_data_without_filename_does_not_get_pdf_filename(self):
        data_url = "data:image/png;base64,iVBORw0KGgo="
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is in this image?"},
                    {
                        "type": "file",
                        "file": {"file_data": data_url},
                    },
                ],
            }
        ]

        result = self.config._transform_messages(messages, model=self.VISION_MODEL)

        assert result[0]["content"][1] == {
            "type": "file",
            "file_data": data_url,
        }

    def test_flat_file_data_block_forwarded_for_deepseek(self):
        data_url = "data:image/png;base64,iVBORw0KGgo="
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is in this image?"},
                    {"type": "file", "file_data": data_url, "filename": "image.png"},
                ],
            }
        ]

        result = self.config._transform_messages(messages, model=self.VISION_MODEL)

        assert result[0]["content"][1] == messages[0]["content"][1]

    def test_nested_file_id_block_flattened_for_deepseek(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is in this image?"},
                    {"type": "file", "file": {"file_id": "file-api-123"}},
                ],
            }
        ]

        result = self.config._transform_messages(messages, model=self.VISION_MODEL)

        assert result[0]["content"][1] == {"type": "file", "file_id": "file-api-123"}

    def test_remote_file_url_converted_before_flattening(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "summarize this file"},
                    {"type": "file", "file": {"file_id": "https://example.com/document.pdf"}},
                ],
            }
        ]
        data_url = "data:application/pdf;base64,JVBERi0="

        with patch(
            "litellm.llms.openai.chat.gpt_transformation.convert_url_to_base64",
            return_value=data_url,
        ) as convert:
            result = self.config._transform_messages(messages, model=self.VISION_MODEL)

        convert.assert_called_once_with("https://example.com/document.pdf")
        assert result[0]["content"][1] == {
            "type": "file",
            "file_data": data_url,
            "filename": "my_file.pdf",
        }

    async def test_async_remote_file_url_converted_before_flattening(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "summarize this file"},
                    {"type": "file", "file": {"file_id": "https://example.com/document.pdf"}},
                ],
            }
        ]
        data_url = "data:application/pdf;base64,JVBERi0="
        convert = AsyncMock(return_value=data_url)

        with (
            patch(
                "litellm.llms.openai.chat.gpt_transformation.async_convert_url_to_base64",
                convert,
            ),
            patch("litellm.llms.openai.chat.gpt_transformation.convert_url_to_base64") as sync_convert,
        ):
            result = await self.config.async_transform_request(
                model=self.VISION_MODEL,
                messages=messages,
                optional_params={},
                litellm_params={},
                headers={},
            )

        convert.assert_awaited_once_with("https://example.com/document.pdf")
        sync_convert.assert_not_called()
        assert result["messages"][0]["content"][1] == {
            "type": "file",
            "file_data": data_url,
            "filename": "my_file.pdf",
        }

    def test_tool_file_data_block_flattened_for_deepseek(self):
        data_url = "data:image/png;base64,iVBORw0KGgo="
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [
                    {"type": "text", "text": "tool image"},
                    {
                        "type": "file",
                        "file": {"file_data": data_url, "filename": "image.png"},
                    },
                ],
            }
        ]

        result = self.config._transform_messages(messages, model=self.VISION_MODEL)

        assert result[0]["content"][1] == {
            "type": "file",
            "file_data": data_url,
            "filename": "image.png",
        }

    def test_malformed_file_block_collapsed(self):
        for file_payload in (
            {},
            {"file_data": "data:image/png;base64,AAAA", "file_id": "file-api-123"},
            {"file_id": "file-api-123", "filename": "image.png"},
        ):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is in this image?"},
                        {"type": "file", "file": file_payload},
                    ],
                }
            ]

            result = self.config._transform_messages(messages, model=self.VISION_MODEL)

            assert result[0]["content"] == "what is in this image?"

    def test_malformed_media_only_block_collapsed_without_crashing(self):
        for media_block in ({"type": "image_url"}, {"type": "file", "file": {}}):
            messages = [{"role": "user", "content": [media_block]}]

            result = self.config._transform_messages(messages, model=self.VISION_MODEL)

            assert result[0]["content"] == ""

    def test_audio_block_collapsed_even_on_vision_model(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "transcribe this"},
                    {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}},
                ],
            }
        ]

        result = self.config._transform_messages(messages, model=self.VISION_MODEL)

        assert result[0]["content"] == "transcribe this"

    def test_typeless_image_block_collapses(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {"image_url": {"url": "https://example.com/image.jpg"}},
                ],
            }
        ]

        result = self.config._transform_messages(messages, model=self.VISION_MODEL)

        assert result[0]["content"] == "what is this"

    def test_text_only_content_list_collapses(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world"},
                ],
            }
        ]

        result = self.config._transform_messages(messages, model=self.VISION_MODEL)

        assert isinstance(result[0]["content"], str)
        assert result[0]["content"] == "Hello world"

    def test_search_results_text_appended_on_forwarded_message(self):
        message = self._image_message()
        message["search_results"] = [{"source": "kb", "content": [{"text": "article body"}]}]

        result = self.config._transform_messages([message], model=self.VISION_MODEL)

        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[-1] == {"type": "text", "text": "kbarticle body"}
        assert any(block.get("type") == "image_url" for block in content)
        assert "search_results" not in result[0]

    def test_search_results_text_kept_on_collapse(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "context: "}],
                "search_results": [{"source": "kb", "content": [{"text": "article body"}]}],
            }
        ]

        result = self.config._transform_messages(messages, model=self.NON_VISION_MODEL)

        assert result[0]["content"] == "context: kbarticle body"

    def test_responses_shape_blocks_collapse_even_on_vision_model(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                ],
            }
        ]

        result = self.config._transform_messages(messages, model=self.VISION_MODEL)

        assert result[0]["content"] == "what is this?"

    def test_image_block_missing_payload_collapses(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi"}, {"type": "image_url"}],
            }
        ]

        result = self.config._transform_messages(messages, model=self.VISION_MODEL)

        assert result[0]["content"] == "hi"

    def test_image_block_empty_payload_object_collapses(self):
        for payload in ({}, {"url": ""}, {"detail": "auto"}, None, 42):
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": payload}],
                }
            ]

            result = self.config._transform_messages(messages, model=self.VISION_MODEL)

            assert result[0]["content"] == "hi"

    def test_image_block_string_payload_forwarded(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": "https://example.com/image.jpg"},
                ],
            }
        ]

        result = self.config._transform_messages(messages, model=self.VISION_MODEL)

        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[1]["image_url"] == {"url": "https://example.com/image.jpg"}

    def test_text_block_missing_text_field_collapses(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "text"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                ],
            }
        ]

        result = self.config._transform_messages(messages, model=self.VISION_MODEL)

        assert result[0]["content"] == "hi"

    def test_string_content_search_results_folded_into_string(self):
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "summarize the docs",
                "search_results": [{"source": "kb", "content": [{"text": "article body"}]}],
            }
        ]

        result = self.config._transform_messages(messages, model=self.NON_VISION_MODEL)

        assert result[0]["content"] == "summarize the docskbarticle body"

    def test_plain_string_content_message_unchanged(self):
        messages = [{"role": "user", "content": "hello"}]

        result = self.config._transform_messages(messages, model=self.VISION_MODEL)

        assert result[0] is messages[0]

    def test_empty_content_list_untouched(self):
        messages = [{"role": "user", "content": []}]

        result = self.config._transform_messages(messages, model=self.NON_VISION_MODEL)

        assert result[0]["content"] == []

    def test_later_messages_still_collapsed_after_forwarded_one(self):
        messages = [
            self._image_message(),
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "and "},
                    {"type": "text", "text": "then?"},
                ],
            },
            self._image_message(),
        ]

        result = self.config._transform_messages(messages, model=self.VISION_MODEL)

        assert isinstance(result[0]["content"], list)
        assert result[1]["content"] == "and then?"
        assert isinstance(result[2]["content"], list)

    def test_transform_request_preserves_image_url_block(self):
        body = self.config.transform_request(
            model=self.VISION_MODEL,
            messages=[self._image_message()],
            optional_params={},
            litellm_params={},
            headers={},
        )

        content = body["messages"][0]["content"]
        assert isinstance(content, list)
        assert any(block.get("type") == "image_url" for block in content)

    async def test_async_transform_request_preserves_image_url_block(self):
        body = await self.config.async_transform_request(
            model=self.VISION_MODEL,
            messages=[self._image_message()],
            optional_params={},
            litellm_params={},
            headers={},
        )

        content = body["messages"][0]["content"]
        assert isinstance(content, list)
        assert any(block.get("type") == "image_url" for block in content)

    async def test_async_file_data_without_filename_does_not_get_pdf_filename(self):
        data_url = "data:image/png;base64,iVBORw0KGgo="
        body = await self.config.async_transform_request(
            model=self.VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is in this image?"},
                        {
                            "type": "file",
                            "file": {"file_data": data_url},
                        },
                    ],
                }
            ],
            optional_params={},
            litellm_params={},
            headers={},
        )

        assert body["messages"][0]["content"][1] == {
            "type": "file",
            "file_data": data_url,
        }


class TestDeepSeekThinkingParams:
    """Test thinking and reasoning_effort parameter handling for DeepSeek."""

    def setup_method(self):
        self.config = DeepSeekChatConfig()
        self.model = "deepseek-reasoner"

    def test_get_supported_openai_params_includes_thinking(self):
        """Test that thinking and reasoning_effort are in supported params."""
        params = self.config.get_supported_openai_params(self.model)
        assert "thinking" in params
        assert "reasoning_effort" in params

    def test_map_thinking_enabled(self):
        """Test that thinking={"type": "enabled"} is passed through correctly."""
        non_default_params = {"thinking": {"type": "enabled"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_thinking_with_budget_tokens_strips_budget(self):
        """Test that budget_tokens is stripped from thinking param (DeepSeek doesn't support it)."""
        non_default_params = {"thinking": {"type": "enabled", "budget_tokens": 2048}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # Should strip budget_tokens, only pass type
        assert result["thinking"] == {"type": "enabled"}
        assert "budget_tokens" not in result.get("thinking", {})

    def test_map_reasoning_effort_medium(self):
        """Test that reasoning_effort='medium' maps to thinking enabled."""
        non_default_params = {"reasoning_effort": "medium"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_reasoning_effort_low(self):
        """Test that reasoning_effort='low' maps to thinking enabled."""
        non_default_params = {"reasoning_effort": "low"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_reasoning_effort_high(self):
        """Test that reasoning_effort='high' maps to thinking enabled."""
        non_default_params = {"reasoning_effort": "high"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_reasoning_effort_none_does_not_enable_thinking(self):
        """Test that reasoning_effort='none' does not enable thinking."""
        non_default_params = {"reasoning_effort": "none"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "disabled"}

    def test_map_reasoning_effort_null_does_not_enable_thinking(self):
        """Test that reasoning_effort=None does not enable thinking."""
        non_default_params = {"reasoning_effort": None}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result

    def test_thinking_takes_precedence_over_reasoning_effort(self):
        """Test that thinking param takes precedence when both are provided."""
        non_default_params = {
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
        }
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # thinking should be set, reasoning_effort should not override
        assert result["thinking"] == {"type": "enabled"}

    def test_invalid_thinking_type_ignored(self):
        """Test that invalid thinking type values are ignored."""
        non_default_params = {"thinking": {"type": "invalid"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result

    def test_thinking_none_value_ignored(self):
        """Test that thinking=None is ignored."""
        non_default_params = {"thinking": None}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result

    def test_drop_unsupported_tools_removes_dangling_tool_choice(self):
        optional_params = {
            "tools": [
                {"type": "namespace", "name": "local_shell"},
                {"type": "function", "function": {"name": "get_weather"}},
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "local_shell"},
            },
            "parallel_tool_calls": True,
        }

        result = self.config._drop_unsupported_tools(optional_params)

        assert result["tools"] == [{"type": "function", "function": {"name": "get_weather"}}]
        assert "tool_choice" not in result
        assert result["parallel_tool_calls"] is True
