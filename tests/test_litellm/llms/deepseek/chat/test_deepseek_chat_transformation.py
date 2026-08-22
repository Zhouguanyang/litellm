from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig


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


class TestDeepSeekVisionMultimodalContent:
    """Test that image/audio/file content lists are preserved for DeepSeek vision models."""

    def setup_method(self):
        self.config = DeepSeekChatConfig()

    def test_transform_messages_preserves_image_url_content_list(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/image.jpg",
                            "detail": "auto",
                        },
                    },
                ],
            }
        ]

        result = self.config._transform_messages(messages, model="deepseek-v4-flash-vision")

        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "image_url"
        assert result[0]["content"][1]["image_url"]["url"] == "https://example.com/image.jpg"

    def test_transform_messages_collapses_text_only_content_list(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world"},
                ],
            }
        ]

        result = self.config._transform_messages(messages, model="deepseek-chat")

        assert isinstance(result[0]["content"], str)
        assert result[0]["content"] == "Hello world"

    def test_transform_messages_keeps_search_results_text_on_collapse(self):
        """Backward compat: search_results text is folded into collapsed content."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "context: "}],
                "search_results": [{"source": "kb", "content": [{"text": "article body"}]}],
            }
        ]

        result = self.config._transform_messages(messages, model="deepseek-chat")

        assert result[0]["content"] == "context: kbarticle body"

    def test_transform_messages_preserves_empty_content_list(self):
        """Backward compat: empty content lists are left untouched."""
        messages = [{"role": "user", "content": []}]

        result = self.config._transform_messages(messages, model="deepseek-chat")

        assert result[0]["content"] == []

    def test_transform_request_preserves_image_url_block(self):
        body = self.config.transform_request(
            model="deepseek-v4-flash-vision",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.com/image.jpg",
                                "detail": "auto",
                            },
                        },
                    ],
                }
            ],
            optional_params={},
            litellm_params={},
            headers={},
        )

        content = body["messages"][0]["content"]
        assert isinstance(content, list)
        assert any(block.get("type") == "image_url" for block in content)

    async def test_async_transform_request_preserves_image_url_block(self):
        body = await self.config.async_transform_request(
            model="deepseek-v4-flash-vision",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.com/image.jpg",
                                "detail": "auto",
                            },
                        },
                    ],
                }
            ],
            optional_params={},
            litellm_params={},
            headers={},
        )

        content = body["messages"][0]["content"]
        assert isinstance(content, list)
        assert any(block.get("type") == "image_url" for block in content)
