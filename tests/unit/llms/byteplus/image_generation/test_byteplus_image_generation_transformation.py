import json

import httpx

import litellm
from litellm.llms.byteplus.image_generation.transformation import BytePlusImageGenerationConfig
from litellm.llms.custom_httpx.http_handler import HTTPHandler


class RequestRecorder:
    def __init__(self):
        self.request: httpx.Request | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.request = request
        return httpx.Response(
            status_code=200,
            json={
                "model": "dola-seedream-5-0-pro-260628",
                "created": 1786813332,
                "data": [{"url": "https://example.com/generated.png", "size": "1536x1536"}],
            },
        )

    def body(self) -> dict:
        assert self.request is not None
        return json.loads(self.request.content)


def make_client(recorder: RequestRecorder) -> HTTPHandler:
    return HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(recorder)))


class TestBytePlusImageGenerationConfig:
    def test_get_supported_openai_params(self):
        config = BytePlusImageGenerationConfig()
        params = config.get_supported_openai_params("byteplus/dola-seedream-5-0-pro-260628")
        assert "n" in params
        assert "size" in params
        assert "response_format" in params
        assert "image" in params
        assert "guidance_scale" in params
        assert "optimize_prompt_options" in params
        assert "output_format" in params
        assert "tools" in params
        assert "watermark" in params

    def test_get_complete_url(self):
        config = BytePlusImageGenerationConfig()
        url = config.get_complete_url(
            api_base="https://ark.ap-southeast.bytepluses.com/api/v3",
            api_key="key",
            model="byteplus/dola-seedream-5-0-pro-260628",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"

    def test_transform_image_generation_request(self):
        config = BytePlusImageGenerationConfig()
        req = config.transform_image_generation_request(
            model="dola-seedream-5-0-pro-260628",
            prompt="a cat",
            optional_params={"size": "2K", "output_format": "png"},
            litellm_params={},
            headers={},
        )
        assert req["model"] == "dola-seedream-5-0-pro-260628"
        assert req["prompt"] == "a cat"
        assert req["size"] == "2K"
        assert req["output_format"] == "png"

    def test_transform_image_generation_request_with_reference_images(self):
        config = BytePlusImageGenerationConfig()

        req = config.transform_image_generation_request(
            model="dola-seedream-5-0-pro-260628",
            prompt="keep the subject and change the style",
            optional_params={
                "image": ["https://example.com/reference.png"],
                "guidance_scale": 5.5,
                "optimize_prompt_options": {"mode": "standard"},
                "tools": [{"type": "web_search"}],
                "watermark": False,
            },
            litellm_params={},
            headers={},
        )

        assert req == {
            "model": "dola-seedream-5-0-pro-260628",
            "prompt": "keep the subject and change the style",
            "image": ["https://example.com/reference.png"],
            "guidance_scale": 5.5,
            "optimize_prompt_options": {"mode": "standard"},
            "tools": [{"type": "web_search"}],
            "watermark": False,
        }

    def test_transform_image_generation_request_extra_body_reserved_fields(self):
        config = BytePlusImageGenerationConfig()
        req = config.transform_image_generation_request(
            model="dola-seedream-5-0-pro-260628",
            prompt="a cat",
            optional_params={
                "extra_body": {
                    "model": "malicious-model",
                    "prompt": "malicious prompt",
                    "custom_field": "custom_val",
                }
            },
            litellm_params={},
            headers={},
        )
        assert req["model"] == "dola-seedream-5-0-pro-260628"
        assert req["prompt"] == "a cat"
        assert req["custom_field"] == "custom_val"


class TestBytePlusImageGenerationEndToEnd:
    def test_reference_image_reaches_byteplus_image_api(self):
        recorder = RequestRecorder()

        response = litellm.image_generation(
            model="byteplus/dola-seedream-5-0-pro-260628",
            prompt="keep the subject and change the style",
            image=["https://example.com/reference.png"],
            guidance_scale=5.5,
            optimize_prompt_options={"mode": "standard"},
            output_format="png",
            size="1.5K",
            tools=[{"type": "web_search"}],
            watermark=False,
            api_key="test-key",
            extra_headers={"X-Custom-Header": "custom-value"},
            client=make_client(recorder),
        )

        assert recorder.request is not None
        assert str(recorder.request.url) == "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
        assert recorder.request.headers["Authorization"] == "Bearer test-key"
        assert recorder.request.headers["X-Custom-Header"] == "custom-value"
        assert recorder.body() == {
            "model": "dola-seedream-5-0-pro-260628",
            "prompt": "keep the subject and change the style",
            "image": ["https://example.com/reference.png"],
            "guidance_scale": 5.5,
            "optimize_prompt_options": {"mode": "standard"},
            "output_format": "png",
            "size": "1.5K",
            "tools": [{"type": "web_search"}],
            "watermark": False,
        }
        assert response.data[0].url == "https://example.com/generated.png"
