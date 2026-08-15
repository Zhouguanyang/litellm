from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import httpx

import litellm
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageResponse
from litellm.utils import convert_to_model_response_object

from ..common_utils import (
    BytePlusError,
    get_byteplus_base_url,
    get_byteplus_headers,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

BYTEPLUS_IMAGE_PARAMS: Final[tuple[OpenAIImageGenerationOptionalParams, ...]] = (
    "guidance_scale",
    "image",
    "n",
    "optimize_prompt_options",
    "output_format",
    "quality",
    "response_format",
    "seed",
    "sequential_image_generation",
    "sequential_image_generation_options",
    "size",
    "tools",
    "user",
    "watermark",
)


class BytePlusImageGenerationConfig(BaseImageGenerationConfig):
    """
    BytePlus Seedream / Dola Seedream image generation config
    Reference: https://docs.byteplus.com/en/docs/ModelArk/2582774
    """

    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return list(BYTEPLUS_IMAGE_PARAMS)

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params: Final = frozenset(BYTEPLUS_IMAGE_PARAMS)
        return {
            **optional_params,
            **{key: value for key, value in non_default_params.items() if key in supported_params},
        }

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        resolved_api_key: Final = (
            api_key or litellm.api_key or get_secret_str("BYTEPLUS_API_KEY") or get_secret_str("ARK_API_KEY")
        )
        if not resolved_api_key:
            raise ValueError("BytePlus API key is required. Set BYTEPLUS_API_KEY or ARK_API_KEY or pass api_key.")
        return get_byteplus_headers(api_key=resolved_api_key, extra_headers=headers)

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        base_url = (
            api_base
            or litellm.api_base
            or get_secret_str("BYTEPLUS_API_BASE")
            or get_secret_str("ARK_API_BASE")
            or get_byteplus_base_url()
        )
        base_url = base_url.rstrip("/")
        if base_url.endswith("/images/generations"):
            return base_url
        if base_url.endswith("/api/v3"):
            return f"{base_url}/images/generations"
        return f"{base_url}/api/v3/images/generations"

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BytePlusError:
        typed_headers: httpx.Headers = headers if isinstance(headers, httpx.Headers) else httpx.Headers(headers or {})
        return BytePlusError(
            status_code=status_code,
            message=error_message,
            headers=typed_headers,
        )

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        native_params: Final = {key: value for key, value in optional_params.items() if key in BYTEPLUS_IMAGE_PARAMS}
        configured_extra_body: Final = optional_params.get("extra_body")
        extra_body: Final[Mapping[str, object]] = (
            {key: value for key, value in configured_extra_body.items() if key not in ("model", "prompt")}
            if isinstance(configured_extra_body, dict)
            else {}
        )
        return {
            "model": model,
            "prompt": prompt,
            **native_params,
            **extra_body,
        }

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: object = None,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        response = raw_response.json()

        if logging_obj:
            logging_obj.post_call(
                input=request_data.get("prompt", ""),
                api_key=api_key,
                additional_args={"complete_input_dict": request_data},
                original_response=response,
            )

        image_response: ImageResponse = convert_to_model_response_object(
            response_object=response,
            model_response_object=model_response,
            response_type="image_generation",
        )

        return image_response
