import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.gemini.videos.transformation import GeminiVideoConfig
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    ModelResponseIterator as GeminiModelResponseIterator,
)
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.types.utils import (
    ModelResponse,
    TextCompletionResponse,
)

if TYPE_CHECKING:
    from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType

    from ..success_handler import PassThroughEndpointLogging
else:
    PassThroughEndpointLogging = Any
    EndpointType = Any


class GeminiPassthroughLoggingHandler:
    @staticmethod
    def gemini_passthrough_handler(
        httpx_response: httpx.Response,
        response_body: dict,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: dict,
        **kwargs,
    ) -> PassThroughEndpointLoggingTypedDict:
        if "predictLongRunning" in url_route:
            model = GeminiPassthroughLoggingHandler.extract_model_from_url(url_route)

            gemini_video_config: Final = GeminiVideoConfig()
            litellm_video_response: Final = gemini_video_config.transform_video_create_response(
                model=model,
                raw_response=httpx_response,
                logging_obj=logging_obj,
                custom_llm_provider="gemini",
                request_data=request_body,
            )
            logging_obj.model = model
            logging_obj.model_call_details["model"] = model
            logging_obj.model_call_details["custom_llm_provider"] = "gemini"
            logging_obj.custom_llm_provider = "gemini"

            response_cost: Final = litellm.completion_cost(
                completion_response=litellm_video_response,
                model=model,
                custom_llm_provider="gemini",
                call_type="create_video",
            )

            # Set response_cost in _hidden_params to prevent recalculation
            if not hasattr(litellm_video_response, "_hidden_params"):
                litellm_video_response._hidden_params = {}
            litellm_video_response._hidden_params["response_cost"] = response_cost

            kwargs["response_cost"] = response_cost
            kwargs["model"] = model
            kwargs["custom_llm_provider"] = "gemini"
            logging_obj.model_call_details["response_cost"] = response_cost
            return {
                "result": litellm_video_response,
                "kwargs": kwargs,
            }

        if "generateContent" in url_route:
            model = GeminiPassthroughLoggingHandler.extract_model_from_url(url_route)

            # Use Gemini config for transformation
            instance_of_gemini_llm: Final = litellm.GoogleAIStudioGeminiConfig()
            litellm_model_response: Final[ModelResponse] = instance_of_gemini_llm.transform_response(
                model=model,
                messages=[{"role": "user", "content": "no-message-pass-through-endpoint"}],
                raw_response=httpx_response,
                model_response=litellm.ModelResponse(),
                logging_obj=logging_obj,
                optional_params={},
                litellm_params={},
                api_key="",
                request_data={},
                encoding=getattr(litellm, "encoding", None),
            )
            kwargs = GeminiPassthroughLoggingHandler._create_gemini_response_logging_payload_for_generate_content(
                litellm_model_response=litellm_model_response,
                model=model,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
                logging_obj=logging_obj,
                custom_llm_provider="gemini",
            )

            return {
                "result": litellm_model_response,
                "kwargs": kwargs,
            }
        else:
            return {
                "result": None,
                "kwargs": kwargs,
            }

    @staticmethod
    def _handle_logging_gemini_collected_chunks(
        litellm_logging_obj: LiteLLMLoggingObj,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
        request_body: dict,
        endpoint_type: EndpointType,
        start_time: datetime,
        all_chunks: list[str],
        model: str | None,
        end_time: datetime,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Takes raw chunks from Gemini passthrough endpoint and logs them in litellm callbacks

        - Builds complete response from chunks
        - Creates standard logging object
        - Logs in litellm callbacks
        """
        kwargs: dict[str, Any] = {}
        model = model or GeminiPassthroughLoggingHandler.extract_model_from_url(url_route)
        complete_streaming_response: Final = GeminiPassthroughLoggingHandler._build_complete_streaming_response(
            all_chunks=all_chunks,
            litellm_logging_obj=litellm_logging_obj,
            model=model,
            url_route=url_route,
        )

        if complete_streaming_response is None:
            verbose_proxy_logger.error(
                "Unable to build complete streaming response for Gemini passthrough endpoint, not logging..."
            )
            return {
                "result": None,
                "kwargs": kwargs,
            }

        kwargs = GeminiPassthroughLoggingHandler._create_gemini_response_logging_payload_for_generate_content(
            litellm_model_response=complete_streaming_response,
            model=model,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
            logging_obj=litellm_logging_obj,
            custom_llm_provider="gemini",
        )

        return {
            "result": complete_streaming_response,
            "kwargs": kwargs,
        }

    @staticmethod
    def _build_complete_streaming_response(
        all_chunks: list[str],
        litellm_logging_obj: LiteLLMLoggingObj,
        model: str,
        url_route: str,
    ) -> ModelResponse | TextCompletionResponse | None:
        parsed_chunks = []
        if "generateContent" in url_route or "streamGenerateContent" in url_route:
            gemini_iterator: Final[Any] = GeminiModelResponseIterator(
                streaming_response=None,
                sync_stream=False,
                logging_obj=litellm_logging_obj,
            )
            chunk_parsing_logic: Final[Any] = gemini_iterator._common_chunk_parsing_logic
            parsed_chunks = [chunk_parsing_logic(chunk) for chunk in all_chunks]
        else:
            return None

        if len(parsed_chunks) == 0:
            return None

        all_openai_chunks: Final = []
        for parsed_chunk in parsed_chunks:
            if parsed_chunk is None:
                continue
            all_openai_chunks.append(parsed_chunk)

        complete_streaming_response: Final = litellm.stream_chunk_builder(chunks=all_openai_chunks)

        return complete_streaming_response

    @staticmethod
    def extract_model_from_url(url: str) -> str:
        pattern: Final = r"/models/([^:]+)"
        match: Final = re.search(pattern, url)
        if match:
            return match.group(1)
        return "unknown"

    @staticmethod
    def _count_generated_images(
        litellm_model_response: ModelResponse | TextCompletionResponse,
    ) -> int:
        image_count = 0
        choices = getattr(litellm_model_response, "choices", None) or []
        for choice in choices:
            message = getattr(choice, "message", None)
            if message is None and isinstance(choice, dict):
                message = choice.get("message")
            if message is None:
                continue
            images = message.get("images") if isinstance(message, dict) else getattr(message, "images", None)
            if isinstance(images, list):
                image_count += len(images)
        return image_count

    @staticmethod
    def _strip_provider_prefix(model: str | None) -> str | None:
        if not isinstance(model, str):
            return None
        if "/" not in model:
            return model
        return model.split("/", 1)[1]

    @staticmethod
    def _is_forced_flat_pricing_model_info(model_info: object) -> bool:
        return isinstance(model_info, dict) and (
            model_info.get("output_cost_per_request") is not None
            or bool(model_info.get("force_output_cost_per_image"))
        )

    @staticmethod
    def _get_forced_flat_pricing_model_info_from_deployment(
        deployment: dict[str, Any],
    ) -> dict[str, Any] | None:
        model_info = deployment.get("model_info")
        if GeminiPassthroughLoggingHandler._is_forced_flat_pricing_model_info(model_info):
            return model_info

        if isinstance(model_info, dict):
            model_id = model_info.get("id")
            registered_model_info = litellm.model_cost.get(model_id) if isinstance(model_id, str) else None
            if GeminiPassthroughLoggingHandler._is_forced_flat_pricing_model_info(registered_model_info):
                return registered_model_info

        litellm_params = deployment.get("litellm_params")
        if GeminiPassthroughLoggingHandler._is_forced_flat_pricing_model_info(litellm_params):
            return litellm_params

        return None

    @staticmethod
    def _get_model_info_from_router(model: str) -> dict[str, Any] | None:
        try:
            from litellm.proxy.proxy_server import llm_model_list, llm_router
        except Exception:
            return None
        deployments = []
        if llm_router is not None:
            deployments.extend(getattr(llm_router, "model_list", []) or [])
        deployments.extend(llm_model_list or [])
        if not deployments:
            return None

        for deployment in deployments:
            if not isinstance(deployment, dict):
                continue
            litellm_params = deployment.get("litellm_params", {}) or {}
            litellm_model = litellm_params.get("model")
            if model not in (
                deployment.get("model_name"),
                litellm_model,
                GeminiPassthroughLoggingHandler._strip_provider_prefix(litellm_model),
            ):
                continue
            model_info = GeminiPassthroughLoggingHandler._get_forced_flat_pricing_model_info_from_deployment(
                deployment=deployment
            )
            if model_info is not None:
                return model_info
        return None

    @staticmethod
    def _get_forced_flat_pricing_model_info(
        model: str,
    ) -> dict[str, Any] | None:
        router_model_info = GeminiPassthroughLoggingHandler._get_model_info_from_router(model=model)
        if router_model_info is not None:
            return router_model_info

        try:
            model_info = litellm.get_model_info(model=model, custom_llm_provider="gemini")
        except Exception:
            return None

        if GeminiPassthroughLoggingHandler._is_forced_flat_pricing_model_info(model_info):
            return model_info
        return None

    @staticmethod
    def _get_forced_flat_pricing_cost(
        litellm_model_response: ModelResponse | TextCompletionResponse,
        model: str,
    ) -> float | None:
        model_info = GeminiPassthroughLoggingHandler._get_forced_flat_pricing_model_info(
            model=model,
        )
        if model_info is None:
            return None

        output_cost_per_request = model_info.get("output_cost_per_request")
        if output_cost_per_request is not None:
            return float(output_cost_per_request)

        image_count = GeminiPassthroughLoggingHandler._count_generated_images(
            litellm_model_response=litellm_model_response,
        )
        if image_count == 0:
            return None

        output_cost_per_image: float = model_info.get("output_cost_per_image") or 0.0
        return output_cost_per_image * image_count

    @staticmethod
    def _create_gemini_response_logging_payload_for_generate_content(
        litellm_model_response: ModelResponse | TextCompletionResponse,
        model: str,
        kwargs: dict,
        start_time: datetime,
        end_time: datetime,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str,
    ):
        """
        Create the standard logging object for Gemini passthrough generateContent (streaming and non-streaming)
        """

        flat_response_cost: Final = GeminiPassthroughLoggingHandler._get_forced_flat_pricing_cost(
            litellm_model_response=litellm_model_response,
            model=model,
        )
        response_cost: Final = (
            flat_response_cost
            if flat_response_cost is not None
            else litellm.completion_cost(
                completion_response=litellm_model_response,
                model=model,
                custom_llm_provider="gemini",
            )
        )

        kwargs["response_cost"] = response_cost
        kwargs["model"] = model
        kwargs["custom_llm_provider"] = custom_llm_provider

        # pretty print standard logging object
        verbose_proxy_logger.debug("kwargs= %s", kwargs)

        # set litellm_call_id to logging response object
        litellm_model_response.id = logging_obj.litellm_call_id
        logging_obj.model = litellm_model_response.model or model
        logging_obj.model_call_details["model"] = logging_obj.model
        logging_obj.model_call_details["custom_llm_provider"] = custom_llm_provider
        logging_obj.model_call_details["response_cost"] = response_cost
        return kwargs
