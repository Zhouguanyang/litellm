import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import litellm
import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.gemini_passthrough_logging_handler import (
    GeminiPassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    EndpointType,
    PassthroughStandardLoggingPayload,
)
from litellm.types.utils import ModelResponse


class TestGeminiPassthroughLoggingHandler:
    """Test the Gemini passthrough logging handler for cost tracking."""

    def setup_method(self):
        """Set up test fixtures"""
        self.start_time = datetime.now()
        self.end_time = datetime.now()
        self.handler = GeminiPassthroughLoggingHandler()

        # Mock Gemini generateContent response
        self.mock_gemini_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello! How can I help you today?"}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": 0,
                    "safetyRatings": [
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "probability": "NEGLIGIBLE",
                        },
                    ],
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 8,
                "totalTokenCount": 18,
            },
        }

    def _create_mock_httpx_response(self) -> httpx.Response:
        """Create a mock httpx.Response for testing"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = json.dumps(self.mock_gemini_response)
        mock_response.json.return_value = self.mock_gemini_response
        mock_response.headers = {"content-type": "application/json"}
        return mock_response

    def _create_mock_logging_obj(self) -> LiteLLMLoggingObj:
        """Create a mock logging object for testing"""
        mock_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {}
        mock_logging_obj.optional_params = {}
        mock_logging_obj.litellm_call_id = "test-call-id-123"
        return mock_logging_obj

    def _create_passthrough_logging_payload(self) -> PassthroughStandardLoggingPayload:
        """Create a mock passthrough logging payload for testing"""
        return PassthroughStandardLoggingPayload(
            url="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            request_body={"contents": [{"parts": [{"text": "Hello"}]}]},
            request_method="POST",
        )

    def test_is_gemini_route(self):
        """Test that Gemini routes are correctly identified"""
        from litellm.proxy.pass_through_endpoints.success_handler import (
            PassThroughEndpointLogging,
        )

        handler = PassThroughEndpointLogging()

        # Test generateContent endpoint
        assert (
            handler.is_gemini_route(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
                custom_llm_provider="gemini",
            )
            is True
        )

        # Test streamGenerateContent endpoint
        assert (
            handler.is_gemini_route(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent",
                custom_llm_provider="gemini",
            )
            is True
        )

        # Test non-Gemini endpoint
        assert (
            handler.is_gemini_route(
                "https://api.openai.com/v1/chat/completions",
                custom_llm_provider="openai",
            )
            is False
        )

    def test_extract_model_from_url(self):
        """Test that model is correctly extracted from Gemini URLs"""
        # Test generateContent endpoint
        model = GeminiPassthroughLoggingHandler.extract_model_from_url(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        )
        assert model == "gemini-1.5-flash"

        # Test streamGenerateContent endpoint
        model = GeminiPassthroughLoggingHandler.extract_model_from_url(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:streamGenerateContent"
        )
        assert model == "gemini-1.5-pro"

    @patch("litellm.completion_cost")
    @patch(
        "litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload"
    )
    def test_gemini_passthrough_handler_success(
        self, mock_get_standard_logging, mock_completion_cost
    ):
        """Test successful cost tracking for Gemini generateContent endpoint"""
        # Arrange
        mock_completion_cost.return_value = 0.000045
        mock_get_standard_logging.return_value = {"test": "logging_payload"}

        mock_httpx_response = self._create_mock_httpx_response()
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()

        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gemini-1.5-flash",
        }

        # Act
        result = GeminiPassthroughLoggingHandler.gemini_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=self.mock_gemini_response,
            logging_obj=mock_logging_obj,
            url_route="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"contents": [{"parts": [{"text": "Hello"}]}]},
            **kwargs,
        )

        # Assert
        assert result is not None
        assert "result" in result
        assert "kwargs" in result
        assert result["kwargs"]["response_cost"] == 0.000045
        assert result["kwargs"]["model"] == "gemini-1.5-flash"
        assert result["kwargs"]["custom_llm_provider"] == "gemini"

        # Verify cost calculation was called
        mock_completion_cost.assert_called_once()

        # Verify logging object was updated
        assert mock_logging_obj.model_call_details["response_cost"] == 0.000045
        assert mock_logging_obj.model_call_details["model"] == "gemini-1.5-flash"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "gemini"

    @patch("litellm.completion_cost")
    def test_gemini_passthrough_handler_streaming(self, mock_completion_cost):
        """Test cost tracking for Gemini streaming endpoint"""
        # Arrange
        mock_completion_cost.return_value = 0.000030

        # Mock streaming response chunks
        mock_chunks = [
            {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]},
            {"candidates": [{"content": {"parts": [{"text": " there!"}]}}]},
        ]

        mock_httpx_response = self._create_mock_httpx_response()
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()

        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gemini-1.5-flash",
        }

        # Act - Use generateContent URL since that's what the handler processes
        result = GeminiPassthroughLoggingHandler.gemini_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=mock_chunks,
            logging_obj=mock_logging_obj,
            url_route="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"contents": [{"parts": [{"text": "Hello"}]}]},
            **kwargs,
        )

        # Assert
        assert result is not None
        assert "result" in result
        assert "kwargs" in result
        assert result["kwargs"]["response_cost"] == 0.000030
        assert result["kwargs"]["model"] == "gemini-1.5-flash"
        assert result["kwargs"]["custom_llm_provider"] == "gemini"

        # Verify cost calculation was called
        mock_completion_cost.assert_called_once()

    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers."
        "gemini_passthrough_logging_handler.litellm.get_model_info"
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers."
        "gemini_passthrough_logging_handler.litellm.completion_cost"
    )
    def test_gemini_passthrough_generate_content_can_force_flat_image_pricing(
        self,
        mock_completion_cost,
        mock_get_model_info,
    ):
        mock_get_model_info.return_value = {
            "force_output_cost_per_image": True,
            "output_cost_per_image": 0.017,
        }
        litellm_model_response = ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "images": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,img1",
                                    "detail": "auto",
                                },
                                "index": 0,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,img2",
                                    "detail": "auto",
                                },
                                "index": 1,
                            },
                        ],
                    }
                }
            ]
        )
        mock_logging_obj = self._create_mock_logging_obj()

        kwargs = GeminiPassthroughLoggingHandler._create_gemini_response_logging_payload_for_generate_content(
            litellm_model_response=litellm_model_response,
            model="gemini-3.1-flash-image",
            kwargs={},
            start_time=self.start_time,
            end_time=self.end_time,
            logging_obj=mock_logging_obj,
            custom_llm_provider="gemini",
        )

        assert kwargs["response_cost"] == pytest.approx(0.034)
        assert mock_logging_obj.model_call_details["response_cost"] == pytest.approx(0.034)
        mock_completion_cost.assert_not_called()

    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers."
        "gemini_passthrough_logging_handler.litellm.get_model_info"
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers."
        "gemini_passthrough_logging_handler.litellm.completion_cost"
    )
    def test_gemini_passthrough_generate_content_can_force_flat_request_pricing_for_text(
        self,
        mock_completion_cost,
        mock_get_model_info,
    ):
        mock_get_model_info.return_value = {
            "output_cost_per_request": 0.017,
        }
        litellm_model_response = ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": "done",
                    }
                }
            ]
        )
        mock_logging_obj = self._create_mock_logging_obj()

        kwargs = GeminiPassthroughLoggingHandler._create_gemini_response_logging_payload_for_generate_content(
            litellm_model_response=litellm_model_response,
            model="gemini-3.1-flash-image",
            kwargs={},
            start_time=self.start_time,
            end_time=self.end_time,
            logging_obj=mock_logging_obj,
            custom_llm_provider="gemini",
        )

        assert kwargs["response_cost"] == pytest.approx(0.017)
        assert mock_logging_obj.model_call_details["response_cost"] == pytest.approx(0.017)
        mock_completion_cost.assert_not_called()

    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers."
        "gemini_passthrough_logging_handler.litellm.completion_cost"
    )
    def test_gemini_passthrough_reads_flat_request_pricing_from_registered_deployment(
        self,
        mock_completion_cost,
    ):
        router = MagicMock()
        router.model_list = [
            {
                "model_name": "gemini-3.1-flash-image",
                "litellm_params": {"model": "gemini/gemini-3.1-flash-image"},
                "model_info": {"id": "deployment-id"},
            }
        ]
        litellm_model_response = ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": "done",
                    }
                }
            ]
        )
        mock_logging_obj = self._create_mock_logging_obj()

        with (
            patch("litellm.proxy.proxy_server.llm_router", router),
            patch.dict(
                litellm.model_cost,
                {"deployment-id": {"output_cost_per_request": 0.017}},
            ),
        ):
            kwargs = GeminiPassthroughLoggingHandler._create_gemini_response_logging_payload_for_generate_content(
                litellm_model_response=litellm_model_response,
                model="gemini-3.1-flash-image",
                kwargs={},
                start_time=self.start_time,
                end_time=self.end_time,
                logging_obj=mock_logging_obj,
                custom_llm_provider="gemini",
            )

        assert kwargs["response_cost"] == pytest.approx(0.017)
        assert mock_logging_obj.model_call_details["response_cost"] == pytest.approx(0.017)
        mock_completion_cost.assert_not_called()

    @pytest.mark.asyncio
    @patch(
        "litellm.proxy.pass_through_endpoints.streaming_handler."
        "VertexPassthroughLoggingHandler._handle_logging_vertex_collected_chunks"
    )
    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers.gemini_passthrough_logging_handler."
        "litellm.completion_cost"
    )
    async def test_gemini_passthrough_streaming_dispatch_preserves_flat_request_pricing(
        self,
        mock_completion_cost,
        mock_vertex_handler,
    ):
        router = MagicMock()
        router.model_list = [
            {
                "model_name": "gemini-3.1-flash-image",
                "litellm_params": {"model": "gemini/gemini-3.1-flash-image"},
                "model_info": {
                    "output_cost_per_request": 0.017,
                },
            }
        ]
        url_route = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-3.1-flash-image:streamGenerateContent?alt=sse"
        )
        logging_obj = LiteLLMLoggingObj(
            model="gemini-3.1-flash-image",
            messages=[{"role": "user", "content": "draw"}],
            stream=True,
            call_type="pass_through_endpoint",
            start_time=self.start_time,
            litellm_call_id="test-call-id-123",
            function_id="1245",
            dynamic_success_callbacks=[],
            dynamic_async_success_callbacks=[],
        )
        logging_obj.update_environment_variables(
            model="gemini-3.1-flash-image",
            user="unknown",
            optional_params={},
            litellm_params={
                "metadata": {},
                "proxy_server_request": {
                    "url": url_route,
                    "method": "POST",
                    "body": {"contents": [{"parts": [{"text": "draw"}]}]},
                    "headers": {},
                },
            },
            call_type="pass_through_endpoint",
        )
        logging_obj.model_call_details["custom_llm_provider"] = "gemini"
        raw_bytes = [
            b'data: {"responseId":"resp-1","candidates":[{"content":{"role":"model","parts":'
            b'[{"thought":true,"inlineData":{"mimeType":"image/png","data":"thinking-img"}},'
            b'{"inlineData":{"mimeType":"image/png","data":"final-img"}}]},"finishReason":"STOP","index":0}],'
            b'"usageMetadata":{"promptTokenCount":3279,"candidatesTokenCount":1423,"totalTokenCount":4702}}\n\n'
        ]
        response_cost_calculator = MagicMock(return_value=0.000261)
        logging_obj._response_cost_calculator = response_cost_calculator

        with (
            patch.object(litellm, "_async_success_callback", []),
            patch.object(litellm, "success_callback", []),
            patch("litellm.proxy.proxy_server.llm_router", router),
        ):
            await PassThroughStreamingHandler._route_streaming_logging_to_handler(
                litellm_logging_obj=logging_obj,
                passthrough_success_handler_obj=PassThroughEndpointLogging(),
                url_route=url_route,
                request_body={"contents": [{"parts": [{"text": "draw"}]}]},
                endpoint_type=EndpointType.VERTEX_AI,
                start_time=self.start_time,
                raw_bytes=raw_bytes,
                end_time=self.end_time,
                model=None,
            )

        standard_logging_object = logging_obj.model_call_details["standard_logging_object"]
        response = standard_logging_object["response"]
        images = response["choices"][0]["message"]["images"]
        assert logging_obj.model_call_details["response_cost"] == pytest.approx(0.017)
        assert standard_logging_object["response_cost"] == pytest.approx(0.017)
        assert len(images) == 1
        assert images[0]["image_url"]["url"] == "data:image/png;base64,final-img"
        response_cost_calculator.assert_not_called()
        mock_vertex_handler.assert_not_called()
        mock_completion_cost.assert_not_called()

    def test_gemini_passthrough_handler_non_gemini_route(self):
        """Test that non-Gemini routes return None"""
        mock_httpx_response = self._create_mock_httpx_response()
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()

        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gpt-4o",
        }

        # Act
        result = GeminiPassthroughLoggingHandler.gemini_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=self.mock_gemini_response,
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/chat/completions",  # Non-Gemini route (no generateContent)
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            **kwargs,
        )

        # Assert - the handler should return a dict with None result for non-Gemini routes
        assert result is not None
        assert result["result"] is None
        assert "kwargs" in result

    @pytest.mark.asyncio
    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers."
        "gemini_passthrough_logging_handler.litellm.completion_cost",
        return_value=0.000050,
    )
    async def test_pass_through_success_handler_gemini_routing(
        self, mock_completion_cost
    ):
        """Test that the success handler correctly routes Gemini requests to the Gemini handler"""
        handler = PassThroughEndpointLogging()

        # Mock the logging object
        mock_logging_obj = self._create_mock_logging_obj()

        # Mock the _handle_logging method to capture the call
        handler._handle_logging = AsyncMock()

        # Mock httpx response
        mock_response = self._create_mock_httpx_response()

        # Create passthrough logging payload
        passthrough_logging_payload = self._create_passthrough_logging_payload()

        # Call the success handler with Gemini route and provider
        result = await handler.pass_through_async_success_handler(
            httpx_response=mock_response,
            response_body=self.mock_gemini_response,
            logging_obj=mock_logging_obj,
            url_route="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"contents": [{"parts": [{"text": "Hello"}]}]},
            passthrough_logging_payload=passthrough_logging_payload,
            custom_llm_provider="gemini",
        )

        # Assert - The success handler returns None on success (following the pattern from other tests)
        assert result is None

        # Verify that the logging object has the cost set (from Gemini handler)
        assert mock_logging_obj.model_call_details["response_cost"] == 0.000050
        assert mock_logging_obj.model_call_details["model"] == "gemini-2.0-flash"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "gemini"

        # Verify that _handle_logging was called with the correct kwargs
        handler._handle_logging.assert_called_once()
        call_kwargs = handler._handle_logging.call_args[1]
        assert call_kwargs["response_cost"] == 0.000050
        assert call_kwargs["model"] == "gemini-2.0-flash"
        assert call_kwargs["custom_llm_provider"] == "gemini"

    @pytest.mark.asyncio
    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers."
        "gemini_passthrough_logging_handler.litellm.completion_cost"
    )
    async def test_pass_through_success_handler_gemini_v1beta_generate_content_uses_flat_request_pricing(
        self, mock_completion_cost
    ):
        handler = PassThroughEndpointLogging()
        handler._handle_logging = AsyncMock()
        router = MagicMock()
        router.model_list = [
            {
                "model_name": "gemini-3.1-flash-image",
                "litellm_params": {"model": "gemini/gemini-3.1-flash-image"},
                "model_info": {
                    "output_cost_per_request": 0.017,
                },
            }
        ]
        mock_logging_obj = self._create_mock_logging_obj()
        mock_response = self._create_mock_httpx_response()
        url_route = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-3.1-flash-image:generateContent"
        )
        passthrough_logging_payload = PassthroughStandardLoggingPayload(
            url=url_route,
            request_body={"contents": [{"parts": [{"text": "draw"}]}]},
            request_method="POST",
        )

        with patch("litellm.proxy.proxy_server.llm_router", router):
            result = await handler.pass_through_async_success_handler(
                httpx_response=mock_response,
                response_body=self.mock_gemini_response,
                logging_obj=mock_logging_obj,
                url_route=url_route,
                result="",
                start_time=self.start_time,
                end_time=self.end_time,
                cache_hit=False,
                request_body={"contents": [{"parts": [{"text": "draw"}]}]},
                passthrough_logging_payload=passthrough_logging_payload,
            )

        assert result is None
        assert mock_logging_obj.model_call_details["response_cost"] == pytest.approx(0.017)
        assert mock_logging_obj.model_call_details["model"] == "gemini-3.1-flash-image"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "gemini"
        handler._handle_logging.assert_called_once()
        call_kwargs = handler._handle_logging.call_args[1]
        assert call_kwargs["response_cost"] == pytest.approx(0.017)
        assert call_kwargs["model"] == "gemini-3.1-flash-image"
        assert call_kwargs["custom_llm_provider"] == "gemini"
        mock_completion_cost.assert_not_called()

    @patch("litellm.completion_cost")
    def test_veo3_passthrough_cost_tracking(self, mock_completion_cost):
        """Test Veo3 video generation cost tracking for passthrough requests"""
        # Mock the completion_cost to return the expected video generation cost
        # For veo-2.0-generate-001 with 8 seconds: 0.35 * 8 = 2.8
        expected_cost = 0.35 * 8.0  # $2.80
        mock_completion_cost.return_value = expected_cost

        # Mock Veo3 predictLongRunning response
        mock_veo_response = {"name": "operations/1234567890123456789"}

        mock_httpx_response = MagicMock(spec=httpx.Response)
        mock_httpx_response.status_code = 200
        mock_httpx_response.json.return_value = mock_veo_response
        mock_httpx_response.headers = {"content-type": "application/json"}

        mock_logging_obj = self._create_mock_logging_obj()

        # Request body with durationSeconds
        request_body = {
            "instances": [
                {
                    "prompt": "A close up of two people staring at a cryptic drawing on a wall,"
                }
            ],
            "parameters": {"durationSeconds": 8},
        }

        kwargs = {
            "passthrough_logging_payload": PassthroughStandardLoggingPayload(
                url="https://generativelanguage.googleapis.com/v1beta/models/veo-2.0-generate-001:predictLongRunning",
                request_body=request_body,
                request_method="POST",
            ),
        }

        # Act
        result = GeminiPassthroughLoggingHandler.gemini_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=mock_veo_response,
            logging_obj=mock_logging_obj,
            url_route="https://generativelanguage.googleapis.com/v1beta/models/veo-2.0-generate-001:predictLongRunning",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body=request_body,
            **kwargs,
        )

        # Assert
        assert result is not None
        assert "result" in result
        assert "kwargs" in result

        # Verify the cost is calculated correctly
        assert result["kwargs"]["response_cost"] == expected_cost
        assert result["kwargs"]["model"] == "veo-2.0-generate-001"
        assert result["kwargs"]["custom_llm_provider"] == "gemini"

        # Verify completion_cost was called with create_video call_type
        mock_completion_cost.assert_called_once()
        call_args = mock_completion_cost.call_args
        assert call_args.kwargs.get("call_type") == "create_video"
        assert call_args.kwargs.get("custom_llm_provider") == "gemini"
        assert call_args.kwargs.get("model") == "veo-2.0-generate-001"

        # Verify the response object has _hidden_params with response_cost
        video_response = result["result"]
        assert hasattr(video_response, "_hidden_params")
        assert video_response._hidden_params.get("response_cost") == expected_cost

        # Verify logging object was updated
        assert mock_logging_obj.model_call_details["response_cost"] == expected_cost
        assert mock_logging_obj.model_call_details["model"] == "veo-2.0-generate-001"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "gemini"
