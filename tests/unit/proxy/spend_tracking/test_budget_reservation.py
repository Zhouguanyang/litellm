from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import patch

import httpx
import pytest
import respx

import litellm
from litellm.caching import DualCache
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.proxy import proxy_server
from litellm.proxy._types import (
    LiteLLM_OrganizationTable,
    LiteLLM_ProjectTableCachedObj,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    project_cache_key,
    team_membership_reservation_cache_key,
)
from litellm.proxy.common_utils.timezone_utils import BudgetResetSettings
from litellm.proxy.spend_tracking.budget_reservation import (
    _get_team_member_budget_counter,
    _get_budget_counters,
    estimate_request_input_cost,
    estimate_request_max_cost,
    get_budget_window_start,
    reconcile_budget_reservation,
    release_unbound_budget_reservation,
    count_request_input_tokens,
    reserve_budget_for_request,
)
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore, user_window_counter


@pytest.mark.parametrize("token_rate", (None, 0.0, 0.001))
@pytest.mark.parametrize("request_price", (0.0, 0.02))
@pytest.mark.parametrize("price_source", ("model_info", "litellm_params", "both"))
@pytest.mark.parametrize(
    "route", ("/chat/completions", "/v1/images/generations", "/v1beta/models/priced:generateContent")
)
def test_configured_request_price_reservations_match_charges(
    local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch,
    token_rate: float | None, request_price: float, route: str, price_source: str,
) -> None:
    model: Final = "openai/unit-reserved-request-price"
    discount: Final = 0.25
    margin: Final = 0.5
    fixed_margin: Final = 0.003
    monkeypatch.setattr(litellm, "cost_discount_config", {model: discount})
    monkeypatch.setattr(litellm, "cost_margin_config", {"openai": {
        "percentage": margin, "fixed_amount": fixed_margin,
    }})
    litellm.register_model({model: {
        "litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }})
    router: Final = Router(model_list=[{
        "model_name": "priced", "litellm_params": {
            "model": model, "api_key": "unused",
            **({"output_cost_per_request": request_price} if price_source != "model_info" else {}),
        },
        "model_info": {
            "id": "reserved-request-price",
            **({"output_cost_per_request": request_price + (0.01 if price_source == "both" else 0.0)}
               if price_source != "litellm_params" else {}),
            **({"input_cost_per_token": token_rate, "output_cost_per_token": token_rate}
               if token_rate is not None else {}),
        },
    }])
    body: Final = {"model": "priced", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 20, "n": 3}
    expected: Final = request_price * (1 - discount) * (1 + margin) + fixed_margin
    assert litellm.completion_cost(
        completion_response=litellm.ModelResponse(
            model=model, usage=litellm.Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        ),
        model=model, custom_llm_provider="openai", custom_pricing=True, router_model_id="reserved-request-price",
    ) == pytest.approx(expected)
    assert estimate_request_max_cost(
        request_body=body, route=route, llm_router=router, input_token_counts={"priced": 10},
    ) == pytest.approx(expected)
    assert estimate_request_input_cost(
        request_body=body, route=route, llm_router=router, input_token_counts={"priced": 10},
    ) == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("price_source", ("model_info", "litellm_params"))
@pytest.mark.parametrize("route", (
    "/chat/completions", "/openai/v1/chat/completions", "/openai_passthrough/v1/chat/completions",
    "/vllm/v1/chat/completions", "/openai/v1/images/generations", "/anthropic/v1/messages",
    "/gemini/v1beta/models/priced:generateContent",
    "/gemini/v1alpha/models/priced:streamGenerateContent",
    "/vertex-ai/v1/projects/p/locations/us/publishers/google/models/priced:generateContent",
    "/vertex_ai/v1/projects/p/locations/us/publishers/anthropic/models/priced:rawPredict",
    "/vertex-ai/v1/projects/p/locations/us/publishers/anthropic/models/priced:streamRawPredict",
    "/bedrock/model/priced/converse", "/bedrock/model/priced/converse-stream",
    "/bedrock/model/priced/invoke", "/bedrock/model/priced/invoke-with-response-stream",
    "/cohere/v2/chat",
    "/azure/openai/deployments/priced/chat/completions?api-version=unit",
    "/azure/openai/responses", "/azure/openai/v1/responses",
    "/azure/openai/deployments/priced/responses", "/azure/priced/openai/responses",
    "/azure/priced/responses", "/azure_ai/openai/responses", "/azure_ai/priced/openai/v1/responses",
    "/azure_ai/models/chat/completions", "/azure_ai/priced/models/chat/completions",
))
async def test_flat_fee_reservations_block_concurrent_requests_and_reconcile_actual_cost(
    local_model_cost_map: None, spend_counter_store: SpendCounterStore, price_source: str, route: str,
) -> None:
    model: Final = "openai/unit-flat-fee-concurrency"
    price: Final = 0.02
    litellm.register_model({model: {
        "litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 0.0, "output_cost_per_token": 0.0,
    }})
    router: Final = Router(model_list=[{
        "model_name": "priced", "litellm_params": {
            "model": model, "api_key": "unused",
            **({"output_cost_per_request": price} if price_source == "litellm_params" else {}),
        },
        "model_info": {
            "id": "reserved-flat-concurrency",
            **({"output_cost_per_request": price} if price_source == "model_info" else {}),
        },
    }])
    key_cache: Final = UserApiKeyCache()
    token: Final = UserAPIKeyAuth(token=f"flat-fee-concurrency-{price_source}-{route}", max_budget=price, spend=0.0)
    requests: Final = await asyncio.gather(*(
        reserve_budget_for_request(
            request_body={"model": "priced", "messages": [{"role": "user", "content": "hello"}]},
            route=route, llm_router=router, valid_token=token,
            team_object=None, user_object=None, prisma_client=None, user_api_key_cache=key_cache,
            proxy_logging_obj=ProxyLogging(user_api_key_cache=key_cache),
        ) for _ in range(3)
    ), return_exceptions=True)
    reservations: Final = tuple(result for result in requests if isinstance(result, dict))
    rejected: Final = tuple(result for result in requests if isinstance(result, litellm.BudgetExceededError))
    assert len(reservations) == 1, requests
    assert len(rejected) == 2, requests
    assert reservations[0]["reserved_cost"] == pytest.approx(price)
    counter_key: Final = f"spend:key:{token.token}"
    assert spend_counter_store.get_local(counter_key) == pytest.approx(price)
    actual_fallback_cost: Final = price / 2
    await reconcile_budget_reservation(reservations[0], actual_cost=actual_fallback_cost)
    assert spend_counter_store.get_local(counter_key) == pytest.approx(actual_fallback_cost)


@pytest.mark.parametrize("mixed", (False, True))
def test_request_price_reservations_keep_the_most_expensive_deployment(
    local_model_cost_map: None, mixed: bool,
) -> None:
    model: Final = "openai/unit-mixed-reservations"
    input_rate: Final = 0.001
    output_rate: Final = 0.002
    litellm.register_model({model: {
        "litellm_provider": "openai", "mode": "chat",
        "input_cost_per_token": input_rate, "output_cost_per_token": output_rate,
    }})
    router: Final = Router(model_list=[{
        "model_name": "priced", "litellm_params": {"model": model, "api_key": "unused"},
        "model_info": {
            "id": f"mixed-reservation-{index}",
            **({"output_cost_per_request": price} if price is not None else {}),
        },
    } for index, price in enumerate((0.0, 0.02, None) if mixed else (0.0, 0.02))])
    body: Final = {"model": "priced", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 20}
    token_cost: Final = 10 * input_rate + 20 * output_rate
    assert estimate_request_max_cost(
        request_body=body, route="/chat/completions", llm_router=router, input_token_counts={"priced": 10},
    ) == pytest.approx(max(0.02, token_cost) if mixed else 0.02)


@pytest.mark.parametrize("route", (
    "/v1/embeddings", "/v1/realtime", "/openai/v1/embeddings", "/openai/v1/realtime",
    "/openai_passthrough/v1/embeddings", "/openai_passthrough/v1/realtime",
    "/openaifake/v1/chat/completions",
    "/azure/openai/embeddings", "/azure/openai/deployments/priced/embeddings",
    "/azure_ai/priced/openai/v1/embeddings", "/azure_ai/priced/models/embeddings",
))
@pytest.mark.parametrize("request_price", (0.0, 0.02))
def test_request_prices_do_not_replace_usage_reservations_on_other_routes(
    local_model_cost_map: None, route: str, request_price: float,
) -> None:
    model: Final = "openai/unit-usage-reservation"
    input_rate: Final = 0.001
    output_rate: Final = 0.002
    litellm.register_model({model: {
        "litellm_provider": "openai", "mode": "embedding" if route.endswith("/embeddings") else "chat",
        "input_cost_per_token": input_rate, "output_cost_per_token": output_rate,
    }})
    router: Final = Router(model_list=[{
        "model_name": "priced", "litellm_params": {"model": model, "api_key": "unused"},
        "model_info": {"id": "usage-reservation", "output_cost_per_request": request_price},
    }])
    body: Final = {"model": "priced", "input": "hello", "max_tokens": 20}
    assert estimate_request_max_cost(
        request_body=body, route=route, llm_router=router, input_token_counts={"priced": 10},
    ) == pytest.approx(10 * input_rate + (0.0 if route.endswith("/embeddings") else 20 * output_rate))
    assert estimate_request_input_cost(
        request_body=body, route=route, llm_router=router, input_token_counts={"priced": 10},
    ) == pytest.approx(10 * input_rate)


@pytest.mark.asyncio
@pytest.mark.parametrize("tiered", (False, True))
@pytest.mark.parametrize("price_source", ("model_info", "litellm_params"))
@pytest.mark.parametrize("price", (0.0, 0.02, 0.08))
async def test_mixed_group_reserves_only_costs_its_deployments_can_charge(
    local_model_cost_map: None, spend_counter_store: SpendCounterStore, tiered: bool, price_source: str, price: float,
) -> None:
    rates: Final = (("flat", 0.1, 0.2), ("tokens", 0.001, 0.002), ("input", 0.004, 0.0))
    for name, input_rate, output_rate in rates:
        litellm.register_model({f"openai/unit-mixed-{name}": {
            "litellm_provider": "openai", "mode": "chat",
            "input_cost_per_token": input_rate, "output_cost_per_token": output_rate,
        }})
    router: Final = Router(model_list=[{
        "model_name": "priced", "litellm_params": {
            "model": f"openai/unit-mixed-{name}", "api_key": "unused",
            **({"output_cost_per_request": price} if name == "flat" and price_source == "litellm_params" else {}),
        },
        "model_info": {
            "id": f"mixed-actual-price-{name}",
            **({"output_cost_per_request": price} if name == "flat" and price_source == "model_info" else {}),
            **({"tiered_pricing": [{
                "range": [0, 1000], "input_cost_per_token": input_rate, "output_cost_per_token": output_rate,
            }]} if tiered and name != "flat" else {}),
        },
    } for name, input_rate, output_rate in rates])
    body: Final = {"model": "priced", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 20}
    response: Final = litellm.ModelResponse(usage=litellm.Usage(
        prompt_tokens=10, completion_tokens=20, total_tokens=30,
    ))
    charges: Final = tuple(litellm.completion_cost(
        completion_response=response, model=f"openai/unit-mixed-{name}", custom_llm_provider="openai",
        custom_pricing=True, router_model_id=f"mixed-actual-price-{name}",
    ) for name, _, _ in rates)
    expected: Final = max(price, 10 * 0.001 + 20 * 0.002, 10 * 0.004)
    assert charges == pytest.approx((price, 0.05, 0.04))
    assert estimate_request_max_cost(
        request_body=body, route="/chat/completions", llm_router=router, input_token_counts={"priced": 10},
    ) == pytest.approx(expected)
    assert estimate_request_input_cost(
        request_body=body, route="/chat/completions", llm_router=router, input_token_counts={"priced": 10},
    ) == pytest.approx(10 * 0.004)
    cache: Final = UserApiKeyCache()
    reservation: Final = await reserve_budget_for_request(
        request_body=body, route="/chat/completions", llm_router=router,
        valid_token=UserAPIKeyAuth(token="mixed-actual-prices", max_budget=expected, spend=0.0),
        team_object=None, user_object=None, prisma_client=None, user_api_key_cache=cache,
        proxy_logging_obj=ProxyLogging(user_api_key_cache=cache), fail_closed_budget_enforcement=True,
    )
    assert reservation is not None
    assert reservation["reserved_cost"] <= expected
    assert spend_counter_store.get_local("spend:key:mixed-actual-prices") == pytest.approx(
        reservation["reserved_cost"],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("alias_style", ("string", "object", "chain"))
@pytest.mark.parametrize("pricing", ("flat", "token", "mixed"))
async def test_request_price_reservations_follow_the_alias_target_once(
    local_model_cost_map: None, spend_counter_store: SpendCounterStore, respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch, alias_style: str, pricing: str,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    model: Final = "openai/unit-budget-alias"
    price: Final = 0.02
    litellm.register_model({model: {
        "litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 0.001,
        "output_cost_per_token": 0.002,
    }})
    target_prices: Final = (price, None) if pricing == "mixed" else (None,) if pricing == "token" else (price,)
    router: Final = Router(model_list=[{
        "model_name": group, "litellm_params": {
            "model": model, "api_key": "unused", "api_base": "https://budget-alias.unit/v1",
        },
        "model_info": {
            "id": f"budget-alias-{index}", **({"output_cost_per_request": fee} if fee is not None else {}),
        },
    } for index, (group, fee) in enumerate((
        ("requested", 0.9), ("distant", 0.7), *(("served", fee) for fee in target_prices),
    ))], model_group_alias={
        "requested": {"model": "served", "hidden": False} if alias_style == "object" else "served",
        **({"served": "distant"} if alias_style == "chain" else {}),
    }, num_retries=0)
    body: Final = {"model": "requested", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 20}
    expected: Final = 0.05 if pricing != "flat" else price
    assert estimate_request_max_cost(
        request_body=body, route="/chat/completions", llm_router=router, input_token_counts={"requested": 10},
    ) == pytest.approx(expected)
    assert estimate_request_input_cost(
        request_body=body, route="/chat/completions", llm_router=router, input_token_counts={"requested": 10},
    ) == pytest.approx(0.01 if pricing != "flat" else 0.0)
    cache: Final = UserApiKeyCache()
    reservation: Final = await reserve_budget_for_request(
        request_body=body, route="/chat/completions", llm_router=router,
        valid_token=UserAPIKeyAuth(token="budget-alias", max_budget=expected, spend=0.0),
        team_object=None, user_object=None, prisma_client=None, user_api_key_cache=cache,
        proxy_logging_obj=ProxyLogging(user_api_key_cache=cache), fail_closed_budget_enforcement=True,
    )
    assert reservation is not None
    upstream: Final = respx_mock.post("https://budget-alias.unit/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "id": "chatcmpl_alias", "object": "chat.completion", "model": model, "created": 1,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }),
    )
    response: Final = await router.acompletion(model="requested", messages=body["messages"])
    assert isinstance(response, litellm.ModelResponse)
    charged: Final = litellm.completion_cost(
        completion_response=response, model=model, custom_llm_provider="openai", custom_pricing=True,
        router_model_id=response._hidden_params["model_id"],
    )
    assert upstream.call_count == 1
    allowed_costs: Final = (price, 0.05) if pricing == "mixed" else (expected,)
    assert any(charged == pytest.approx(cost) for cost in allowed_costs)
    assert charged <= expected


def test_request_price_reservation_keeps_explicit_provider_cost_adjustments(
    local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model: Final = "unit-explicit-provider-reservation"
    request_price: Final = 0.02
    margin: Final = 0.003
    monkeypatch.setattr(litellm, "cost_margin_config", {"openai": {"fixed_amount": margin}})
    router: Final = Router(model_list=[{
        "model_name": "priced",
        "litellm_params": {"model": model, "api_key": "unused", "custom_llm_provider": "openai"},
        "model_info": {"id": "explicit-provider-reservation", "output_cost_per_request": request_price},
    }])
    assert estimate_request_max_cost(
        request_body={"model": "priced", "messages": [{"role": "user", "content": "hello"}]},
        route="/chat/completions", llm_router=router,
    ) == pytest.approx(request_price + margin)

TOKEN_COUNTING_ROUTES: Final = (
    "/responses/input_tokens",
    "/v1/responses/input_tokens",
    "/openai/v1/responses/input_tokens",
    "/utils/token_counter",
    "/v1/messages/count_tokens",
    "/v1beta/models/gemini-3.8-flash:countTokens",
    "/models/gemini-3.8-flash:countTokens",
    "/bedrock/v1/messages/count-tokens",
    "/bedrock/model/us.anthropic.claude-sonnet-4-6/count-tokens",
    "/vertex_ai/v1/projects/p/locations/us-east5/publishers/anthropic/models/count-tokens:rawPredict",
    "/vertex-ai/v1/projects/p/locations/us-east5/publishers/anthropic/models/count-tokens:rawPredict",
)


def _budgeted_token() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test", token="hashed-token", max_budget=100.0, spend=0.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("team, apply, expected", [(False, False, 2), (True, False, 0), (True, True, 2)])
async def test_user_reservation_includes_each_window_only_when_user_budget_applies(
    team: bool, apply: bool, expected: int
) -> None:
    cache: Final = UserApiKeyCache()
    counters: Final = await _get_budget_counters(
        request_body={},
        valid_token=UserAPIKeyAuth(),
        team_object=LiteLLM_TeamTable(team_id="t") if team else None,
        user_object=LiteLLM_UserTable.model_validate(
            {
                "user_id": "u",
                "budget_limits": [
                    {"budget_duration": "1h", "max_budget": 5.0, "reset_at": "2026-08-01T01:00:00Z"},
                    {"budget_duration": "30d", "max_budget": 100.0, "reset_at": "2026-08-31T00:00:00Z"},
                ],
            }
        ),
        prisma_client=None,
        user_api_key_cache=cache,
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
        apply_user_budget_to_team_keys=apply,
    )
    assert len(counters) == expected
    assert {counter.counter_key for counter in counters} == (
        {user_window_counter("u", "1h"), user_window_counter("u", "30d")} if expected else set()
    )


async def _reserve(route: str) -> dict | None:
    return await reserve_budget_for_request(
        request_body={"model": "gpt-4o", "input": "hello"},
        route=route,
        llm_router=None,
        valid_token=_budgeted_token(),
        team_object=None,
        user_object=None,
        prisma_client=None,
        user_api_key_cache=UserApiKeyCache(),
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
    )


ANTHROPIC_MESSAGES: Final = [{"role": "user", "content": "hello!!!"}]
COUNT_TOKENS_REQUESTS: Final[tuple[tuple[str, dict[str, object]], ...]] = (
    ("/v1/messages/count_tokens", {"model": "claude-sonnet-5", "messages": ANTHROPIC_MESSAGES}),
    ("/v1beta/models/gemini-3.8-flash:countTokens", {"contents": [{"role": "user", "parts": [{"text": "hello!!!"}]}]}),
    (
        "/vertex_ai/v1/projects/p/locations/us-east5/publishers/anthropic/models/count-tokens:rawPredict",
        {"model": "claude-sonnet-5", "messages": ANTHROPIC_MESSAGES},
    ),
    ("/bedrock/v1/messages/count-tokens", {"model": "claude-sonnet-5", "messages": ANTHROPIC_MESSAGES}),
)
TINY_BUDGET_KEY_TOKEN: Final = "hashed-count-tokens-key"


@pytest.fixture
def spend_counter_store(monkeypatch: pytest.MonkeyPatch) -> SpendCounterStore:
    store: Final = SpendCounterStore(connect(None).spend_counters())
    monkeypatch.setattr(proxy_server, "spend_counter_store", store)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    return store


async def _reserve_for_tiny_budget_key(route: str, request_body: dict[str, object]) -> dict[str, object] | None:
    return await reserve_budget_for_request(
        request_body=request_body,
        route=route,
        llm_router=None,
        valid_token=UserAPIKeyAuth(token=TINY_BUDGET_KEY_TOKEN, max_budget=0.01, spend=0.0),
        team_object=None,
        user_object=None,
        prisma_client=None,
        user_api_key_cache=UserApiKeyCache(),
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
    )


BEDROCK_SONNET: Final = "us.anthropic.claude-sonnet-4-6"
CONVERSE_BODY: Final = {
    "messages": [{"role": "user", "content": [{"text": "Reply with one word: pong"}]}],
    "inferenceConfig": {"maxTokens": 5},
}
INVOKE_BODY: Final = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 5,
    "messages": [{"role": "user", "content": "Reply with one word: pong"}],
}


def test_bedrock_converse_body_reserves_the_prompt_not_the_context_window():
    converse_cost: Final = estimate_request_max_cost(
        request_body=CONVERSE_BODY,
        route=f"/bedrock/model/{BEDROCK_SONNET}/converse",
        llm_router=None,
        input_token_counts={},
    )
    invoke_cost: Final = estimate_request_max_cost(
        request_body=INVOKE_BODY,
        route=f"/bedrock/model/{BEDROCK_SONNET}/invoke",
        llm_router=None,
        input_token_counts={},
    )
    assert converse_cost is not None and invoke_cost is not None
    assert invoke_cost < converse_cost < 2 * invoke_cost


def _tiered_deployment(input_cost_per_token: float) -> Deployment:
    return Deployment(
        model_name="tiered-group",
        litellm_params=LiteLLM_Params(model="dashscope/qwen3-max", api_key="sk-fake"),
        model_info=ModelInfo(
            id="tiered-deployment",
            max_output_tokens=1000,
            tiered_pricing=[
                {
                    "input_cost_per_token": input_cost_per_token,
                    "output_cost_per_token": input_cost_per_token,
                    "range": [0, 128000],
                }
            ],
        ),
    )


TIERED_BODY: Final = {"model": "tiered-group", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 10}


def test_repeated_estimates_reuse_cached_model_cost_info() -> None:
    router: Final = Router(model_list=[_tiered_deployment(1e-06).model_dump()])
    first: Final = estimate_request_max_cost(request_body=TIERED_BODY, route="/chat/completions", llm_router=router)
    hits_before: Final = router.cached_deployment_model_info.cache_info().hits

    second: Final = estimate_request_max_cost(request_body=TIERED_BODY, route="/chat/completions", llm_router=router)

    assert second == first
    assert router.cached_deployment_model_info.cache_info().hits == hits_before + 1


def test_deployment_pricing_update_invalidates_cached_estimate() -> None:
    router: Final = Router(model_list=[_tiered_deployment(1e-06).model_dump()])
    before: Final = estimate_request_max_cost(request_body=TIERED_BODY, route="/chat/completions", llm_router=router)
    assert before is not None

    router.upsert_deployment(_tiered_deployment(1e-03))

    after: Final = estimate_request_max_cost(request_body=TIERED_BODY, route="/chat/completions", llm_router=router)
    assert after is not None
    assert math.isclose(after, before * 1000)


def _priced_group(model_name: str, cost_per_token: float) -> dict[str, object]:
    return Deployment(
        model_name=model_name,
        litellm_params=LiteLLM_Params(model="dashscope/qwen3-max", api_key="sk-fake"),
        model_info=ModelInfo(
            id=model_name,
            max_output_tokens=1000,
            tiered_pricing=[
                {"input_cost_per_token": cost_per_token, "output_cost_per_token": cost_per_token, "range": [0, 128000]}
            ],
        ),
    ).model_dump()


def test_google_route_reserves_on_the_model_its_path_param_selects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "user_model", None)
    router: Final = Router(model_list=[_priced_group("cheap-group", 1e-06), _priced_group("premium-group", 1e-03)])
    route: Final = "/v1beta/models/premium-group:generateContent"
    body: Final = {"model": "cheap-group", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 10}

    url_model_cost: Final = estimate_request_max_cost(request_body=body, route=route, llm_router=router)
    path_param_cost: Final = estimate_request_max_cost(
        request_body=body, route=route, llm_router=router, path_params={"model_name": "cheap-group"}
    )
    cheap_cost: Final = estimate_request_max_cost(
        request_body=body, route="/v1beta/models/cheap-group:generateContent", llm_router=router
    )

    assert url_model_cost is not None and cheap_cost is not None
    assert path_param_cost == cheap_cost
    assert math.isclose(url_model_cost, cheap_cost * 1000)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expiry_offset, expected_max_budget",
    [
        (timedelta(days=1), 3.0),
        (timedelta(days=-1), 2.0),
    ],
)
async def test_team_member_reservation_counter_honours_temp_budget_increase(
    expiry_offset: timedelta, expected_max_budget: float
) -> None:
    user_id: Final = "member-temp"
    team_id: Final = "team-temp"
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(
        key=team_membership_reservation_cache_key(user_id=user_id, team_id=team_id),
        value=LiteLLM_TeamMembership(
            user_id=user_id,
            team_id=team_id,
            spend=0.5,
            budget_id="budget-temp",
            litellm_budget_table=LiteLLM_BudgetTable(
                max_budget=2.0,
                temp_budget_increase=1.0,
                temp_budget_expiry=datetime.now(timezone.utc) + expiry_offset,
            ),
        ),
    )

    counter: Final = await _get_team_member_budget_counter(
        valid_token=UserAPIKeyAuth(token="hashed", user_id=user_id, team_id=team_id),
        team_object=LiteLLM_TeamTable(team_id=team_id),
        user_object=LiteLLM_UserTable(user_id=user_id),
        user_api_key_cache=cache,
    )

    assert counter is not None
    assert counter.max_budget == expected_max_budget
    assert counter.fallback_spend == 0.5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "default_cap, expiry_offset, expected_max_budget",
    [
        (2.0, timedelta(days=1), 3.0),
        (2.0, timedelta(days=-1), 2.0),
        (0.0, timedelta(days=1), None),
    ],
)
async def test_team_member_reservation_counter_adds_temp_increase_to_live_team_default(
    default_cap: float, expiry_offset: timedelta, expected_max_budget: float | None
) -> None:
    user_id: Final = "member-bare"
    team_id: Final = "team-bare"
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(
        key="team_member_default_budget:default-bare",
        value=LiteLLM_BudgetTable(budget_id="default-bare", max_budget=default_cap),
    )
    await cache.async_set_cache(
        key=team_membership_reservation_cache_key(user_id=user_id, team_id=team_id),
        value=LiteLLM_TeamMembership(
            user_id=user_id,
            team_id=team_id,
            spend=0.5,
            budget_id="budget-bare",
            litellm_budget_table=LiteLLM_BudgetTable(
                max_budget=None,
                temp_budget_increase=1.0,
                temp_budget_expiry=datetime.now(timezone.utc) + expiry_offset,
            ),
        ),
    )

    counter: Final = await _get_team_member_budget_counter(
        valid_token=UserAPIKeyAuth(token="hashed", user_id=user_id, team_id=team_id),
        team_object=LiteLLM_TeamTable(team_id=team_id, metadata={"team_member_budget_id": "default-bare"}),
        user_object=LiteLLM_UserTable(user_id=user_id),
        user_api_key_cache=cache,
    )

    if expected_max_budget is None:
        assert counter is None
        return
    assert counter is not None
    assert counter.max_budget == expected_max_budget
    assert counter.fallback_spend == 0.5


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        (
            '{"budget_duration": "1h", "reset_at": "2030-01-01T01:00:00Z"}',
            datetime(2030, 1, 1, 0, 0, tzinfo=timezone.utc),
        ),
        ('{"reset_at": "2030-01-01T01:00:00Z"}', None),
        ("{}", None),
        ('["budget_duration", "1h"]', None),
        ('"1h"', None),
        ("null", None),
        ("not json", None),
    ],
)
def test_budget_window_start_reads_json_encoded_windows(window: str, expected: datetime | None) -> None:
    assert get_budget_window_start(window) == expected


@pytest.mark.parametrize("today", ["2028-03-01", "2029-03-01"])
@pytest.mark.parametrize(
    ("duration", "reset_at", "expected"),
    [
        ("12mo", "2029-01-01T00:00:00Z", datetime(2028, 1, 1, tzinfo=timezone.utc)),
        ("12mo", "2028-01-01T00:00:00", datetime(2027, 1, 1, tzinfo=timezone.utc)),
    ],
)
def test_annual_window_keeps_all_spend_from_its_reset_boundary(
    today: str, duration: str, reset_at: str, expected: datetime
) -> None:
    with patch("time.time", return_value=datetime.fromisoformat(today).replace(tzinfo=timezone.utc).timestamp()):
        assert get_budget_window_start({"budget_duration": duration, "reset_at": reset_at}) == expected


@pytest.mark.parametrize("duration", ("1h", "1mo", "12mo"))
@pytest.mark.parametrize("reset_at", ("2026-11-15T05:48:02.123456Z", "2026-11-15T11:33:02.123456+05:45"))
def test_legacy_window_start_uses_database_millisecond_precision(duration: str, reset_at: str) -> None:
    window: Final = {"budget_duration": duration, "reset_at": reset_at}
    whole_second: Final = {"budget_duration": duration, "reset_at": "2026-11-15T05:48:02Z"}
    start: Final = get_budget_window_start(whole_second)
    assert start is not None
    assert get_budget_window_start(window) == start.replace(microsecond=123000)


class _AnnualWindowDatabase:
    def __init__(self) -> None:
        self.db = self
        self.litellm_budgetwindowspend = self
        self.litellm_spendlogs = self
        self.start: datetime | None = None
        self.spend = 0.0
        self.scans = 0

    async def execute_raw(self, sql: str, *parameters: object) -> int:
        start: Final = parameters[3]
        assert isinstance(start, datetime)
        self.start = start
        self.spend = 0.0
        return 1

    async def find_unique(self, **query: object) -> SimpleNamespace:
        return SimpleNamespace(window_start=self.start, spend=self.spend)

    async def group_by(self, **query: object) -> list[dict[str, object]]:
        self.scans += 1
        return [{"api_key": "calendar-key", "_sum": {"spend": 0.0}}]


@pytest.mark.asyncio
async def test_reservation_counts_a_pdf_with_the_deployment_behind_a_gpt_named_group() -> None:
    from litellm.proxy.spend_tracking.input_tokens import count_input_tokens_for_model
    from tests.unit.proxy.spend_tracking.test_input_tokens import _one_page_pdf_message

    request_body: Final = {"model": "gpt-4o", "messages": _one_page_pdf_message()}
    router: Final = Router(
        model_list=[{"model_name": "gpt-4o", "litellm_params": {"model": "anthropic/claude-haiku-5-5", "api_key": "x"}}]
    )

    counts: Final = await count_request_input_tokens(
        request_body=request_body, route="/chat/completions", llm_router=router, team_id=None
    )

    assert counts == {"gpt-4o": count_input_tokens_for_model(request_body=request_body, model="claude-haiku-5-5")}
    assert counts["gpt-4o"] > count_input_tokens_for_model(request_body=request_body, model="gpt-4o")


_MULTILINGUAL_MESSAGES: Final = [{"role": "user", "content": "日本語のテキストとहिन्दी पाठ और русский текст. " * 20}]


def _router_with_groups(*groups: tuple[str, str]) -> Router:
    return Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": model, "api_key": "sk-test"}} for name, model in groups
        ]
    )


async def _input_tokens(router: Router, model: str) -> dict[str, int]:
    return dict(
        await count_request_input_tokens(
            request_body={"model": model, "messages": _MULTILINGUAL_MESSAGES},
            route="/v1/chat/completions",
            llm_router=router,
            team_id=None,
        )
    )


async def _reserved_input_tokens(router: Router, model: str, route: str, messages: list[dict[str, object]]) -> int:
    counts: Final = await count_request_input_tokens(
        request_body={"model": model, "messages": messages}, route=route, llm_router=router, team_id=None
    )
    return counts[model]


@pytest.mark.asyncio
async def test_an_openai_chat_request_reserves_what_openai_chat_bills_on_top_of_each_pdf_files_pages() -> None:
    from litellm.constants import OPENAI_CHAT_PDF_FILE_TOKENS
    from tests.unit.proxy.spend_tracking.test_input_tokens import _one_page_pdf_message

    router: Final = _router_with_groups(("on-openai", "openai/gpt-5.6-sol"), ("on-azure", "azure/gpt-5.6-sol"))
    two_pdfs: Final = [*_one_page_pdf_message(), *_one_page_pdf_message()]

    assert await _reserved_input_tokens(router, "on-openai", "/v1/chat/completions", two_pdfs) == (
        await _reserved_input_tokens(router, "on-azure", "/v1/chat/completions", two_pdfs)
        + 2 * OPENAI_CHAT_PDF_FILE_TOKENS
    )
    assert await _reserved_input_tokens(router, "on-openai", "/v1/messages", two_pdfs) == (
        await _reserved_input_tokens(router, "on-azure", "/v1/messages", two_pdfs)
    )
    assert await _reserved_input_tokens(router, "on-openai", "/v1/chat/completions", _MULTILINGUAL_MESSAGES) == (
        await _reserved_input_tokens(router, "on-azure", "/v1/chat/completions", _MULTILINGUAL_MESSAGES)
    )


@pytest.mark.asyncio
async def test_alias_group_of_an_openai_model_reserves_on_that_models_token_count():
    router: Final = _router_with_groups(("gpt-6.1-sol", "openai/gpt-6.1-sol"), ("prod-gpt-6", "openai/gpt-6.1-sol"))

    by_name: Final = await _input_tokens(router, "gpt-6.1-sol")

    assert await _input_tokens(router, "prod-gpt-6") == {"prod-gpt-6": by_name["gpt-6.1-sol"]}


@pytest.mark.asyncio
async def test_model_group_alias_of_an_openai_group_reserves_on_that_groups_token_count():
    router: Final = Router(
        model_list=[
            {"model_name": "gpt-6.1-sol", "litellm_params": {"model": "openai/gpt-6.1-sol", "api_key": "sk-test"}}
        ],
        model_group_alias={"prod": "gpt-6.1-sol"},
    )

    by_name: Final = await _input_tokens(router, "gpt-6.1-sol")

    assert await _input_tokens(router, "prod") == {"prod": by_name["gpt-6.1-sol"]}


@pytest.mark.asyncio
async def test_model_group_alias_of_a_group_no_openai_model_serves_reserves_on_the_alias_as_written():
    router: Final = Router(
        model_list=[
            {
                "model_name": "claude-sonnet-5-5",
                "litellm_params": {"model": "anthropic/claude-sonnet-5-5", "api_key": "sk-test"},
            }
        ],
        model_group_alias={"claude-alias": "claude-sonnet-5-5"},
    )

    assert await _input_tokens(router, "claude-alias") == {
        "claude-alias": litellm.token_counter(model="claude-alias", messages=_MULTILINGUAL_MESSAGES)
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["gpt-6-claude", "o4-alias"])
async def test_release_named_group_or_alias_no_openai_model_serves_reserves_on_an_unknown_names_count(
    name: str,
) -> None:
    router: Final = Router(
        model_list=[
            {"model_name": group, "litellm_params": {"model": "anthropic/claude-sonnet-5-5", "api_key": "sk-test"}}
            for group in ("claude-sonnet-5-5", "gpt-6-claude")
        ],
        model_group_alias={"o4-alias": "claude-sonnet-5-5"},
    )

    assert await _input_tokens(router, name) == {
        name: litellm.token_counter(model="my-model", messages=_MULTILINGUAL_MESSAGES)
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("group", "counted_as"),
    [("gpt-6.1-sol", "my-model"), ("openai/gpt-6.1-sol", "my-model"), ("mixed", "my-model"), ("gpt-4o", "gpt-4o")],
)
async def test_group_mixing_in_a_non_openai_model_reserves_on_its_name_as_written_and_its_openai_models(
    group: str, counted_as: str
):
    served: Final = ("openai/gpt-6.1-sol", "cohere/command-r-plus")
    router: Final = _router_with_groups(*((group, model) for model in served))

    reserved: Final = (await _input_tokens(router, group))[group]

    assert reserved == max(
        litellm.token_counter(model=counted_as, messages=_MULTILINGUAL_MESSAGES),
        litellm.token_counter(model="openai/gpt-6.1-sol", messages=_MULTILINGUAL_MESSAGES),
    )


@pytest.mark.asyncio
async def test_group_reserves_on_its_current_deployments_after_another_vendors_deployment_joins():
    router: Final = _router_with_groups(("prod", "openai/gpt-6.1-sol"))
    openai_only: Final = (await _input_tokens(router, "prod"))["prod"]

    router.add_deployment(
        Deployment(
            model_name="prod",
            litellm_params=LiteLLM_Params(model="anthropic/claude-sonnet-5-5", api_key="sk-test"),
        )
    )

    by_own_name: Final = litellm.token_counter(model="prod", messages=_MULTILINGUAL_MESSAGES)
    assert openai_only != by_own_name
    assert (await _input_tokens(router, "prod"))["prod"] == by_own_name


@pytest.mark.parametrize("window", [{}, {"budget_duration": "1d"}, {"reset_at": "2026-11-03T00:00:00Z"}])
def test_fixed_budget_window_requires_duration_and_reset_boundary(window: object) -> None:
    from litellm.proxy.spend_tracking.budget_reservation import fixed_budget_window

    assert fixed_budget_window(window) is None
