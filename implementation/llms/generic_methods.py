import os
import time
import json
import re
import sys
import csv
import asyncio
import random
import hashlib
import logging
from functools import lru_cache

import gradio as gr
import httpx
from openai import OpenAI, AsyncOpenAI
import anthropic
from anthropic import AsyncAnthropic
from google import genai
from groq import AsyncGroq
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, field_validator, Field, RootModel
from enum import Enum
from typing import List, Optional, Dict, Tuple, Any
from datetime import date
from concurrent import futures
from tqdm import tqdm
from pathlib import Path
from openai.lib._pydantic import to_strict_json_schema
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from implementation.llms.pricing import compute_llm_cost_usd
from observability.cost_tracking import add_request_cost
from observability.names import (
    LLM_GENERATE,
    LLM_COST_USD,
    LLM_PROMPT_VERSION,
    LLM_ATTEMPT_COUNT,
)

# Load environment variables (for API key)
load_dotenv()


class LLMProvider(Enum):
    """Supported LLM provider backends for structured generation."""
    OPENAI = "openai"
    KIMI = "kimi"
    GEMINI = "gemini"
    GROQ = "groq"
    ALIBABA = "alibaba"
    ANTHROPIC = "anthropic"
    WHAM = "wham"  # ChatGPT WHAM backend (Codex models via OAuth)


# ===============================
#           Clients
# ===============================
#
# Clients are exposed as lazy proxies rather than constructed at import
# time. Eager construction forced every caller — including the API
# server at boot and unrelated test files — to have every provider's
# credentials in scope, even when only one provider was actually used.
#
# A simple module-level __getattr__ (PEP 562) is not sufficient because
# it only fires for *external* attribute access. Bare-name references
# inside this module's own functions (e.g. `gemini_client.aio.models...`
# in generate_gemini_response_async) compile to LOAD_GLOBAL, which
# consults the module __dict__ directly and bypasses __getattr__ — they
# would raise NameError. Using a proxy object lets bare-name lookups
# find a real module global while still deferring construction until
# first attribute access on the client itself.


class _LazyClient:
    """Proxy that defers construction of an LLM client until first
    attribute access.

    The proxy itself is cheap to build and is what gets bound to the
    module global, so LOAD_GLOBAL inside this module's functions
    always resolves. The wrapped factory runs the first time any
    attribute is read off the proxy (e.g. `.chat`, `.aio`,
    `.messages`); the resulting client is cached for the lifetime of
    the proxy and subsequent attribute access proxies straight
    through to it.
    """

    __slots__ = ("_factory", "_instance")

    def __init__(self, factory):
        self._factory = factory
        self._instance = None

    def __getattr__(self, name):
        # __getattr__ fires only when normal lookup misses; the two
        # slot attributes (`_factory`, `_instance`) are resolved via
        # the slot descriptors and never reach here, so every "real"
        # client attribute (e.g. .chat, .aio) routes through this
        # method.
        #
        # Thread-safety: the check-then-set below is not atomic. Safe
        # under the current single-event-loop deployment (no awaits
        # in the construction path, so no two coroutines can
        # interleave inside this method). If this code is ever
        # invoked from a thread pool or multi-threaded worker model,
        # wrap the check + factory call in a threading.Lock to avoid
        # double-construction races.
        instance = self._instance
        if instance is None:
            instance = self._factory()
            self._instance = instance
        return getattr(instance, name)

    def __repr__(self):
        # Show the underlying client type when constructed so the
        # proxy abstraction does not leak into stack traces / logs.
        # Does not force construction — keeps repr() side-effect-free.
        inst = self._instance
        inner = type(inst).__name__ if inst is not None else "(uninitialized)"
        return f"<_LazyClient {inner}>"


# Env vars are read inside each factory (not at module load), so values
# picked up by load_dotenv() above are always in scope by the time the
# factory runs.
openai_client = _LazyClient(lambda: OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
async_openai_client = _LazyClient(
    lambda: AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
)
kimi_client = _LazyClient(
    lambda: OpenAI(
        api_key=os.getenv("MOONSHOT_API_KEY"),
        base_url="https://api.moonshot.ai/v1",
    )
)
async_kimi_client = _LazyClient(
    lambda: AsyncOpenAI(
        api_key=os.getenv("MOONSHOT_API_KEY"),
        base_url="https://api.moonshot.ai/v1",
    )
)
# Gemini — uses Google's native genai SDK
gemini_client = _LazyClient(lambda: genai.Client(api_key=os.getenv("GOOGLE_API_KEY")))
# Groq — uses native Groq SDK (async only, matching project pattern)
async_groq_client = _LazyClient(lambda: AsyncGroq(api_key=os.getenv("GROQ_API_KEY")))
# Alibaba/Qwen — uses OpenAI-compatible routing via DashScope
async_alibaba_client = _LazyClient(
    lambda: AsyncOpenAI(
        api_key=os.getenv("ALIBABA_API_KEY"),
        base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    )
)
# Anthropic — uses OAuth token (ANTHROPIC_API_KEY) rather than API key;
# intended for reference generation and judge calls in the evaluation
# pipeline, but also available as a generation candidate.
async_anthropic_client = _LazyClient(
    lambda: AsyncAnthropic(auth_token=os.getenv("ANTHROPIC_API_KEY"))
)


# ===============================
#     Base Generation Methods
# ===============================
#
# The ASYNC provider functions below all return a 4-tuple
# `(parsed, input_tokens, output_tokens, cached_input_tokens)` — a uniform
# usage contract so the router (`generate_llm_response_async`) can unpack any
# dispatched provider identically and price the call. The two SYNC helpers
# (`generate_openai_response`, `generate_kimi_response`) deliberately stay on
# the older 3-tuple shape: they are an offline / ingestion utility path that
# does not flow through the router's cost telemetry, so their callers are left
# untouched.


def _extract_cached_tokens(usage: Any) -> int:
    """Best-effort cached-input-token count from a provider usage object.

    Cached tokens are the slice of the *input* the provider served from its
    prompt cache at a discounted rate — a SUBSET of the reported input/prompt
    tokens, never additional to them (OpenAI, Gemini, and the OpenAI-compatible
    providers all use this "cached ⊆ input" accounting). The field name differs
    per provider, so we probe the known locations and fall back to 0 (cache
    miss / provider without caching / field absent):

      - OpenAI Chat Completions & compatible (Kimi, Groq, Qwen):
        usage.prompt_tokens_details.cached_tokens
      - OpenAI Responses API (WHAM): usage.input_tokens_details.cached_tokens
      - Gemini: usage_metadata.cached_content_token_count

    Anthropic is deliberately NOT probed: it reports cache reads separately
    (`cache_read_input_tokens`) and does NOT fold them into `input_tokens`, so
    it breaks the "cached ⊆ input" assumption the cost formula relies on. It is
    also unpriced, so its cost is omitted regardless. Revisit if Anthropic ever
    joins the priced serving set.
    """
    if usage is None:
        return 0
    # OpenAI-family: cached_tokens nested under a *_tokens_details object,
    # named differently between the Chat Completions and Responses APIs.
    for details_attr in ("prompt_tokens_details", "input_tokens_details"):
        details = getattr(usage, details_attr, None)
        if details is not None:
            cached = getattr(details, "cached_tokens", None)
            if cached:
                return int(cached)
    # Gemini: flat field on usage_metadata.
    gemini_cached = getattr(usage, "cached_content_token_count", None)
    if gemini_cached:
        return int(gemini_cached)
    return 0


def _account_llm_call_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int,
) -> None:
    """Add this LLM attempt's USD cost to the active request-cost accumulator.

    Called by each async provider immediately after token usage is extracted
    and BEFORE the parse/validate that may raise — so an attempt the provider
    billed still counts toward the request total even when it fails and is
    retried (the request rollup is the sum over ALL billed attempts, per the
    /query_search cost-tracking requirement). Outside a tracked request (the
    offline ingestion / eval paths) `add_request_cost` is a no-op. An unpriced
    model yields `None` here, which the accumulator ignores.
    """
    add_request_cost(
        compute_llm_cost_usd(model, input_tokens, output_tokens, cached_input_tokens)
    )


def generate_openai_response(
    user_prompt: str,
    system_prompt: str,
    response_format: BaseModel,
    model: str = "gpt-5-mini",
    reasoning_effort: str = "low",
    verbosity: str = "low",
    **kwargs,
) -> Tuple[BaseModel, int, int]:
    """
    Returns a tuple of (parsed_response, input_tokens, output_tokens).

    Additional OpenAI-specific params (max_completion_tokens, etc.)
    can be passed via kwargs.
    """
    try:
        response = openai_client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            **kwargs,
        )

        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

        # Extract the parsed response - OpenAI automatically validates structure matches response_format
        parsed = response.choices[0].message.parsed
        return parsed, input_tokens, output_tokens
    except Exception as e:
        raise ValueError(f"OpenAI failed to generate response: {e}")

async def generate_openai_response_async(
    user_prompt: str,
    system_prompt: str,
    response_format: BaseModel,
    model: str = "gpt-5-mini",
    reasoning_effort: str = "low",
    verbosity: str = "low",
    **kwargs,
) -> Tuple[BaseModel, int, int, int]:
    """Async counterpart to generate_openai_response.

    Uses async_openai_client.chat.completions.parse() with the same
    parameters as the sync version.

    Additional OpenAI-specific params (max_completion_tokens, etc.)
    can be passed via kwargs.

    Returns (parsed_response, input_tokens, output_tokens, cached_input_tokens).
    """
    try:
        response = await async_openai_client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            **kwargs,
        )

        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        cached_input_tokens = _extract_cached_tokens(usage)
        # Account the billed attempt into the request-cost rollup before the
        # parse below (so a billed-but-failed attempt still counts).
        _account_llm_call_cost(model, input_tokens, output_tokens, cached_input_tokens)

        # Extract the parsed response - OpenAI automatically validates structure matches response_format
        parsed = response.choices[0].message.parsed
        return parsed, input_tokens, output_tokens, cached_input_tokens
    except Exception as e:
        print(f"OpenAI async failed to generate response: {e}")
        raise ValueError(f"OpenAI async failed to generate response: {e}")


async def generate_kimi_response_async(
    user_prompt: str,
    system_prompt: str,
    response_format: BaseModel,
    enable_thinking: bool = False,
) -> Tuple[BaseModel, int, int, int]:
    """Generate a structured response using the Kimi (Moonshot) API.

    Returns (parsed_response, input_tokens, output_tokens, cached_input_tokens).
    """
    try:
        thinking_type = "enabled" if enable_thinking else "disabled"
        schema = to_strict_json_schema(response_format)

        response = await async_kimi_client.chat.completions.create(
            model="kimi-k2.5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1.0 if enable_thinking else 0.6,
            top_p=0.95,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "strict": True,
                    "schema": schema,
                },
            },
            extra_body={
                "thinking": {"type": thinking_type}
            }
        )

        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        cached_input_tokens = _extract_cached_tokens(usage)
        # Account the billed attempt before validation below, so a billed-but-
        # failed (unparseable / schema-mismatch) attempt still counts. Kimi
        # hardcodes its model, so pass the literal (currently unpriced → no-op).
        _account_llm_call_cost("kimi-k2.5", input_tokens, output_tokens, cached_input_tokens)

        # Extract the parsed response and enforce schema structure
        raw = response.choices[0].message.content
        data = json.loads(raw)
        metadata = response_format.model_validate(data)

        return metadata, input_tokens, output_tokens, cached_input_tokens
    except Exception as e:
        raise ValueError(f"Kimi failed to generate response: {e}")

def generate_kimi_response(
    user_prompt: str,
    system_prompt: str,
    response_format: BaseModel,
    enable_thinking: bool = False,
) -> Tuple[BaseModel, int, int]:
    """Generate a structured response using the Kimi (Moonshot) API (sync).

    Returns a tuple of (parsed_response, input_tokens, output_tokens).
    """
    try:
        thinking_type = "enabled" if enable_thinking else "disabled"
        schema = to_strict_json_schema(response_format)

        response = kimi_client.chat.completions.create(
            model="kimi-k2.5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1.0 if enable_thinking else 0.6,
            top_p=0.95,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "strict": True,
                    "schema": schema,
                },
            },
            extra_body={
                "thinking": {"type": thinking_type}
            }
        )

        # Extract the parsed response and enforce schema structure
        raw = response.choices[0].message.content
        data = json.loads(raw)
        metadata = response_format.model_validate(data)

        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

        return metadata, input_tokens, output_tokens
    except Exception as e:
        raise ValueError(f"Kimi failed to generate response: {e}")


async def generate_gemini_response_async(
    user_prompt: str,
    system_prompt: str,
    response_format: BaseModel,
    model: str = "gemini-2.5-flash",
    **kwargs,
) -> Tuple[BaseModel, int, int, int]:
    """Generate a structured response using Google's Gemini API.

    Uses the native google-genai SDK with JSON schema structured output.
    Additional Gemini-specific params (temperature, top_p, top_k,
    max_output_tokens, etc.) can be passed via kwargs.

    Returns (parsed_response, input_tokens, output_tokens, cached_input_tokens).
    """
    try:
        # Build the generation config: caller kwargs first, then our
        # required keys override to prevent accidental clobbering of
        # structured output or system instruction settings.
        config = {
            **kwargs,
            "response_mime_type": "application/json",
            "response_json_schema": response_format.model_json_schema(),
            "system_instruction": system_prompt,
        }

        response = await gemini_client.aio.models.generate_content(
            model=model,
            contents=user_prompt,
            config=config,
        )

        # Extract token usage from Gemini's usage metadata
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count
        # Implicit cache hits — Gemini 2.5+ caches automatically when the request
        # shares a prefix with a recent call (min 1,024 tokens). The count is a
        # subset of prompt tokens, billed at the discounted cached rate.
        cached_input_tokens = _extract_cached_tokens(usage)
        # Account the billed attempt before the parse below, so a billed-but-
        # failed (unparseable) attempt still counts.
        _account_llm_call_cost(model, input_tokens, output_tokens, cached_input_tokens)

        # Parse the JSON response into the Pydantic model
        parsed = response_format.model_validate_json(response.text)

        return parsed, input_tokens, output_tokens, cached_input_tokens
    except Exception as e:
        raise ValueError(f"Gemini async failed to generate response: {e}")


async def generate_groq_response_async(
    user_prompt: str,
    system_prompt: str,
    response_format: BaseModel,
    model: str = "llama-3.3-70b-versatile",
    **kwargs,
) -> Tuple[BaseModel, int, int, int]:
    """Generate a structured response using Groq's native API.

    Uses the Groq SDK with json_schema response format (same pattern as Kimi).
    Additional Groq-specific params (temperature, top_p, max_completion_tokens,
    etc.) can be passed via kwargs.

    Returns (parsed_response, input_tokens, output_tokens, cached_input_tokens).
    """
    try:
        schema = to_strict_json_schema(response_format)

        response = await async_groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "strict": False,
                    "schema": schema,
                },
            },
            **kwargs,
        )

        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        cached_input_tokens = _extract_cached_tokens(usage)
        # Account the billed attempt before validation below, so a billed-but-
        # failed attempt still counts.
        _account_llm_call_cost(model, input_tokens, output_tokens, cached_input_tokens)

        # Parse the JSON string into the Pydantic model (same approach as Kimi)
        raw = response.choices[0].message.content
        data = json.loads(raw)
        parsed = response_format.model_validate(data)

        return parsed, input_tokens, output_tokens, cached_input_tokens
    except Exception as e:
        raise ValueError(f"Groq async failed to generate response: {e}")


async def generate_alibaba_response_async(
    user_prompt: str,
    system_prompt: str,
    response_format: BaseModel,
    model: str = "qwen-plus",
    **kwargs,
) -> Tuple[BaseModel, int, int, int]:
    """Generate a structured response using Alibaba's Qwen API via OpenAI-compatible routing.

    Uses AsyncOpenAI.chat.completions.parse() pointed at DashScope's
    compatible endpoint. Additional params (temperature, top_p, etc.)
    can be passed via kwargs.

    Returns (parsed_response, input_tokens, output_tokens, cached_input_tokens).
    """
    try:
        response = await async_alibaba_client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_format,
            **kwargs,
        )

        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        cached_input_tokens = _extract_cached_tokens(usage)
        # Account the billed attempt before reading the parsed output below.
        _account_llm_call_cost(model, input_tokens, output_tokens, cached_input_tokens)

        parsed = response.choices[0].message.parsed
        return parsed, input_tokens, output_tokens, cached_input_tokens
    except Exception as e:
        raise ValueError(f"Alibaba/Qwen async failed to generate response: {e}")


async def generate_anthropic_response_async(
    user_prompt: str,
    system_prompt: str,
    response_format: BaseModel,
    model: str = "claude-opus-4-6",
    **kwargs,
) -> Tuple[BaseModel, int, int, int]:
    """Generate a structured response using the Anthropic API via OAuth token.

    Uses tool use to force structured output: the response_format Pydantic
    model is registered as a single tool, and tool_choice forces the model
    to call it. This is the standard structured output approach for Claude.

    max_tokens defaults to 4096 if not provided — it is required by the
    Anthropic API but optional in the unified interface.

    When cache_control=True, the system prompt, user message, and tool
    schema are wrapped in content blocks with cache_control breakpoints
    for Anthropic's prompt caching (90% discount on cached token reads).
    Callers should stagger repeated calls so the first response populates
    the cache before subsequent calls fire.

    Additional Anthropic-specific params (temperature, top_p, etc.) can be
    passed via kwargs.

    Returns (parsed_response, input_tokens, output_tokens, cached_input_tokens);
    cached_input_tokens is always 0 here (see the return site).
    """
    try:
        # max_tokens is required by the Anthropic API; default if not specified
        max_tokens = kwargs.pop("max_tokens", 4096)

        # Extended thinking: when budget_tokens is provided, enable Anthropic's
        # extended thinking mode. max_tokens must cover both thinking tokens and
        # the structured output, so we expand it to budget_tokens + 4096.
        # Temperature must not be set when thinking is enabled (API enforces 1.0).
        budget_tokens = kwargs.pop("budget_tokens", None)
        if budget_tokens is not None:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
            max_tokens = budget_tokens + 4096

        # Prompt caching: when enabled, wrap system/user/tool content in
        # cache_control blocks so repeated calls with the same prefix get
        # 90% input token discount. Popped from kwargs to prevent leaking
        # to the Anthropic API.
        enable_cache = kwargs.pop("cache_control", False)

        # Build system parameter — content block list when caching, plain string otherwise
        if enable_cache:
            system_param = [{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }]
        else:
            system_param = system_prompt

        # Build messages — content block list when caching, plain string otherwise
        if enable_cache:
            messages_param = [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": user_prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
            }]
        else:
            messages_param = [{"role": "user", "content": user_prompt}]

        # Build tool definition — add cache_control when caching is enabled
        tool_def = {
            "name": "structured_output",
            "description": "Submit the structured output.",
            "input_schema": response_format.model_json_schema(),
        }
        if enable_cache:
            tool_def["cache_control"] = {"type": "ephemeral"}

        # Register the response schema as a tool and force the model to call it.
        # tool_choice={"type": "tool"} guarantees the output matches the schema.
        response = await async_anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_param,
            messages=messages_param,
            tools=[tool_def],
            tool_choice={"type": "tool", "name": "structured_output"},
            **kwargs,
        )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        # Anthropic reports cache reads separately (cache_read_input_tokens) and
        # does NOT fold them into input_tokens, so _extract_cached_tokens returns
        # 0 here — its accounting breaks the cached ⊆ input cost model, and
        # Anthropic is unpriced anyway. Revisit if it joins the priced set.
        cached_input_tokens = _extract_cached_tokens(response.usage)
        # Account the billed attempt before the tool-use extraction/validation
        # below (Anthropic is currently unpriced, so this is a no-op today).
        _account_llm_call_cost(model, input_tokens, output_tokens, cached_input_tokens)

        # Extract the tool_use block — guaranteed present when tool_choice forces it
        tool_use_block = next(
            block for block in response.content if block.type == "tool_use"
        )
        parsed = response_format.model_validate(tool_use_block.input)

        return parsed, input_tokens, output_tokens, cached_input_tokens
    except anthropic.RateLimitError:
        raise  # Let rate limits propagate for caller-side retry
    except Exception as e:
        raise ValueError(f"Anthropic async failed to generate response: {e}")


# ===============================
#     WHAM (ChatGPT Backend)
# ===============================

# WHAM base URL — ChatGPT's internal backend for Codex models (gpt-5.4, etc.)
# The SDK appends /responses to this, so the final endpoint becomes
# chatgpt.com/backend-api/codex/responses (the actual WHAM Responses API path).
WHAM_BASE_URL = "https://chatgpt.com/backend-api/codex"


async def generate_wham_response_async(
    user_prompt: str,
    system_prompt: str,
    response_format: BaseModel,
    model: str = "gpt-5.4",
    api_key: Optional[str] = None,
    account_id: Optional[str] = None,
    **kwargs,
) -> Tuple[BaseModel, int, int, int]:
    """Generate a structured response via ChatGPT's WHAM backend.

    Uses the OpenAI Responses API (responses.parse) against the WHAM endpoint,
    which is accessed via ChatGPT OAuth tokens rather than standard API keys.

    WHAM-specific requirements:
      - base_url must be chatgpt.com/backend-api/codex
      - ChatGPT-Account-Id header is required
      - store=False is mandatory
      - User content type must be "input_text" (not "text")
      - System prompt goes in the 'instructions' parameter

    Args:
        api_key: OAuth access_token from the ChatGPT PKCE flow.
        account_id: ChatGPT account ID extracted from the JWT claims.

    Returns (parsed_response, input_tokens, output_tokens, cached_input_tokens).
    """
    if not api_key or not account_id:
        raise ValueError(
            "WHAM provider requires api_key and account_id from ChatGPT OAuth. "
            "Ensure get_valid_auth() is called before making WHAM requests."
        )

    # Create a per-call client scoped to the WHAM endpoint with the
    # account ID header. A new client per call avoids stale-token issues
    # since OAuth tokens are short-lived.
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=WHAM_BASE_URL,
        default_headers={"ChatGPT-Account-Id": account_id},
    )

    # WHAM does not support max_tokens/max_output_tokens at all, and rejects
    # temperature when reasoning_effort != "none". Pop them from kwargs so
    # they don't leak into the API call if a caller passes them generically.
    kwargs.pop("max_tokens", None)
    kwargs.pop("max_output_tokens", None)
    kwargs.pop("temperature", None)

    verbosity = kwargs.pop("verbosity", None)

    # Responses API uses a nested reasoning object: {"effort": "low"|"medium"|"high"}
    # Accept reasoning_effort as a flat kwarg for caller convenience.
    reasoning_effort = kwargs.pop("reasoning_effort", None)

    # Build optional params dict — only include non-None values
    optional_params = {}
    if verbosity is not None:
        optional_params["verbosity"] = verbosity
    if reasoning_effort is not None:
        optional_params["reasoning"] = {"effort": reasoning_effort}

    try:
        # WHAM requires stream=True for all requests. responses.parse() does
        # not support streaming, so we use responses.stream() with text_format
        # which gives us both mandatory streaming AND automatic Pydantic parsing.
        async with client.responses.stream(
            model=model,
            instructions=system_prompt,
            input=[{
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            }],
            text_format=response_format,
            store=False,
            **optional_params,
        ) as stream:
            response = await stream.get_final_response()

        usage = response.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0
        cached_input_tokens = _extract_cached_tokens(usage)
        # Account the billed attempt before the refusal check below, so a
        # billed-but-refused response still counts toward the request total.
        _account_llm_call_cost(model, input_tokens, output_tokens, cached_input_tokens)

        parsed = response.output_parsed
        if parsed is None:
            raise ValueError(
                "WHAM response did not contain parsed output. "
                "The model may have refused or returned an unexpected format."
            )

        return parsed, input_tokens, output_tokens, cached_input_tokens
    except Exception as e:
        raise ValueError(f"WHAM async failed to generate response: {e}")


# ===============================
#     Unified Routing Method
# ===============================

# Maps each provider to its async generation function.
# Kimi ignores the model param (hardcoded internally), so we
# strip it before forwarding.
_PROVIDER_DISPATCH = {
    LLMProvider.OPENAI: generate_openai_response_async,
    LLMProvider.KIMI: generate_kimi_response_async,
    LLMProvider.GEMINI: generate_gemini_response_async,
    LLMProvider.GROQ: generate_groq_response_async,
    LLMProvider.ALIBABA: generate_alibaba_response_async,
    LLMProvider.ANTHROPIC: generate_anthropic_response_async,
    LLMProvider.WHAM: generate_wham_response_async,
}

# Providers whose async function does not accept a `model` parameter
_PROVIDERS_WITHOUT_MODEL_PARAM = {LLMProvider.KIMI}


# Per-attempt timeout for every routed LLM call. Lives here at the
# lowest layer (the actual SDK invocation) rather than being nested
# at orchestrator + handler levels. One retry total — a second
# failure re-raises so the caller decides whether to soft-fail.
LLM_PER_ATTEMPT_TIMEOUT_SECONDS = 25.0
LLM_MAX_ATTEMPTS = 2
# Backoff window between attempts. Jitter avoids the "retry storm"
# pattern where many concurrent calls all wake up at the same
# wall-clock moment after a transient 5xx / rate-limit.
_LLM_RETRY_BACKOFF_MIN_SECONDS = 0.05
_LLM_RETRY_BACKOFF_MAX_SECONDS = 0.25

_llm_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM-call telemetry (Bite 1 — the single instrumentation point for every
# routed LLM call). See observability_context/query_search_planning.md §2.2–2.4
# and §2.8 for the decisions this implements.
# ---------------------------------------------------------------------------

# Module tracer, mirroring the api/main.py convention. When setup_tracing has
# not run (e.g. an offline ingestion/eval process that imports this module),
# get_tracer returns a no-op tracer and every span below is a cheap no-op.
_tracer = trace.get_tracer(__name__)

# Standard OTel GenAI semantic-convention attribute keys + the standard
# `error.type`. These are spec-owned strings, deliberately NOT authored in
# observability/names.py (which never re-spells a standard root). Named here as
# constants purely for typo-safety.
_GEN_AI_SYSTEM = "gen_ai.system"                      # provider (e.g. "openai")
_GEN_AI_REQUEST_MODEL = "gen_ai.request.model"        # requested model
_GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
_GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
# Cache reads (tokens served from the provider's prompt cache — a subset of
# input_tokens, billed at a discount). The GenAI semconv added this key, so we
# emit the spec string rather than a project-authored `llm.*` name.
_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
_ERROR_TYPE = "error.type"                            # normalized failure class

# Span-event messages (human-readable, per the names.py scope note — not Names).
_LLM_RETRY_EVENT = "llm.retry"      # one per failed-but-retried attempt
_LLM_PAYLOAD_EVENT = "llm.payload"  # full prompt + response, sample-gated

# Model-performance payload capture (decision §2.4). A float sample rate read
# once at import: >= 1.0 captures every successful call (the default — "capture
# 100% now"), 0.0 disables entirely, else Bernoulli(rate) per successful call.
# When enabled (> 0), a terminal failure captures the prompt unconditionally
# (always-on-error), so the prompt that broke is never sampled away. Changing
# the rate requires a restart (same discipline as the OTEL_* vars).
_PAYLOAD_SAMPLE_RATE = float(os.getenv("LLM_PAYLOAD_CAPTURE_SAMPLE_RATE", "1.0"))


@lru_cache(maxsize=256)
def _prompt_version(system_prompt: str) -> str:
    """Short, stable content hash of a system prompt.

    Changes exactly when the prompt text changes, letting evals slice by prompt
    revision. Cached so identical prompts (the handful of module-level prompt
    constants in this app) hash once, matching decision §2.3's "computed once".
    """
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:12]


def _is_timeout(exc: BaseException) -> bool:
    """True when the failure is the per-attempt `asyncio.wait_for` ceiling.

    On Python 3.11+ `asyncio.TimeoutError` is an alias of builtin `TimeoutError`;
    both are checked for explicitness.
    """
    return isinstance(exc, (asyncio.TimeoutError, TimeoutError))


def _error_type(exc: BaseException) -> str:
    """Normalized `error.type` value — collapse the timeout ceiling to a single
    stable token, everything else to its exception class name."""
    return "timeout" if _is_timeout(exc) else type(exc).__name__


def _should_capture_payload() -> bool:
    """Per-successful-call sampling decision for the payload event."""
    if _PAYLOAD_SAMPLE_RATE >= 1.0:
        return True
    if _PAYLOAD_SAMPLE_RATE <= 0.0:
        return False
    return random.random() < _PAYLOAD_SAMPLE_RATE


def _emit_payload_event(
    span,
    system_prompt: str,
    user_prompt: str,
    response_json: Optional[str],
) -> None:
    """Attach the full resolved prompt (+ response, when present) as a span
    event. Response is omitted on the failure path (there is none)."""
    attributes = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    if response_json is not None:
        attributes["response"] = response_json
    span.add_event(_LLM_PAYLOAD_EVENT, attributes)


async def generate_llm_response_async(
    provider: LLMProvider,
    user_prompt: str,
    system_prompt: str,
    response_format: BaseModel,
    model: str,
    timeout: float | None = None,
    **kwargs,
) -> Tuple[BaseModel, int, int]:
    """Route a structured-output request to the appropriate provider,
    with timeout + jittered single retry applied at this layer.

    Accepts provider-agnostic params (prompts, response_format, model) plus
    any provider-specific kwargs (e.g. reasoning_effort for OpenAI,
    enable_thinking for Kimi, temperature for Gemini/Groq/Alibaba).

    The timeout / retry / jitter discipline lives here (the lowest
    layer) rather than being duplicated at orchestrator + handler
    levels. A pathological hang therefore burns at most one
    `per_attempt_timeout * LLM_MAX_ATTEMPTS` window instead of
    compounding across stacked retry wrappers.

    Callers that run heavier prompts (e.g. reasoning-model generators
    like concept_tags) can pass an explicit `timeout` to override the
    default `LLM_PER_ATTEMPT_TIMEOUT_SECONDS`. When None, the module
    default is used.

    Telemetry: this is the single instrumentation point for every routed LLM
    call. One `llm.generate` span wraps the whole retry loop, carrying provider
    / model / token usage (`gen_ai.*`), cache-read input tokens
    (`gen_ai.usage.cache_read.input_tokens`), cache-adjusted `llm.cost_usd`,
    `llm.prompt_version`, `llm.attempt_count`, and failure marking per
    query_search_planning.md §2.8 (recovered retries stay UNSET with `llm.retry`
    events; only an exhausted retry marks the span ERROR). Full prompt/response
    ride sample-gated `llm.payload` events. Each billed attempt (including
    retried/failed ones) also self-accounts its cost into the request-level
    rollup (query_search.cost_usd) via _account_llm_call_cost.

    Returns a tuple of (parsed_response, input_tokens, output_tokens).
    """
    generate_fn = _PROVIDER_DISPATCH[provider]
    pass_model = provider not in _PROVIDERS_WITHOUT_MODEL_PARAM
    per_attempt_timeout = (
        timeout if timeout is not None else LLM_PER_ATTEMPT_TIMEOUT_SECONDS
    )

    # One span wraps the WHOLE retry loop (decision §2.2): step identity comes
    # from the parent span nesting, never duplicated here. Auto error/status is
    # disabled so we mark it by hand — a recovered retry must stay UNSET, and
    # only a true exhaustion reads ERROR (decision §2.8).
    with _tracer.start_as_current_span(
        LLM_GENERATE, record_exception=False, set_status_on_exception=False
    ) as span:
        span.set_attribute(_GEN_AI_SYSTEM, provider.value)
        span.set_attribute(_GEN_AI_REQUEST_MODEL, model)
        span.set_attribute(LLM_PROMPT_VERSION, _prompt_version(system_prompt))
        # Sample decision for successful-call payloads is taken once up front so
        # it can't vary across attempts of the same logical call.
        capture_payload = _should_capture_payload()

        last_exc: BaseException | None = None
        for attempt in range(LLM_MAX_ATTEMPTS):
            try:
                if pass_model:
                    parsed, input_tokens, output_tokens, cached_input_tokens = await asyncio.wait_for(
                        generate_fn(
                            user_prompt=user_prompt,
                            system_prompt=system_prompt,
                            response_format=response_format,
                            model=model,
                            **kwargs,
                        ),
                        timeout=per_attempt_timeout,
                    )
                else:
                    parsed, input_tokens, output_tokens, cached_input_tokens = await asyncio.wait_for(
                        generate_fn(
                            user_prompt=user_prompt,
                            system_prompt=system_prompt,
                            response_format=response_format,
                            **kwargs,
                        ),
                        timeout=per_attempt_timeout,
                    )
            except Exception as exc:  # noqa: BLE001 — broad catch is intentional
                last_exc = exc
                if attempt < LLM_MAX_ATTEMPTS - 1:
                    # Short jittered backoff — long enough to dodge a
                    # rate-limit window, short enough that a recovered
                    # call still feels snappy.
                    backoff = random.uniform(
                        _LLM_RETRY_BACKOFF_MIN_SECONDS,
                        _LLM_RETRY_BACKOFF_MAX_SECONDS,
                    )
                    # Transient failure that will be retried: record it as an
                    # event but leave the span status UNSET — it may still win.
                    span.add_event(
                        _LLM_RETRY_EVENT,
                        {
                            "attempt": attempt + 1,
                            _ERROR_TYPE: _error_type(exc),
                            "timeout": _is_timeout(exc),
                            "backoff_seconds": backoff,
                        },
                    )
                    _llm_logger.warning(
                        "LLM call %s/%s failed on attempt %d; retrying after %.2fs (%r)",
                        provider.value, model, attempt + 1, backoff, exc,
                    )
                    await asyncio.sleep(backoff)
                    continue
                _llm_logger.error(
                    "LLM call %s/%s failed on final attempt %d (%r)",
                    provider.value, model, attempt + 1, exc,
                )
                break
            else:
                # Success (possibly after retries). The span stays UNSET; the
                # attempt count is what separates a clean call from a recovered
                # one (decision §2.8).
                span.set_attribute(LLM_ATTEMPT_COUNT, attempt + 1)
                span.set_attribute(_GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
                span.set_attribute(_GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
                # Cache-read (discounted) input tokens — a subset of
                # input_tokens. Recorded on every span (even 0) so cache-hit
                # rate is queryable in aggregate.
                span.set_attribute(
                    _GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, cached_input_tokens
                )
                # This span's `llm.cost_usd` reflects the SUCCESSFUL attempt.
                # The request-level rollup (query_search.cost_usd) is the
                # superset: each provider self-accounts every billed attempt —
                # including ones that then failed and were retried — into the
                # request accumulator (see _account_llm_call_cost). The router
                # deliberately does NOT add here, to avoid double-counting the
                # successful attempt the provider already accounted.
                cost_usd = compute_llm_cost_usd(
                    model, input_tokens, output_tokens, cached_input_tokens
                )
                if cost_usd is not None:
                    span.set_attribute(LLM_COST_USD, cost_usd)
                else:
                    # Never fabricate a cost — surface the missing pricing entry.
                    _llm_logger.warning(
                        "No pricing for model %r; llm.cost_usd omitted", model,
                    )
                if capture_payload:
                    _emit_payload_event(
                        span, system_prompt, user_prompt, parsed.model_dump_json()
                    )
                return parsed, input_tokens, output_tokens

        # Reached only by `break` on the final attempt: retries exhausted, the
        # call genuinely failed. Mark ERROR + normalized error.type + record the
        # exception, attempt_count at the ceiling. Always-on-error prompt capture
        # whenever payload capture is enabled, so a failing prompt is never
        # sampled away.
        assert last_exc is not None
        span.set_attribute(LLM_ATTEMPT_COUNT, LLM_MAX_ATTEMPTS)
        span.set_attribute(_ERROR_TYPE, _error_type(last_exc))
        span.set_status(Status(StatusCode.ERROR))
        span.record_exception(last_exc)
        if _PAYLOAD_SAMPLE_RATE > 0.0:
            _emit_payload_event(span, system_prompt, user_prompt, None)
        raise last_exc


# Embedding client uses a dedicated httpx pool with explicit Limits so
# keepalive connections are reused across requests rather than paying a
# TCP + TLS handshake per call. (The prior fresh-per-call implementation
# was based on a misreading of openai-python #769, which was about
# not closing clients, not about reuse.)
#
# Wrapped in _LazyClient (same pattern as the provider clients above)
# so importing this module does not require OPENAI_API_KEY to be set —
# the underlying AsyncOpenAI + httpx pool are built on first attribute
# access and cached for the lifetime of the proxy.
def _build_embedding_client() -> AsyncOpenAI:
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=60.0,
        ),
        timeout=httpx.Timeout(10.0, connect=2.0),
    )
    return AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        http_client=http_client,
        max_retries=0,  # We handle retries at the caller level.
    )


_embedding_client = _LazyClient(_build_embedding_client)


async def generate_vector_embedding(
    text: list[str],
    model: str = "text-embedding-3-large",
) -> list[list[float]]:
    """Embed a batch of texts in a single OpenAI call.

    The OpenAI embeddings endpoint accepts a list of up to 2048
    inputs per call; callers that need multiple embeddings should
    pass them all in `text` rather than fanning out N single-input
    calls. Returns one embedding vector per input, in the same order.
    """
    try:
        response = await _embedding_client.embeddings.create(
            model=model,
            input=text,
        )
        # Account the embedding cost into the request-cost rollup. Embeddings
        # have no output tokens and no prompt caching, so total_tokens is pure
        # input. No-op outside a tracked /query_search request; unpriced model
        # yields None which the accumulator ignores.
        usage = getattr(response, "usage", None)
        if usage is not None:
            add_request_cost(
                compute_llm_cost_usd(model, usage.total_tokens, 0)
            )
        return [item.embedding for item in response.data]
    except Exception as e:
        raise ValueError(f"OpenAI failed to generate vector embedding: {e}")