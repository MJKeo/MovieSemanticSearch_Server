import os
import time
import json
import re
import sys
import csv
import asyncio
import random
import logging

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
) -> Tuple[BaseModel, int, int]:
    """Async counterpart to generate_openai_response.

    Uses async_openai_client.chat.completions.parse() with the same
    parameters and return type as the sync version.

    Additional OpenAI-specific params (max_completion_tokens, etc.)
    can be passed via kwargs.

    Returns a tuple of (parsed_response, input_tokens, output_tokens).
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

        # Extract the parsed response - OpenAI automatically validates structure matches response_format
        parsed = response.choices[0].message.parsed
        return parsed, input_tokens, output_tokens
    except Exception as e:
        print(f"OpenAI async failed to generate response: {e}")
        raise ValueError(f"OpenAI async failed to generate response: {e}")


async def generate_kimi_response_async(
    user_prompt: str,
    system_prompt: str,
    response_format: BaseModel,
    enable_thinking: bool = False,
) -> Tuple[BaseModel, int, int]:
    """Generate a structured response using the Kimi (Moonshot) API.

    Returns a tuple of (parsed_response, input_tokens, output_tokens).
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
) -> Tuple[BaseModel, int, int]:
    """Generate a structured response using Google's Gemini API.

    Uses the native google-genai SDK with JSON schema structured output.
    Additional Gemini-specific params (temperature, top_p, top_k,
    max_output_tokens, etc.) can be passed via kwargs.

    Returns a tuple of (parsed_response, input_tokens, output_tokens).
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

        # Parse the JSON response into the Pydantic model
        parsed = response_format.model_validate_json(response.text)

        # Extract token usage from Gemini's usage metadata
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count

        # Log implicit cache hits — Gemini 2.5+ caches automatically when the
        # request shares a prefix with a recent call (min 1,024 tokens). The
        # field is None on a cache miss; coerce to 0 for readable output.
        cached_tokens = getattr(usage, "cached_content_token_count", None) or 0
        print(f"[Gemini] cached_content_token_count={cached_tokens} (prompt={input_tokens})")

        return parsed, input_tokens, output_tokens
    except Exception as e:
        raise ValueError(f"Gemini async failed to generate response: {e}")


async def generate_groq_response_async(
    user_prompt: str,
    system_prompt: str,
    response_format: BaseModel,
    model: str = "llama-3.3-70b-versatile",
    **kwargs,
) -> Tuple[BaseModel, int, int]:
    """Generate a structured response using Groq's native API.

    Uses the Groq SDK with json_schema response format (same pattern as Kimi).
    Additional Groq-specific params (temperature, top_p, max_completion_tokens,
    etc.) can be passed via kwargs.

    Returns a tuple of (parsed_response, input_tokens, output_tokens).
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

        # Parse the JSON string into the Pydantic model (same approach as Kimi)
        raw = response.choices[0].message.content
        data = json.loads(raw)
        parsed = response_format.model_validate(data)

        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

        return parsed, input_tokens, output_tokens
    except Exception as e:
        raise ValueError(f"Groq async failed to generate response: {e}")


async def generate_alibaba_response_async(
    user_prompt: str,
    system_prompt: str,
    response_format: BaseModel,
    model: str = "qwen-plus",
    **kwargs,
) -> Tuple[BaseModel, int, int]:
    """Generate a structured response using Alibaba's Qwen API via OpenAI-compatible routing.

    Uses AsyncOpenAI.chat.completions.parse() pointed at DashScope's
    compatible endpoint. Additional params (temperature, top_p, etc.)
    can be passed via kwargs.

    Returns a tuple of (parsed_response, input_tokens, output_tokens).
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

        parsed = response.choices[0].message.parsed
        return parsed, input_tokens, output_tokens
    except Exception as e:
        raise ValueError(f"Alibaba/Qwen async failed to generate response: {e}")


async def generate_anthropic_response_async(
    user_prompt: str,
    system_prompt: str,
    response_format: BaseModel,
    model: str = "claude-opus-4-6",
    **kwargs,
) -> Tuple[BaseModel, int, int]:
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

    Returns a tuple of (parsed_response, input_tokens, output_tokens).
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

        # Extract the tool_use block — guaranteed present when tool_choice forces it
        tool_use_block = next(
            block for block in response.content if block.type == "tool_use"
        )
        parsed = response_format.model_validate(tool_use_block.input)

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        return parsed, input_tokens, output_tokens
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
) -> Tuple[BaseModel, int, int]:
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

    Returns a tuple of (parsed_response, input_tokens, output_tokens).
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

        parsed = response.output_parsed
        if parsed is None:
            raise ValueError(
                "WHAM response did not contain parsed output. "
                "The model may have refused or returned an unexpected format."
            )

        usage = response.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0

        return parsed, input_tokens, output_tokens
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


async def generate_llm_response_async(
    provider: LLMProvider,
    user_prompt: str,
    system_prompt: str,
    response_format: BaseModel,
    model: str,
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
    `LLM_PER_ATTEMPT_TIMEOUT_SECONDS * LLM_MAX_ATTEMPTS` window
    instead of compounding across stacked retry wrappers.

    Returns a tuple of (parsed_response, input_tokens, output_tokens).
    """
    generate_fn = _PROVIDER_DISPATCH[provider]
    pass_model = provider not in _PROVIDERS_WITHOUT_MODEL_PARAM

    last_exc: BaseException | None = None
    for attempt in range(LLM_MAX_ATTEMPTS):
        try:
            if pass_model:
                return await asyncio.wait_for(
                    generate_fn(
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        response_format=response_format,
                        model=model,
                        **kwargs,
                    ),
                    timeout=LLM_PER_ATTEMPT_TIMEOUT_SECONDS,
                )
            return await asyncio.wait_for(
                generate_fn(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    response_format=response_format,
                    **kwargs,
                ),
                timeout=LLM_PER_ATTEMPT_TIMEOUT_SECONDS,
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
    assert last_exc is not None
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
        return [item.embedding for item in response.data]
    except Exception as e:
        raise ValueError(f"OpenAI failed to generate vector embedding: {e}")