"""
Unit tests for implementation.llms.generic_methods.

Tests the LLMProvider enum, individual provider generation functions
(Kimi async/sync, Gemini, Groq, Alibaba), and the unified router
(generate_llm_response_async).

All LLM client calls are mocked — no real API traffic.
"""

import json
from enum import Enum
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from implementation.llms.generic_methods import (
    LLMProvider,
    _PROVIDER_DISPATCH,
    generate_kimi_response_async,
    generate_kimi_response,
    generate_gemini_response_async,
    generate_groq_response_async,
    generate_alibaba_response_async,
    generate_llm_response_async,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyResponse(BaseModel):
    """Minimal Pydantic model used as response_format in tests."""
    value: str


def _make_openai_style_response(content: str, input_tokens: int = 10, output_tokens: int = 5):
    """Build a mock OpenAI-style API response with usage stats."""
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content, parsed=_DummyResponse(value="ok")),
    )
    usage = SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_gemini_style_response(text: str, input_tokens: int = 10, output_tokens: int = 5):
    """Build a mock Gemini-style API response with usage_metadata."""
    usage = SimpleNamespace(prompt_token_count=input_tokens, candidates_token_count=output_tokens)
    return SimpleNamespace(text=text, usage_metadata=usage)


# ---------------------------------------------------------------------------
# Tests: LLMProvider enum
# ---------------------------------------------------------------------------


class TestLLMProvider:
    """Tests for the LLMProvider enum."""

    def test_llm_provider_values(self) -> None:
        """All five members exist with expected string values."""
        expected = {
            "OPENAI": "openai",
            "KIMI": "kimi",
            "GEMINI": "gemini",
            "GROQ": "groq",
            "ALIBABA": "alibaba",
        }
        for name, value in expected.items():
            assert LLMProvider[name].value == value

    def test_llm_provider_is_enum(self) -> None:
        """LLMProvider members are proper Enum instances and support identity comparison."""
        assert isinstance(LLMProvider.OPENAI, Enum)
        assert LLMProvider.OPENAI is LLMProvider.OPENAI
        assert LLMProvider.OPENAI is not LLMProvider.KIMI


# ---------------------------------------------------------------------------
# Tests: generate_kimi_response_async
# ---------------------------------------------------------------------------


class TestGenerateKimiResponseAsync:
    """Tests for the async Kimi generation function."""

    async def test_kimi_async_returns_tuple_of_three(self) -> None:
        """Return value is a 3-tuple of (parsed_model, input_tokens, output_tokens)."""
        response_json = json.dumps({"value": "hello"})
        mock_response = _make_openai_style_response(response_json, input_tokens=100, output_tokens=50)

        with patch(
            "implementation.llms.generic_methods.async_kimi_client.chat.completions.create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await generate_kimi_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
            )

        assert isinstance(result, tuple)
        assert len(result) == 3
        parsed, input_tokens, output_tokens = result
        assert isinstance(parsed, _DummyResponse)
        assert parsed.value == "hello"
        assert input_tokens == 100
        assert output_tokens == 50

    async def test_kimi_async_uses_class_name_not_instance_name(self) -> None:
        """The json_schema.name field uses response_format.__name__ (the class name string)."""
        response_json = json.dumps({"value": "ok"})
        mock_response = _make_openai_style_response(response_json)
        mock_create = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.async_kimi_client.chat.completions.create",
            mock_create,
        ):
            await generate_kimi_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
            )

        # Inspect the response_format dict passed to create()
        call_kwargs = mock_create.call_args[1]
        schema_name = call_kwargs["response_format"]["json_schema"]["name"]
        assert schema_name == "_DummyResponse"

    async def test_kimi_async_raises_value_error_on_failure(self) -> None:
        """ValueError is raised with 'Kimi failed' prefix when the API call fails."""
        with patch(
            "implementation.llms.generic_methods.async_kimi_client.chat.completions.create",
            new_callable=AsyncMock,
            side_effect=RuntimeError("connection lost"),
        ):
            with pytest.raises(ValueError, match="Kimi failed"):
                await generate_kimi_response_async(
                    user_prompt="test",
                    system_prompt="test",
                    response_format=_DummyResponse,
                )

    async def test_kimi_async_thinking_enabled_sets_temperature_1(self) -> None:
        """When enable_thinking=True, temperature is set to 1.0."""
        response_json = json.dumps({"value": "ok"})
        mock_response = _make_openai_style_response(response_json)
        mock_create = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.async_kimi_client.chat.completions.create",
            mock_create,
        ):
            await generate_kimi_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                enable_thinking=True,
            )

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["temperature"] == 1.0


# ---------------------------------------------------------------------------
# Tests: generate_kimi_response (sync)
# ---------------------------------------------------------------------------


class TestGenerateKimiResponseSync:
    """Tests for the sync Kimi generation function."""

    def test_kimi_sync_returns_tuple_of_three(self) -> None:
        """Return value is a 3-tuple of (parsed_model, input_tokens, output_tokens)."""
        response_json = json.dumps({"value": "hello"})
        mock_response = _make_openai_style_response(response_json, input_tokens=200, output_tokens=75)

        with patch(
            "implementation.llms.generic_methods.kimi_client.chat.completions.create",
            return_value=mock_response,
        ):
            result = generate_kimi_response(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
            )

        assert isinstance(result, tuple)
        assert len(result) == 3
        parsed, input_tokens, output_tokens = result
        assert isinstance(parsed, _DummyResponse)
        assert input_tokens == 200
        assert output_tokens == 75

    def test_kimi_sync_raises_value_error_on_failure(self) -> None:
        """ValueError wrapping on exception."""
        with patch(
            "implementation.llms.generic_methods.kimi_client.chat.completions.create",
            side_effect=RuntimeError("timeout"),
        ):
            with pytest.raises(ValueError, match="Kimi failed"):
                generate_kimi_response(
                    user_prompt="test",
                    system_prompt="test",
                    response_format=_DummyResponse,
                )


# ---------------------------------------------------------------------------
# Tests: generate_gemini_response_async
# ---------------------------------------------------------------------------


class TestGenerateGeminiResponseAsync:
    """Tests for the async Gemini generation function."""

    async def test_gemini_async_returns_parsed_model_and_tokens(self) -> None:
        """Returns 3-tuple with correct token counts from usage_metadata."""
        response_text = json.dumps({"value": "gemini-ok"})
        mock_response = _make_gemini_style_response(response_text, input_tokens=30, output_tokens=15)
        mock_generate = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.gemini_client.aio.models.generate_content",
            mock_generate,
        ):
            parsed, in_tok, out_tok = await generate_gemini_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
            )

        assert isinstance(parsed, _DummyResponse)
        assert parsed.value == "gemini-ok"
        assert in_tok == 30
        assert out_tok == 15

    async def test_gemini_async_required_config_overrides_caller_kwargs(self) -> None:
        """Our required keys override caller-supplied kwargs in the config dict."""
        response_text = json.dumps({"value": "ok"})
        mock_response = _make_gemini_style_response(response_text)
        mock_generate = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.gemini_client.aio.models.generate_content",
            mock_generate,
        ):
            await generate_gemini_response_async(
                user_prompt="test",
                system_prompt="sys",
                response_format=_DummyResponse,
                # Caller tries to override the required mime type
                response_mime_type="text/plain",
            )

        call_kwargs = mock_generate.call_args[1]
        config = call_kwargs["config"]
        # Our override wins — structured output requires application/json
        assert config["response_mime_type"] == "application/json"

    async def test_gemini_async_system_instruction_in_config(self) -> None:
        """system_prompt is passed as system_instruction in the config, not as a message."""
        response_text = json.dumps({"value": "ok"})
        mock_response = _make_gemini_style_response(response_text)
        mock_generate = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.gemini_client.aio.models.generate_content",
            mock_generate,
        ):
            await generate_gemini_response_async(
                user_prompt="test",
                system_prompt="You are helpful.",
                response_format=_DummyResponse,
            )

        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["config"]["system_instruction"] == "You are helpful."

    async def test_gemini_async_raises_value_error_on_failure(self) -> None:
        """ValueError with 'Gemini async failed' prefix on exception."""
        with patch(
            "implementation.llms.generic_methods.gemini_client.aio.models.generate_content",
            new_callable=AsyncMock,
            side_effect=RuntimeError("quota exceeded"),
        ):
            with pytest.raises(ValueError, match="Gemini async failed"):
                await generate_gemini_response_async(
                    user_prompt="test",
                    system_prompt="test",
                    response_format=_DummyResponse,
                )

    async def test_gemini_async_forwards_model_param(self) -> None:
        """The model string is forwarded to generate_content."""
        response_text = json.dumps({"value": "ok"})
        mock_response = _make_gemini_style_response(response_text)
        mock_generate = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.gemini_client.aio.models.generate_content",
            mock_generate,
        ):
            await generate_gemini_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                model="gemini-2.5-pro",
            )

        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["model"] == "gemini-2.5-pro"


# ---------------------------------------------------------------------------
# Tests: generate_groq_response_async
# ---------------------------------------------------------------------------


class TestGenerateGroqResponseAsync:
    """Tests for the async Groq generation function."""

    async def test_groq_async_returns_parsed_model_and_tokens(self) -> None:
        """Returns 3-tuple with correctly parsed model and token counts."""
        response_json = json.dumps({"value": "groq-ok"})
        mock_response = _make_openai_style_response(response_json, input_tokens=20, output_tokens=10)
        mock_create = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.async_groq_client.chat.completions.create",
            mock_create,
        ):
            parsed, in_tok, out_tok = await generate_groq_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
            )

        assert isinstance(parsed, _DummyResponse)
        assert parsed.value == "groq-ok"
        assert in_tok == 20
        assert out_tok == 10

    async def test_groq_async_uses_strict_false(self) -> None:
        """The json_schema dict has strict=False (differs from Kimi's True)."""
        response_json = json.dumps({"value": "ok"})
        mock_response = _make_openai_style_response(response_json)
        mock_create = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.async_groq_client.chat.completions.create",
            mock_create,
        ):
            await generate_groq_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
            )

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["response_format"]["json_schema"]["strict"] is False

    async def test_groq_async_forwards_kwargs(self) -> None:
        """Provider-specific kwargs (e.g. temperature) are forwarded to create()."""
        response_json = json.dumps({"value": "ok"})
        mock_response = _make_openai_style_response(response_json)
        mock_create = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.async_groq_client.chat.completions.create",
            mock_create,
        ):
            await generate_groq_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                temperature=0.5,
            )

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["temperature"] == 0.5

    async def test_groq_async_raises_value_error_on_failure(self) -> None:
        """ValueError wrapping with 'Groq async failed' prefix."""
        with patch(
            "implementation.llms.generic_methods.async_groq_client.chat.completions.create",
            new_callable=AsyncMock,
            side_effect=RuntimeError("rate limited"),
        ):
            with pytest.raises(ValueError, match="Groq async failed"):
                await generate_groq_response_async(
                    user_prompt="test",
                    system_prompt="test",
                    response_format=_DummyResponse,
                )


# ---------------------------------------------------------------------------
# Tests: generate_alibaba_response_async
# ---------------------------------------------------------------------------


class TestGenerateAlibabaResponseAsync:
    """Tests for the async Alibaba/Qwen generation function."""

    async def test_alibaba_async_returns_parsed_model_and_tokens(self) -> None:
        """Returns 3-tuple with parsed model from .parse() and token counts."""
        mock_response = _make_openai_style_response("unused", input_tokens=40, output_tokens=20)
        mock_parse = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.async_alibaba_client.chat.completions.parse",
            mock_parse,
        ):
            parsed, in_tok, out_tok = await generate_alibaba_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
            )

        assert isinstance(parsed, _DummyResponse)
        assert in_tok == 40
        assert out_tok == 20

    async def test_alibaba_async_uses_parse_not_create(self) -> None:
        """Alibaba uses .parse() (not .create()) for structured output."""
        mock_response = _make_openai_style_response("unused")
        mock_parse = AsyncMock(return_value=mock_response)
        mock_create = AsyncMock()

        with patch(
            "implementation.llms.generic_methods.async_alibaba_client.chat.completions.parse",
            mock_parse,
        ), patch(
            "implementation.llms.generic_methods.async_alibaba_client.chat.completions.create",
            mock_create,
        ):
            await generate_alibaba_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
            )

        mock_parse.assert_called_once()
        mock_create.assert_not_called()

    async def test_alibaba_async_raises_value_error_on_failure(self) -> None:
        """ValueError wrapping with 'Alibaba/Qwen async failed' prefix."""
        with patch(
            "implementation.llms.generic_methods.async_alibaba_client.chat.completions.parse",
            new_callable=AsyncMock,
            side_effect=RuntimeError("auth error"),
        ):
            with pytest.raises(ValueError, match="Alibaba/Qwen async failed"):
                await generate_alibaba_response_async(
                    user_prompt="test",
                    system_prompt="test",
                    response_format=_DummyResponse,
                )


# ---------------------------------------------------------------------------
# Tests: generate_llm_response_async (unified router)
# ---------------------------------------------------------------------------


class TestGenerateLLMResponseAsync:
    """Tests for the unified LLM router.

    The router reads from _PROVIDER_DISPATCH (populated at module load),
    so we patch dict entries directly rather than module-level names.
    """

    async def test_router_dispatches_to_openai(self) -> None:
        """Router calls the OPENAI dispatch entry with model param."""
        mock_fn = AsyncMock(return_value=(_DummyResponse(value="ok"), 10, 5))

        with patch.dict(_PROVIDER_DISPATCH, {LLMProvider.OPENAI: mock_fn}):
            result = await generate_llm_response_async(
                provider=LLMProvider.OPENAI,
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                model="gpt-5-mini",
            )

        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["model"] == "gpt-5-mini"
        assert result == (_DummyResponse(value="ok"), 10, 5)

    async def test_router_dispatches_to_kimi_without_model(self) -> None:
        """Router calls the KIMI dispatch entry WITHOUT model param."""
        mock_fn = AsyncMock(return_value=(_DummyResponse(value="ok"), 10, 5))

        with patch.dict(_PROVIDER_DISPATCH, {LLMProvider.KIMI: mock_fn}):
            await generate_llm_response_async(
                provider=LLMProvider.KIMI,
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                model="ignored",
            )

        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args[1]
        assert "model" not in call_kwargs

    async def test_router_dispatches_to_gemini(self) -> None:
        """Router calls the GEMINI dispatch entry with model param."""
        mock_fn = AsyncMock(return_value=(_DummyResponse(value="ok"), 10, 5))

        with patch.dict(_PROVIDER_DISPATCH, {LLMProvider.GEMINI: mock_fn}):
            await generate_llm_response_async(
                provider=LLMProvider.GEMINI,
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                model="gemini-2.5-flash",
            )

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["model"] == "gemini-2.5-flash"

    async def test_router_dispatches_to_groq(self) -> None:
        """Router calls the GROQ dispatch entry with model param."""
        mock_fn = AsyncMock(return_value=(_DummyResponse(value="ok"), 10, 5))

        with patch.dict(_PROVIDER_DISPATCH, {LLMProvider.GROQ: mock_fn}):
            await generate_llm_response_async(
                provider=LLMProvider.GROQ,
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                model="llama-3.3-70b",
            )

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["model"] == "llama-3.3-70b"

    async def test_router_dispatches_to_alibaba(self) -> None:
        """Router calls the ALIBABA dispatch entry with model param."""
        mock_fn = AsyncMock(return_value=(_DummyResponse(value="ok"), 10, 5))

        with patch.dict(_PROVIDER_DISPATCH, {LLMProvider.ALIBABA: mock_fn}):
            await generate_llm_response_async(
                provider=LLMProvider.ALIBABA,
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                model="qwen-plus",
            )

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["model"] == "qwen-plus"

    async def test_router_forwards_kwargs_to_provider(self) -> None:
        """Provider-specific kwargs pass through the router to the underlying function."""
        mock_fn = AsyncMock(return_value=(_DummyResponse(value="ok"), 10, 5))

        with patch.dict(_PROVIDER_DISPATCH, {LLMProvider.GEMINI: mock_fn}):
            await generate_llm_response_async(
                provider=LLMProvider.GEMINI,
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                model="gemini-2.5-flash",
                temperature=0.7,
            )

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["temperature"] == 0.7

    async def test_router_raises_key_error_for_unknown_provider(self) -> None:
        """KeyError when an unknown provider is passed (tests _PROVIDER_DISPATCH dict)."""
        with pytest.raises(KeyError):
            await generate_llm_response_async(
                provider="nonexistent",
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                model="test",
            )

    async def test_router_propagates_provider_errors_unchanged(self) -> None:
        """Errors from underlying provider functions propagate without re-wrapping."""
        original_error = ValueError("Kimi failed to generate response: timeout")
        mock_fn = AsyncMock(side_effect=original_error)

        with patch.dict(_PROVIDER_DISPATCH, {LLMProvider.KIMI: mock_fn}):
            with pytest.raises(ValueError) as exc_info:
                await generate_llm_response_async(
                    provider=LLMProvider.KIMI,
                    user_prompt="test",
                    system_prompt="test",
                    response_format=_DummyResponse,
                    model="ignored",
                )

        assert exc_info.value is original_error
