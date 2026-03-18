"""
Unit tests for implementation.llms.generic_methods.

Tests the LLMProvider enum, individual provider generation functions
(Kimi async/sync, Gemini, Groq, Alibaba, WHAM), and the unified router
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
    generate_anthropic_response_async,
    generate_wham_response_async,
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


def _make_anthropic_style_response(
    tool_input: dict,
    input_tokens: int = 10,
    output_tokens: int = 5,
):
    """Build a mock Anthropic API response with a tool_use content block."""
    tool_use_block = SimpleNamespace(type="tool_use", input=tool_input)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=[tool_use_block], usage=usage)


# ---------------------------------------------------------------------------
# Tests: LLMProvider enum
# ---------------------------------------------------------------------------


class TestLLMProvider:
    """Tests for the LLMProvider enum."""

    def test_llm_provider_values(self) -> None:
        """All seven members exist with expected string values."""
        expected = {
            "OPENAI": "openai",
            "KIMI": "kimi",
            "GEMINI": "gemini",
            "GROQ": "groq",
            "ALIBABA": "alibaba",
            "ANTHROPIC": "anthropic",
            "WHAM": "wham",
        }
        for name, value in expected.items():
            assert LLMProvider[name].value == value

    def test_llm_provider_includes_anthropic(self) -> None:
        """LLMProvider.ANTHROPIC exists with value 'anthropic'."""
        assert LLMProvider.ANTHROPIC.value == "anthropic"

    def test_llm_provider_includes_wham(self) -> None:
        """LLMProvider.WHAM exists with value 'wham'."""
        assert LLMProvider.WHAM.value == "wham"

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
# Tests: generate_anthropic_response_async
# ---------------------------------------------------------------------------


class TestGenerateAnthropicResponseAsync:
    """Tests for the async Anthropic generation function."""

    async def test_anthropic_async_returns_parsed_model_and_tokens(self) -> None:
        """Return value is a 3-tuple of (parsed Pydantic model, input_tokens, output_tokens)."""
        mock_response = _make_anthropic_style_response(
            {"value": "anthropic-ok"}, input_tokens=80, output_tokens=40,
        )

        with patch(
            "implementation.llms.generic_methods.async_anthropic_client.messages.create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await generate_anthropic_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
            )

        assert isinstance(result, tuple)
        assert len(result) == 3
        parsed, input_tokens, output_tokens = result
        assert isinstance(parsed, _DummyResponse)
        assert parsed.value == "anthropic-ok"
        assert input_tokens == 80
        assert output_tokens == 40

    async def test_anthropic_async_uses_tool_use_pattern(self) -> None:
        """The messages.create call receives a tools list with name='structured_output'
        and tool_choice forcing that tool."""
        mock_response = _make_anthropic_style_response({"value": "ok"})
        mock_create = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.async_anthropic_client.messages.create",
            mock_create,
        ):
            await generate_anthropic_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
            )

        call_kwargs = mock_create.call_args[1]
        # Verify tools contains a structured_output tool
        tools = call_kwargs["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "structured_output"
        # Verify tool_choice forces the structured_output tool
        assert call_kwargs["tool_choice"] == {"type": "tool", "name": "structured_output"}

    async def test_anthropic_async_passes_system_as_system_param(self) -> None:
        """system_prompt is passed as the system= param (not as a message)."""
        mock_response = _make_anthropic_style_response({"value": "ok"})
        mock_create = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.async_anthropic_client.messages.create",
            mock_create,
        ):
            await generate_anthropic_response_async(
                user_prompt="test",
                system_prompt="You are a helpful assistant.",
                response_format=_DummyResponse,
            )

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["system"] == "You are a helpful assistant."

    async def test_anthropic_async_passes_user_prompt_as_message(self) -> None:
        """Messages contain a single user message with the user_prompt content."""
        mock_response = _make_anthropic_style_response({"value": "ok"})
        mock_create = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.async_anthropic_client.messages.create",
            mock_create,
        ):
            await generate_anthropic_response_async(
                user_prompt="Tell me about movies.",
                system_prompt="test",
                response_format=_DummyResponse,
            )

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "Tell me about movies."}]

    async def test_anthropic_async_default_max_tokens(self) -> None:
        """max_tokens defaults to 4096 when not provided in kwargs."""
        mock_response = _make_anthropic_style_response({"value": "ok"})
        mock_create = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.async_anthropic_client.messages.create",
            mock_create,
        ):
            await generate_anthropic_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
            )

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["max_tokens"] == 4096

    async def test_anthropic_async_custom_max_tokens_from_kwargs(self) -> None:
        """max_tokens is extracted from kwargs and passed to messages.create (not double-passed)."""
        mock_response = _make_anthropic_style_response({"value": "ok"})
        mock_create = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.async_anthropic_client.messages.create",
            mock_create,
        ):
            await generate_anthropic_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                max_tokens=2048,
            )

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["max_tokens"] == 2048

    async def test_anthropic_async_forwards_additional_kwargs(self) -> None:
        """Extra kwargs (e.g., temperature) are forwarded to messages.create."""
        mock_response = _make_anthropic_style_response({"value": "ok"})
        mock_create = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.async_anthropic_client.messages.create",
            mock_create,
        ):
            await generate_anthropic_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                temperature=0.3,
            )

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["temperature"] == 0.3

    async def test_anthropic_async_raises_value_error_on_failure(self) -> None:
        """ValueError is raised with 'Anthropic async failed' prefix when the API call raises."""
        with patch(
            "implementation.llms.generic_methods.async_anthropic_client.messages.create",
            new_callable=AsyncMock,
            side_effect=RuntimeError("connection refused"),
        ):
            with pytest.raises(ValueError, match="Anthropic async failed"):
                await generate_anthropic_response_async(
                    user_prompt="test",
                    system_prompt="test",
                    response_format=_DummyResponse,
                )

    async def test_anthropic_async_extracts_tool_use_block(self) -> None:
        """Correctly finds the tool_use block from response.content and validates against response_format."""
        # Response with a text block first, then the tool_use block
        text_block = SimpleNamespace(type="text", text="Some thinking text")
        tool_use_block = SimpleNamespace(type="tool_use", input={"value": "from-tool"})
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        mock_response = SimpleNamespace(content=[text_block, tool_use_block], usage=usage)

        with patch(
            "implementation.llms.generic_methods.async_anthropic_client.messages.create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            parsed, _, _ = await generate_anthropic_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
            )

        assert isinstance(parsed, _DummyResponse)
        assert parsed.value == "from-tool"

    async def test_anthropic_async_no_tool_use_block_raises(self) -> None:
        """StopIteration (from next()) is caught and wrapped in ValueError when no tool_use block exists."""
        text_block = SimpleNamespace(type="text", text="No tool use here")
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        mock_response = SimpleNamespace(content=[text_block], usage=usage)

        with patch(
            "implementation.llms.generic_methods.async_anthropic_client.messages.create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with pytest.raises(ValueError, match="Anthropic async failed"):
                await generate_anthropic_response_async(
                    user_prompt="test",
                    system_prompt="test",
                    response_format=_DummyResponse,
                )

    async def test_anthropic_async_max_tokens_not_double_passed(self) -> None:
        """max_tokens is popped from kwargs, so it isn't passed twice to messages.create."""
        mock_response = _make_anthropic_style_response({"value": "ok"})
        mock_create = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.async_anthropic_client.messages.create",
            mock_create,
        ):
            await generate_anthropic_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                max_tokens=4096,
            )

        # messages.create should have been called with max_tokens exactly once
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["max_tokens"] == 4096
        # Verify max_tokens doesn't appear in any extra **kwargs expansion
        # (if it were double-passed, the call would fail with a duplicate keyword arg)
        mock_create.assert_called_once()

    async def test_budget_tokens_activates_thinking_dict_in_api_call(self) -> None:
        """When budget_tokens is passed, messages.create receives thinking={"type":"enabled","budget_tokens":N}."""
        mock_response = _make_anthropic_style_response({"value": "ok"})
        mock_create = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.async_anthropic_client.messages.create",
            mock_create,
        ):
            await generate_anthropic_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                budget_tokens=8000,
            )

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 8000}

    async def test_budget_tokens_overrides_max_tokens_to_budget_plus_4096(self) -> None:
        """When budget_tokens=8000, the API call receives max_tokens=12096 (8000 + 4096)."""
        mock_response = _make_anthropic_style_response({"value": "ok"})
        mock_create = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.async_anthropic_client.messages.create",
            mock_create,
        ):
            await generate_anthropic_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                budget_tokens=8000,
            )

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["max_tokens"] == 12096  # 8000 + 4096

    async def test_budget_tokens_without_max_tokens_uses_budget_plus_4096(self) -> None:
        """budget_tokens=2000 with no explicit max_tokens produces max_tokens=6096, not 4096."""
        mock_response = _make_anthropic_style_response({"value": "ok"})
        mock_create = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.async_anthropic_client.messages.create",
            mock_create,
        ):
            await generate_anthropic_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                budget_tokens=2000,
            )

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["max_tokens"] == 6096  # 2000 + 4096, not the default 4096

    async def test_budget_tokens_override_wins_over_explicit_max_tokens(self) -> None:
        """When both max_tokens=1000 and budget_tokens=2000 are provided, budget_tokens+4096 wins."""
        mock_response = _make_anthropic_style_response({"value": "ok"})
        mock_create = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.async_anthropic_client.messages.create",
            mock_create,
        ):
            await generate_anthropic_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                max_tokens=1000,
                budget_tokens=2000,
            )

        call_kwargs = mock_create.call_args[1]
        # The budget branch runs after the max_tokens pop, overriding whatever was there
        assert call_kwargs["max_tokens"] == 6096  # 2000 + 4096, not 1000

    async def test_budget_tokens_not_forwarded_to_messages_create(self) -> None:
        """budget_tokens is consumed (popped) before the API call and must not appear in forwarded kwargs."""
        mock_response = _make_anthropic_style_response({"value": "ok"})
        mock_create = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.async_anthropic_client.messages.create",
            mock_create,
        ):
            await generate_anthropic_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                budget_tokens=8000,
            )

        call_kwargs = mock_create.call_args[1]
        assert "budget_tokens" not in call_kwargs

    async def test_without_budget_tokens_no_thinking_key_injected(self) -> None:
        """When budget_tokens is absent, the thinking key must not appear in the API call kwargs."""
        mock_response = _make_anthropic_style_response({"value": "ok"})
        mock_create = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.async_anthropic_client.messages.create",
            mock_create,
        ):
            await generate_anthropic_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                temperature=0.3,
            )

        call_kwargs = mock_create.call_args[1]
        assert "thinking" not in call_kwargs

    async def test_budget_tokens_none_does_not_inject_thinking(self) -> None:
        """Explicitly passing budget_tokens=None must not inject the thinking dict."""
        mock_response = _make_anthropic_style_response({"value": "ok"})
        mock_create = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.async_anthropic_client.messages.create",
            mock_create,
        ):
            await generate_anthropic_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                budget_tokens=None,
            )

        call_kwargs = mock_create.call_args[1]
        assert "thinking" not in call_kwargs

    async def test_budget_tokens_boundary_zero_sets_thinking_dict(self) -> None:
        """budget_tokens=0 still enables extended thinking mode with budget 0."""
        mock_response = _make_anthropic_style_response({"value": "ok"})
        mock_create = AsyncMock(return_value=mock_response)

        with patch(
            "implementation.llms.generic_methods.async_anthropic_client.messages.create",
            mock_create,
        ):
            await generate_anthropic_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                budget_tokens=0,
            )

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 0}


# ---------------------------------------------------------------------------
# Tests: generate_wham_response_async
# ---------------------------------------------------------------------------


class _MockWhamStream:
    """Async context manager that mimics client.responses.stream().

    Captures the kwargs passed to stream() for assertion, and returns
    a configurable mock response from get_final_response().
    """

    def __init__(self, response):
        self._response = response
        self.call_kwargs: dict = {}

    def __call__(self, **kwargs):
        """Capture kwargs when used as stream(...)."""
        self.call_kwargs = kwargs
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def get_final_response(self):
        return self._response


def _make_wham_response(parsed_output=None, input_tokens=10, output_tokens=5):
    """Build a mock WHAM streaming final response."""
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(output_parsed=parsed_output, usage=usage)


def _setup_wham_mock(parsed_output=None, input_tokens=10, output_tokens=5):
    """Create a mock AsyncOpenAI class and stream for WHAM tests.

    Returns (mock_openai_cls, stream_mock) where stream_mock.call_kwargs
    contains the args passed to responses.stream() after the function runs.
    """
    if parsed_output is None:
        parsed_output = _DummyResponse(value="wham-ok")
    response = _make_wham_response(parsed_output, input_tokens, output_tokens)
    stream_mock = _MockWhamStream(response)

    mock_client = MagicMock()
    mock_client.responses.stream = stream_mock

    mock_openai_cls = MagicMock(return_value=mock_client)
    return mock_openai_cls, stream_mock


class TestGenerateWhamResponseAsync:
    """Tests for the async WHAM generation function.

    All tests patch AsyncOpenAI to avoid real network calls and to capture
    the parameters passed to the WHAM streaming API.
    """

    _OPENAI_CLS_PATCH = "implementation.llms.generic_methods.AsyncOpenAI"

    # -- Parameter stripping ------------------------------------------------

    async def test_wham_async_strips_max_tokens_from_kwargs(self) -> None:
        """max_tokens passed in kwargs does NOT appear in the stream() call."""
        mock_cls, stream = _setup_wham_mock()

        with patch(self._OPENAI_CLS_PATCH, mock_cls):
            await generate_wham_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                api_key="tok",
                account_id="acct",
                max_tokens=4096,
            )

        assert "max_tokens" not in stream.call_kwargs

    async def test_wham_async_strips_max_output_tokens_from_kwargs(self) -> None:
        """max_output_tokens passed in kwargs does NOT appear in the stream() call."""
        mock_cls, stream = _setup_wham_mock()

        with patch(self._OPENAI_CLS_PATCH, mock_cls):
            await generate_wham_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                api_key="tok",
                account_id="acct",
                max_output_tokens=2048,
            )

        assert "max_output_tokens" not in stream.call_kwargs

    async def test_wham_async_strips_temperature_from_kwargs(self) -> None:
        """temperature passed in kwargs does NOT appear in the stream() call."""
        mock_cls, stream = _setup_wham_mock()

        with patch(self._OPENAI_CLS_PATCH, mock_cls):
            await generate_wham_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                api_key="tok",
                account_id="acct",
                temperature=0.5,
            )

        assert "temperature" not in stream.call_kwargs

    async def test_wham_async_strips_all_unsupported_params_simultaneously(self) -> None:
        """All three unsupported params (max_tokens, max_output_tokens, temperature) stripped at once."""
        mock_cls, stream = _setup_wham_mock()

        with patch(self._OPENAI_CLS_PATCH, mock_cls):
            await generate_wham_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                api_key="tok",
                account_id="acct",
                max_tokens=4096,
                max_output_tokens=2048,
                temperature=0.5,
            )

        for key in ("max_tokens", "max_output_tokens", "temperature"):
            assert key not in stream.call_kwargs

    # -- Supported param forwarding -----------------------------------------

    async def test_wham_async_forwards_verbosity(self) -> None:
        """verbosity kwarg is forwarded to the stream() call."""
        mock_cls, stream = _setup_wham_mock()

        with patch(self._OPENAI_CLS_PATCH, mock_cls):
            await generate_wham_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                api_key="tok",
                account_id="acct",
                verbosity="low",
            )

        assert stream.call_kwargs["verbosity"] == "low"

    async def test_wham_async_forwards_reasoning_effort_as_nested_object(self) -> None:
        """reasoning_effort='low' becomes reasoning={"effort": "low"} in the stream() call."""
        mock_cls, stream = _setup_wham_mock()

        with patch(self._OPENAI_CLS_PATCH, mock_cls):
            await generate_wham_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                api_key="tok",
                account_id="acct",
                reasoning_effort="low",
            )

        assert stream.call_kwargs["reasoning"] == {"effort": "low"}

    async def test_wham_async_reasoning_effort_none_not_forwarded(self) -> None:
        """When reasoning_effort is not passed, 'reasoning' key does not appear in stream() call."""
        mock_cls, stream = _setup_wham_mock()

        with patch(self._OPENAI_CLS_PATCH, mock_cls):
            await generate_wham_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                api_key="tok",
                account_id="acct",
            )

        assert "reasoning" not in stream.call_kwargs

    # -- Validation and error paths -----------------------------------------

    async def test_wham_async_raises_without_api_key(self) -> None:
        """ValueError mentioning 'OAuth' when api_key is None."""
        with pytest.raises(ValueError, match="OAuth"):
            await generate_wham_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                api_key=None,
                account_id="acct",
            )

    async def test_wham_async_raises_without_account_id(self) -> None:
        """ValueError when account_id is None."""
        with pytest.raises(ValueError, match="account_id"):
            await generate_wham_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                api_key="tok",
                account_id=None,
            )

    async def test_wham_async_raises_when_parsed_is_none(self) -> None:
        """ValueError mentioning 'parsed output' when output_parsed is None."""
        mock_cls, _ = _setup_wham_mock()
        # Override the response to have output_parsed=None
        response = _make_wham_response(parsed_output=None)
        stream_mock = _MockWhamStream(response)
        mock_client = MagicMock()
        mock_client.responses.stream = stream_mock
        mock_cls.return_value = mock_client

        with patch(self._OPENAI_CLS_PATCH, mock_cls):
            with pytest.raises(ValueError, match="parsed output"):
                await generate_wham_response_async(
                    user_prompt="test",
                    system_prompt="test",
                    response_format=_DummyResponse,
                    api_key="tok",
                    account_id="acct",
                )

    async def test_wham_async_wraps_exceptions_as_value_error(self) -> None:
        """Generic exceptions are wrapped with 'WHAM async failed' prefix."""
        mock_client = MagicMock()
        # Make stream() raise when entered as a context manager
        mock_stream = MagicMock()
        mock_stream.__aenter__ = AsyncMock(side_effect=RuntimeError("connection lost"))
        mock_stream.__aexit__ = AsyncMock()
        mock_client.responses.stream = MagicMock(return_value=mock_stream)
        mock_cls = MagicMock(return_value=mock_client)

        with patch(self._OPENAI_CLS_PATCH, mock_cls):
            with pytest.raises(ValueError, match="WHAM async failed"):
                await generate_wham_response_async(
                    user_prompt="test",
                    system_prompt="test",
                    response_format=_DummyResponse,
                    api_key="tok",
                    account_id="acct",
                )

    # -- Return value -------------------------------------------------------

    async def test_wham_async_returns_tuple_of_three(self) -> None:
        """Return value is a 3-tuple of (parsed_response, input_tokens, output_tokens)."""
        mock_cls, _ = _setup_wham_mock(
            parsed_output=_DummyResponse(value="wham-ok"),
            input_tokens=100,
            output_tokens=50,
        )

        with patch(self._OPENAI_CLS_PATCH, mock_cls):
            result = await generate_wham_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                api_key="tok",
                account_id="acct",
            )

        assert isinstance(result, tuple)
        assert len(result) == 3
        parsed, input_tokens, output_tokens = result
        assert isinstance(parsed, _DummyResponse)
        assert parsed.value == "wham-ok"
        assert input_tokens == 100
        assert output_tokens == 50

    # -- Client construction ------------------------------------------------

    async def test_wham_async_creates_client_with_correct_base_url_and_headers(self) -> None:
        """AsyncOpenAI is constructed with WHAM base_url and ChatGPT-Account-Id header."""
        mock_cls, _ = _setup_wham_mock()

        with patch(self._OPENAI_CLS_PATCH, mock_cls):
            await generate_wham_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                api_key="my-token",
                account_id="my-account",
            )

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["api_key"] == "my-token"
        assert "chatgpt.com" in call_kwargs["base_url"]
        assert call_kwargs["default_headers"]["ChatGPT-Account-Id"] == "my-account"

    # -- Stream call structure ----------------------------------------------

    async def test_wham_async_stream_uses_store_false(self) -> None:
        """store=False is always passed to the stream() call (WHAM requirement)."""
        mock_cls, stream = _setup_wham_mock()

        with patch(self._OPENAI_CLS_PATCH, mock_cls):
            await generate_wham_response_async(
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                api_key="tok",
                account_id="acct",
            )

        assert stream.call_kwargs["store"] is False

    async def test_wham_async_passes_instructions_not_system_message(self) -> None:
        """System prompt is passed as 'instructions' param, not in messages."""
        mock_cls, stream = _setup_wham_mock()

        with patch(self._OPENAI_CLS_PATCH, mock_cls):
            await generate_wham_response_async(
                user_prompt="test",
                system_prompt="You are helpful.",
                response_format=_DummyResponse,
                api_key="tok",
                account_id="acct",
            )

        assert stream.call_kwargs["instructions"] == "You are helpful."


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

    async def test_router_dispatches_to_anthropic(self) -> None:
        """Router calls the ANTHROPIC dispatch entry with model param."""
        mock_fn = AsyncMock(return_value=(_DummyResponse(value="ok"), 10, 5))

        with patch.dict(_PROVIDER_DISPATCH, {LLMProvider.ANTHROPIC: mock_fn}):
            result = await generate_llm_response_async(
                provider=LLMProvider.ANTHROPIC,
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                model="claude-opus-4-6",
            )

        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-6"
        assert result == (_DummyResponse(value="ok"), 10, 5)

    def test_provider_dispatch_includes_anthropic(self) -> None:
        """LLMProvider.ANTHROPIC is a key in _PROVIDER_DISPATCH."""
        assert LLMProvider.ANTHROPIC in _PROVIDER_DISPATCH

    async def test_router_forwards_budget_tokens_to_anthropic(self) -> None:
        """budget_tokens kwarg passes through the router unchanged to the Anthropic provider function."""
        mock_fn = AsyncMock(return_value=(_DummyResponse(value="ok"), 10, 5))

        with patch.dict(_PROVIDER_DISPATCH, {LLMProvider.ANTHROPIC: mock_fn}):
            await generate_llm_response_async(
                provider=LLMProvider.ANTHROPIC,
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                model="claude-sonnet-4-6",
                budget_tokens=8000,
            )

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["budget_tokens"] == 8000

    async def test_router_dispatches_to_wham(self) -> None:
        """Router calls the WHAM dispatch entry with model param and kwargs."""
        mock_fn = AsyncMock(return_value=(_DummyResponse(value="ok"), 10, 5))

        with patch.dict(_PROVIDER_DISPATCH, {LLMProvider.WHAM: mock_fn}):
            result = await generate_llm_response_async(
                provider=LLMProvider.WHAM,
                user_prompt="test",
                system_prompt="test",
                response_format=_DummyResponse,
                model="gpt-5.4",
                api_key="tok",
                account_id="acct",
            )

        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["model"] == "gpt-5.4"
        assert call_kwargs["api_key"] == "tok"
        assert result == (_DummyResponse(value="ok"), 10, 5)

    def test_provider_dispatch_includes_wham(self) -> None:
        """LLMProvider.WHAM is a key in _PROVIDER_DISPATCH."""
        assert LLMProvider.WHAM in _PROVIDER_DISPATCH
