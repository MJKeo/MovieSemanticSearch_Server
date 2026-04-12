"""Prompt-shape regression tests for vector subquery system prompts."""

from implementation.prompts.vector_subquery_prompts import WATCH_CONTEXT_SYSTEM_PROMPT


class TestWatchContextSubqueryPrompt:
    def test_describes_labeled_embedding_format(self) -> None:
        prompt = WATCH_CONTEXT_SYSTEM_PROMPT.lower()
        assert "fixed-order labeled lines" in prompt
        assert "self_experience_motivations:" in prompt
        assert "external_motivations:" in prompt
        assert "key_movie_feature_draws:" in prompt
        assert "watch_scenarios:" in prompt

    def test_no_longer_describes_flat_unlabeled_format(self) -> None:
        prompt = WATCH_CONTEXT_SYSTEM_PROMPT.lower()
        assert "flat, unlabeled" not in prompt
        assert "comma-separated list of short search-query-like" not in prompt
