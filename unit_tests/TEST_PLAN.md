# Test Plan

Generated: 2026-04-02
Based on diff: main..HEAD + unstaged changes

## Summary

Two source files changed: the production keywords generator had its signature
finalized (removed `provider`, `model`, `system_prompt`, `response_format`,
and `**kwargs` parameters, replacing them with hardcoded module-level
constants), and the production keywords prompt was substantially rewritten
with expanded classification categories and boundary rules. The existing test
file `test_production_keywords_generator.py` is now broken because every
`generate_production_keywords` call still passes the removed `provider` and
`model` keyword arguments.

## Changed Source Files

- `movie_ingestion/metadata_generation/generators/production_keywords.py` — Removed all configurable parameters from `generate_production_keywords`; provider, model, system_prompt, response_format, and `**kwargs` are now hardcoded as module-level constants (`_PROVIDER`, `_MODEL`, `_KWARGS`).
- `movie_ingestion/metadata_generation/prompts/production_keywords.py` — Rewrote `_PREAMBLE` with expanded 7-category classification taxonomy, explicit boundary rules, and a "WHAT DOES NOT COUNT" exclusion list. Output section wording updated for both variants.

## Test Coverage Analysis

### `movie_ingestion/metadata_generation/generators/production_keywords.py`
**What changed:** `generate_production_keywords` signature reduced to `(movie: MovieInputData)` only. All LLM config is hardcoded. `_DEFAULT_PROVIDER`/`_DEFAULT_MODEL` renamed to `_PROVIDER`/`_MODEL`, and `_KWARGS` added.
**Existing tests:** `unit_tests/test_production_keywords_generator.py`
**Coverage status:** Broken — 5 of 7 async tests pass removed kwargs and will fail with `TypeError`

#### Stale Tests That Must Be Updated

- `TestGenerateProductionKeywords::test_returns_output_and_token_usage` — Passes `provider=` and `model=` kwargs that no longer exist. Must call with `(movie)` only.
- `TestGenerateProductionKeywords::test_no_default_reasoning_effort_injected` — Tests that reasoning_effort is NOT passed. This is now inverted: reasoning_effort IS always passed (as "low"). Must be rewritten to assert the opposite.
- `TestGenerateProductionKeywordsErrors::test_wraps_llm_exception` — Passes `provider=` and `model=` kwargs. Must remove them.
- `TestGenerateProductionKeywordsErrors::test_raises_empty_response_error` — Same issue.
- `TestGenerateProductionKeywordsErrors::test_error_chains_original_cause` — Same issue.

#### New/Updated Tests Needed

- [ ] `test_returns_output_and_token_usage` — Update: call `generate_production_keywords(movie)` with no extra args. Assert returns `(ProductionKeywordsOutput, TokenUsage)`.
- [ ] `test_hardcoded_reasoning_effort` — New: assert `reasoning_effort='low'` is always passed to the LLM call (mirrors `test_source_of_inspiration_generator.py::test_hardcoded_reasoning_effort`).
- [ ] `test_hardcoded_llm_params` — New: assert all 5 hardcoded params are passed to the LLM call: `provider=LLMProvider.OPENAI`, `model="gpt-5-mini"`, `system_prompt=SYSTEM_PROMPT`, `response_format=ProductionKeywordsOutput`, `reasoning_effort="low"`. Follow the pattern from `TestSourceOfInspirationSignatureLockdown::test_hardcoded_llm_params`.
- [ ] `test_generate_does_not_accept_provider_kwarg` — New: use `inspect.signature` to verify `provider` is not in the function's parameters. Follow pattern from source_of_inspiration tests.
- [ ] `test_generate_does_not_accept_model_kwarg` — New: same for `model`.
- [ ] `test_generate_does_not_accept_kwargs` — New: verify no `**kwargs` in signature (i.e., no `VAR_KEYWORD` parameter kind).
- [ ] `test_token_usage_uses_hardcoded_model` — New: assert the returned `TokenUsage.model` equals `"gpt-5-mini"` (the hardcoded `_MODEL`), not a caller-provided value.

#### Edge Cases

- [ ] Verify that passing unexpected keyword arguments (e.g., `provider=...`) raises `TypeError` at the Python level (signature lockdown).

### `movie_ingestion/metadata_generation/prompts/production_keywords.py`
**What changed:** `_PREAMBLE` substantially rewritten. Now has 7 numbered categories (production medium, origin/language, source material, production process, franchise/ecosystem, production form, production era) plus explicit "WHAT DOES NOT COUNT" section and boundary rules. Output sections (`_OUTPUT_NO_JUSTIFICATIONS`, `_OUTPUT_WITH_JUSTIFICATIONS`) updated with new wording.
**Existing tests:** None dedicated. `test_prompt_constants.py` covers plot_events and source_of_inspiration prompts but not production_keywords.
**Coverage status:** No coverage

#### New Tests Needed

- [ ] `test_system_prompt_is_non_empty` — Assert `SYSTEM_PROMPT` is a non-empty string.
- [ ] `test_system_prompt_with_justifications_is_non_empty` — Assert `SYSTEM_PROMPT_WITH_JUSTIFICATIONS` is a non-empty string.
- [ ] `test_system_prompt_contains_core_test` — Assert the "core test" question appears (e.g., "real world" or "how the movie was made").
- [ ] `test_system_prompt_contains_all_seven_categories` — Assert the 7 category headings are present: "Production medium", "Origin and language", "Source material", "Production process", "Franchise", "Production form", "Production era".
- [ ] `test_system_prompt_contains_exclusion_section` — Assert "WHAT DOES NOT COUNT" section is present.
- [ ] `test_system_prompt_contains_no_invention_rule` — Assert the anti-hallucination rule is present (e.g., "Inventing new keywords is a catastrophic failure").
- [ ] `test_system_prompt_contains_empty_list_guidance` — Assert guidance about empty terms list being correct/expected.
- [ ] `test_justifications_variant_mentions_justification` — Assert `SYSTEM_PROMPT_WITH_JUSTIFICATIONS` contains "justification" in output section.
- [ ] `test_no_justifications_variant_omits_justification` — Assert `SYSTEM_PROMPT` (no-justifications variant) does not contain "justification" in the output section (note: it may appear in preamble boundary rules, so scope the check to the output section or check `_OUTPUT_NO_JUSTIFICATIONS` directly).

#### Edge Cases

- [ ] Verify both prompt variants share the same preamble (both start with the same `_PREAMBLE` content).

## Stale Tests

All 5 async tests in `test_production_keywords_generator.py` are broken:

| Test | Issue |
|------|-------|
| `test_returns_output_and_token_usage` | Passes `provider=`, `model=` kwargs |
| `test_no_default_reasoning_effort_injected` | Asserts reasoning_effort is absent; it is now always present |
| `test_wraps_llm_exception` | Passes `provider=`, `model=` kwargs |
| `test_raises_empty_response_error` | Passes `provider=`, `model=` kwargs |
| `test_error_chains_original_cause` | Passes `provider=`, `model=` kwargs |

The 4 prompt-building tests (`TestBuildProductionKeywordsUserPrompt`, `TestProductionKeywordsPromptFields`) are unaffected — `build_production_keywords_user_prompt` signature did not change.

## Conventions Observed

From the existing test suite (especially `test_source_of_inspiration_generator.py` as the closest analog):

- **File naming:** `test_{generator_name}_generator.py`
- **Class grouping:** Separate test classes by concern: prompt building, LLM delegation, error paths, signature lockdown, prompt field coverage
- **Mock pattern:** Patch `generate_llm_response_async` at the module level via `_LLM_PATCH` constant; use `AsyncMock`
- **Helper factories:** `_make_movie(**overrides)` for constructing `MovieInputData` with sensible defaults; `_make_{type}_output()` for constructing expected LLM output
- **Signature lockdown tests:** Use `inspect.signature` to assert removed parameters are absent (not just that calling with them fails)
- **Hardcoded param tests:** Inspect `mock_fn.call_args[1]` (kwargs dict) to verify all hardcoded params are forwarded correctly
- **Async tests:** No decorators needed (`asyncio_mode = "auto"`)
- **No real API calls:** All LLM calls mocked
- **Prompt constant tests:** Grouped in `test_prompt_constants.py` with lightweight structural assertions (non-empty, contains key phrases)
