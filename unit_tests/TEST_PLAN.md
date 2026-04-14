# Test Plan

Analysis of current test coverage against committed source code changes plus
uncommitted working-tree modifications. Organized by module, with each entry
specifying what to test, why, key assertions, and whether it is NEW or UPDATE.

---

## 1. schemas/enums.py

### 1a. Concept tag description attribute (UPDATE — uncommitted change)

**File:** `unit_tests/test_concept_tag_enums.py` (already written, untracked)

All 7 concept tag enums gained a `description: str` field (third tuple element).
The new test file `test_concept_tag_enums.py` already covers:
- Stability of (value, concept_tag_id) pairs for all 7 enums
- Member counts
- `description` attribute existence, non-emptiness, and type
- `str` subclass behavior, value lookup, invalid value rejection
- `ALL_CONCEPT_TAGS` correctness (count=25, excludes NO_CLEAR_CHOICE, global ID
  and value uniqueness)

**Status:** Coverage is complete. No additional tests needed for this change.

### 1b. LineagePosition enum (GAP — no dedicated stability tests)

**File:** `unit_tests/test_enums.py` — NEW section

LineagePosition has stable integer IDs persisted in Postgres. No dedicated
stability/uniqueness tests exist (it is used in integration-level tests for
franchise ingestion, but those don't guard against ID drift).

| Test | Assertions |
|------|-----------|
| `test_lineage_position_values_are_stable` | SEQUEL=("sequel",1), PREQUEL=("prequel",2), REMAKE=("remake",3), REBOOT=("reboot",4) |
| `test_lineage_position_member_count` | `len(LineagePosition) == 4` |
| `test_lineage_position_is_str_subclass` | `isinstance(LineagePosition.SEQUEL, str)` |
| `test_lineage_position_id_accessible` | `.lineage_position_id` returns int |
| `test_no_duplicate_ids` | All `lineage_position_id` values unique |
| `test_no_duplicate_values` | All `.value` strings unique |

---

## 2. schemas/metadata.py

### 2a. FranchiseOutput.validate_and_fix() (GAP — no tests)

**File:** `unit_tests/test_metadata_schemas.py` — NEW section `TestFranchiseOutputValidateAndFix`

The `validate_and_fix()` classmethod has three deterministic fixup rules with
non-trivial edge cases. No tests exist.

| Test | Why | Key assertions |
|------|-----|----------------|
| `test_lineage_null_clears_shared_universe_and_subgroups` | Rule 1: null-propagation | lineage=None -> shared_universe=None, recognized_subgroups=[], launched_subgroup=False |
| `test_lineage_null_preserves_lineage_position` | Rule 1 exception: position/spinoff/crossover preserved | lineage=None but lineage_position=SEQUEL remains |
| `test_lineage_null_preserves_is_crossover` | Rule 1 exception | is_crossover=True preserved when lineage=None |
| `test_lineage_null_preserves_is_spinoff` | Rule 1 exception | is_spinoff=True preserved when lineage=None |
| `test_launched_subgroup_false_when_no_recognized_subgroups` | Rule 2: coupling | launched_subgroup=True + recognized_subgroups=[] -> launched_subgroup=False |
| `test_launched_subgroup_true_with_recognized_subgroups` | Rule 2: positive case | launched_subgroup=True + non-empty list -> stays True |
| `test_launched_franchise_false_when_lineage_null` | Rule 3a | launched_franchise forced False |
| `test_launched_franchise_false_when_lineage_position_populated` | Rule 3b | position=SEQUEL -> launched_franchise forced False |
| `test_launched_franchise_false_when_is_spinoff` | Rule 3c | is_spinoff=True -> launched_franchise forced False |
| `test_launched_franchise_true_when_all_preconditions_met` | Rule 3: positive | lineage set, position None, not spinoff -> stays True |

### 2b. ConceptTagsOutput.validate_and_fix() and helpers (GAP — no tests)

**File:** `unit_tests/test_metadata_schemas.py` — NEW section `TestConceptTagsOutputValidateAndFix`

| Test | Why | Key assertions |
|------|-----|----------------|
| `test_deduplicates_tags` | LLMs occasionally repeat tags | Duplicate PLOT_TWIST in narrative_structure.tags -> deduplicated to 1 |
| `test_twist_villain_implies_plot_twist` | Deterministic implication rule | TWIST_VILLAIN without PLOT_TWIST -> PLOT_TWIST added |
| `test_twist_villain_no_duplicate_when_plot_twist_present` | Don't double-add | Both present -> no duplicate |
| `test_validate_and_fix_roundtrip` | Full pipeline | JSON -> validate_and_fix -> both fixups applied |
| `test_all_concept_tag_ids_filters_no_clear_choice` | NO_CLEAR_CHOICE has id=-1 | endings.tag=NO_CLEAR_CHOICE -> not in all_concept_tag_ids() |
| `test_all_concept_tag_ids_includes_positive_ending` | Positive ending tags stored | endings.tag=HAPPY_ENDING -> 41 in result |
| `test_all_concept_tag_ids_sorted_and_unique` | Contract | Result is sorted, no duplicates |
| `test_all_concept_tag_ids_empty_when_no_tags` | Edge case | All empty assessments + NO_CLEAR_CHOICE ending -> [] |

### 2c. ProductionTechniquesOutput (GAP — no schema validation tests)

**File:** `unit_tests/test_metadata_schemas.py` — NEW section

| Test | Why | Key assertions |
|------|-----|----------------|
| `test_extra_forbid` | Structural guard | Extra fields rejected |
| `test_empty_terms_valid` | Selective classifier may return nothing | `terms=[]` accepted |
| `test_is_embeddable_subclass` | Contract | subclass of EmbeddableOutput |
| `test_embedding_text_normalizes` | Consistency | Uses normalize_string per-term |
| `test_str_lowercases` | Consistency | str() returns lowercased comma-separated |

### 2d. Stale parity tests referencing deleted classes (UPDATE)

**File:** `unit_tests/test_metadata_schemas.py`

The following classes were removed from `schemas/metadata.py`:
- `WatchContextWithJustificationsOutput`
- `NarrativeTechniquesWithJustificationsOutput`
- `ProductionKeywordsWithJustificationsOutput`
- `SourceOfInspirationWithJustificationsOutput`
- `SourceOfInspirationWithReasoningOutput`
- `OptionalTermsWithNegationsSection`

Tests that import/use them will fail at import time:
- `TestWatchContextWithJustificationsStrParity` (lines 477-502) — **DELETE**
- `TestNarrativeTechniquesWithJustificationsStrParity` (lines 520-542) — **DELETE**
- `TestProductionKeywordsWithJustificationsStrParity` (lines 549-563) — **DELETE**
- `TestSourceOfInspirationWithJustificationsStrParity` (lines 570-592) — **DELETE**
- `TestSourceOfInspirationPromptAliasRemoval` (lines 788-795) — **DELETE** (no longer relevant)
- `TestSourceOfInspirationWithReasoningEvidenceConstraints` (lines 798-808) — **DELETE**
- `TestViewerExperienceOutputStrContent.test_viewer_experience_rejects_optional_wrapper` (line 664) — **DELETE** (references deleted `OptionalTermsWithNegationsSection`)

### 2e. PlotAnalysisOutput embedding_text() label format (UPDATE)

**File:** `unit_tests/test_metadata_embedding_text.py`

`TestPlotAnalysisEmbeddingText.test_has_labels` asserts `"character arcs:"` and
`"genre signatures:"` (with spaces). Current code uses snake_case labels:
`"character_arcs:"`, `"genre_signatures:"`.

| Test | Change needed |
|------|--------------|
| `test_has_labels` | Update assertions to `"genre_signatures:"`, `"conflict:"`, `"character_arcs:"`, `"themes:"` |

**File:** `unit_tests/test_metadata_embedding_text.py`

`TestEmbeddingTextVsStr.test_plot_analysis_diverges` asserts that
`"genre signatures:"` is in `embedding_text()` but not in `str()`. Now `__str__()`
delegates to `embedding_text()`, so both have it.

| Test | Change needed |
|------|--------------|
| `test_plot_analysis_diverges` | **DELETE or REWRITE** — `__str__()` now delegates to `embedding_text()`, so they don't diverge. Replace with a test that confirms delegation (`str(output) == output.embedding_text()`). |

---

## 3. schemas/movie.py

### 3a. Stale tests for removed methods (UPDATE)

**File:** `unit_tests/test_schemas_movie.py`

- `TestTitleWithOriginal` (lines 176-187) — `title_with_original()` was removed. **DELETE entire class.**
- `TestProductionText` (lines 406-447) — `production_text()` was removed. **DELETE entire class.**

### 3b. New methods missing tests (NEW)

**File:** `unit_tests/test_schemas_movie.py` — NEW sections

| Method | Tests needed | Key assertions |
|--------|-------------|----------------|
| `box_office_status()` | Already has tests (TestBoxOfficeStatus, lines 283-375). **No change needed.** | |
| `source_material_type_ids()` | NEW: returns IDs from `source_material_v2_metadata`; empty when None | `[1,4]` for NOVEL_ADAPTATION+TRUE_STORY; `[]` when metadata is None |
| `keyword_ids()` | NEW: maps IMDB overall_keywords to int IDs via `keyword_from_string()` | Known keywords produce correct IDs; unknown keywords skipped; deduplicates |
| `concept_tag_ids()` | NEW: delegates to `concept_tags_metadata.all_concept_tag_ids()` | Returns sorted IDs; `[]` when metadata is None |
| `award_ceremony_win_ids()` | Already has tests in `test_ingest_movie.py`. **No change needed.** | |

### 3c. New Movie fields need fixture support (UPDATE)

**File:** `unit_tests/test_schemas_movie.py` — UPDATE `_make_movie` helper

The helper may need to handle new metadata fields (`franchise_metadata`,
`concept_tags_metadata`, `production_techniques_metadata`) for tests that
explicitly set them. Verify the defaults (None) work without change.

---

## 4. schemas/movie_input.py

### 4a. top_billed_cast() (GAP — no tests)

**File:** `unit_tests/test_metadata_inputs.py` — NEW section `TestTopBilledCast`

| Test | Why | Key assertions |
|------|-----|----------------|
| `test_returns_none_when_no_actors` | Explicit absence signal | actors=[] -> None |
| `test_pairs_actors_with_characters` | Core behavior | actors=["A","B"], characters=["X","Y"] -> "X (A), Y (B)" |
| `test_actor_without_character` | Shorter characters list | actors=["A","B"], characters=["X"] -> "X (A), B" |
| `test_limits_to_n` | Default n=5 | 10 actors -> only first 5 in output |
| `test_skips_blank_actor` | Empty actor names filtered | actors=["A","","C"] -> skips blank |
| `test_none_vs_empty_string_distinction` | Callers must handle None | Verify None return type, not "" |

---

## 5. movie_ingestion/final_ingestion/vector_text.py

### 5a. Anchor vector text (UPDATE — rewritten)

**File:** `unit_tests/test_vector_text.py` — UPDATE `TestAnchorVectorText`

The anchor vector builder was rewritten. Several tests assert stale labels/fields:

| Test | Issue | Action |
|------|-------|--------|
| `test_full_movie` | Asserts `"genres:" in result` — now `"genre_signatures:"`. Asserts `"source material:"` — now excluded from anchor. Asserts `"emotional palette:"` — now `"emotional_palette:"`. Asserts `"key draws:"` — now `"key_draws:"`. Asserts `"reception:"` — now `"reception_summary:"`. | **UPDATE** all label assertions |
| `test_uses_normalize_string_for_terms` | Asserts keywords appear — keywords are excluded from anchor now. | **REWRITE** to test a field that IS normalize_string'd (genre_signatures or themes) |
| `test_all_optional_metadata_none` | Asserts `"genres:"` and `"keywords:"` — both stale. | **UPDATE** assertions; without metadata, only title and overview appear |
| `test_includes_source_material` | Source material excluded from anchor. | **DELETE** |
| `test_falls_back_to_overview` | Label is now `"identity_overview:"` not just content appearing. | **UPDATE** assertion |
| `test_includes_reception` | Label now `"reception_summary:"` not `"reception:"`. | **UPDATE** assertion |

NEW tests needed:
| Test | Why | Key assertions |
|------|-----|----------------|
| `test_includes_original_title_when_different` | New field in anchor | original_title present and different -> `"original_title:"` in result |
| `test_excludes_original_title_when_same` | Edge case | original_title == title -> no `"original_title:"` line |
| `test_includes_identity_pitch` | New field | plot_analysis_metadata with pitch -> `"identity_pitch:"` in result |

### 5b. Plot analysis vector text (UPDATE — TMDB genre merge removed)

**File:** `unit_tests/test_vector_text.py` — UPDATE `TestPlotAnalysisVectorText`

| Test | Issue | Action |
|------|-------|--------|
| `test_merges_imdb_genres` | TMDB genre merge removed — now a thin wrapper. | **DELETE** |
| `test_deduplicates_genres` | Same — no genre merging. | **DELETE** |
| `test_genre_mutation_does_not_corrupt_source` | Same — no genre merging. | **DELETE** |
| `test_returns_none_without_metadata` | Still valid. | **KEEP** |

NEW:
| Test | Assertions |
|------|-----------|
| `test_delegates_to_embedding_text` | Result equals `metadata.embedding_text()` exactly |

### 5c. Production vector text (UPDATE — completely rewritten)

**File:** `unit_tests/test_vector_text.py` — UPDATE `TestProductionVectorText`

The production vector now carries only filming locations (non-animation) plus
production_techniques. All other fields (production medium, source material,
production keywords, country, company, language, decade, budget) were removed.

| Test | Issue | Action |
|------|-------|--------|
| `test_excludes_filming_locations_for_animation` | Still valid but label changed. | **UPDATE** assertion to `"filming_locations:"` |
| `test_includes_filming_locations_for_live_action` | Label changed. | **UPDATE** assertion to `"filming_locations:"` |
| `test_production_medium_animation` | Removed from production. | **DELETE** |
| `test_production_medium_live_action` | Removed from production. | **DELETE** |
| `test_default_original_screenplay` | Source material removed. | **DELETE** |
| `test_uses_source_embedding_text` | Source material removed. | **DELETE** |
| `test_includes_production_keywords` | Replaced by production_techniques. | **DELETE** |

NEW:
| Test | Assertions |
|------|-----------|
| `test_returns_none_when_no_data` | No locations (animation) + no techniques -> None |
| `test_includes_production_techniques` | production_techniques_metadata with terms -> `"production_techniques:"` in result |
| `test_production_techniques_only` | Animation movie with techniques but no locations -> result has only techniques |
| `test_filming_locations_only` | Live action with locations but no techniques -> result has only locations |
| `test_filming_locations_limited_to_3` | Only first 3 locations included |

### 5d. Reception vector text — award wins (GAP — no tests)

**File:** `unit_tests/test_vector_text.py` — NEW section `TestReceptionAwardWins`

`_reception_award_wins_text()` is new functionality with non-trivial logic:
prestige ordering, Razzie exclusion, ceremony deduplication.

| Test | Why | Key assertions |
|------|-----|----------------|
| `test_includes_award_wins_in_reception_text` | Core behavior | Academy Awards win -> `"major_award_wins: academy awards"` in result |
| `test_excludes_razzie` | Razzie deliberately excluded | Razzie win only -> no `"major_award_wins:"` line |
| `test_excludes_nominees` | Only wins included | Nominee-only awards -> no `"major_award_wins:"` line |
| `test_deduplicates_ceremonies` | Multiple wins in same ceremony -> single entry | Two Oscar wins -> `"academy awards"` appears once |
| `test_prestige_ordering` | Display order | Academy Awards + Sundance -> academy awards before sundance |
| `test_unknown_ceremony_skipped` | Graceful handling | Unknown ceremony string -> skipped |
| `test_no_awards_returns_none` | Edge case | Empty awards list -> None from helper |

---

## 6. schemas/imdb_models.py (NEW file)

### 6a. AwardNomination methods (GAP — partial coverage)

**File:** `unit_tests/test_schemas_movie.py` already has `TestAwardNominationCeremonyId`.
`did_win()` is trivial but untested.

| Test | Where | Assertions |
|------|-------|-----------|
| `test_did_win_true_for_winner` | NEW in test_schemas_movie.py or test_enums.py | `AwardNomination(outcome=WINNER).did_win() is True` |
| `test_did_win_false_for_nominee` | Same | `AwardNomination(outcome=NOMINEE).did_win() is False` |

---

## 7. unit_tests/conftest.py

### 7a. base_movie_factory missing new fields (POTENTIAL UPDATE)

The `base_movie_factory` builds `BaseMovie` (implementation/classes/movie.py),
not the new `Movie` (schemas/movie.py). If `BaseMovie` gained new fields
(franchise_metadata, concept_tags_metadata, etc.), the factory needs defaults.
Verify whether `BaseMovie` changed — if not, no update needed.

---

## Priority ordering

**P0 — Tests that will fail (stale tests referencing deleted code):**
1. test_metadata_schemas.py deleted class imports (2d)
2. test_vector_text.py stale anchor assertions (5a)
3. test_vector_text.py stale plot_analysis genre merge tests (5b)
4. test_vector_text.py stale production vector tests (5c)
5. test_schemas_movie.py deleted method tests (3a)
6. test_metadata_embedding_text.py stale label assertions (2e)

**P1 — Missing tests for new logic with non-trivial correctness concerns:**
1. FranchiseOutput.validate_and_fix() (2a) — 3 fixup rules, 10 test cases
2. ConceptTagsOutput.validate_and_fix() and all_concept_tag_ids() (2b) — dedup, implication, filtering
3. Reception award wins text (5d) — prestige ordering, Razzie exclusion
4. top_billed_cast() (4a) — new method with pairing/truncation logic

**P2 — Missing stability tests for persisted IDs:**
1. LineagePosition enum (1b)
2. Movie.source_material_type_ids(), keyword_ids(), concept_tag_ids() (3b)

**P3 — Minor gaps:**
1. AwardNomination.did_win() (6a)
2. ProductionTechniquesOutput validation (2c)
