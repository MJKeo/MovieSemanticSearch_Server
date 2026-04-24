# Search V2 — Step 2 (Query Pre-pass) output schema.
#
# Single-pass structure: every requirement goes through the same
# decomposition. There is no separate "problematic" list — atoms that
# need unpacking show up as multi-entry coverage_evidence with
# per-category atomic_rewrites, composed into full_rewrite.
#
# Field order within CoverageEvidence is observations-first:
#   1. captured_meaning — what atom is being observed (before
#      committing to a category label).
#   2. category_name — the category that covers the observed atom.
#   3. fit_quality — clean vs partial vs no_fit.
#   4. atomic_rewrite — the atom expressed as a category-grounded
#      request, preserving specifics from the original.

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

from schemas.enums import CategoryName


LanguageType = Literal[
    "attribute",
    "selection_rule",
    "role_marker",
    "polarity_modifier",
]

FitQuality = Literal["clean", "partial", "no_fit"]


class CoverageEvidence(BaseModel):
    """One category-grounded atom extracted from a fragment."""

    captured_meaning: str = Field(
        ...,
        description=(
            "One short sentence stating what aspect of the "
            "fragment's meaning this entry captures — written "
            "BEFORE naming the category. State the observation in "
            "neutral terms (what the user is asking for along some "
            "dimension), not in terms of a specific category label. "
            "This is the evidence that justifies the category pick "
            "in the next field. If another fragment in the query "
            "definitionally rules out one reading of an ambiguous "
            "term, note the ruled-out reading here (e.g. 'franchise "
            "membership; the character reading is ruled out by the "
            "spinoff qualifier')."
        ),
    )
    category_name: CategoryName = Field(
        ...,
        description=(
            "The category whose concept definition covers the "
            "captured_meaning above. Choose from the taxonomy "
            "enum. If the meaning is real but no structured "
            "category fits, use Interpretation-required. If the "
            "captured_meaning turned out to be speculative or "
            "empty, use Interpretation-required with "
            "fit_quality='no_fit'."
        ),
    )
    fit_quality: FitQuality = Field(
        ...,
        description=(
            "'clean' = the category's concept definition squarely "
            "covers captured_meaning. 'partial' = the category "
            "covers part of captured_meaning but the rest is "
            "handled elsewhere (typically another entry in this "
            "coverage_evidence list). 'no_fit' = no category, "
            "including Interpretation-required, captures the "
            "captured_meaning — the observation was speculative "
            "and downstream should discard this entry. Prefer "
            "Interpretation-required with 'clean' over 'no_fit' "
            "whenever the meaning is real."
        ),
    )
    atomic_rewrite: str = Field(
        ...,
        description=(
            "A concise phrase that expresses the captured_meaning "
            "as a request this category can handle. Must preserve "
            "specific details from the original query — do NOT "
            "generalize ('brother' must not become 'sibling', "
            "'starring' must not become 'featuring', '1990s' must "
            "not become 'older'). Preserve polarity ('not too X' "
            "stays as 'with a preference against X')."
        ),
    )


class RequirementFragment(BaseModel):
    """A contiguous chunk of the query that conveys one thing about
    the desired results. Preserves the user's exact wording.
    """

    query_text: str = Field(
        ...,
        description=(
            "The fragment exactly as it appears in the query, "
            "preserving wording and typos. Do not paraphrase."
        ),
    )
    description: str = Field(
        ...,
        description=(
            "One sentence describing what this fragment contributes "
            "to the query. For attributes, describe the trait. For "
            "selection rules, describe the ordering/filtering. For "
            "role markers, describe the binding they create. For "
            "polarity modifiers, describe the sign/strength change."
        ),
    )
    type: LanguageType = Field(
        ...,
        description=(
            "One of: attribute, selection_rule, role_marker, "
            "polarity_modifier. See the prompt for definitions."
        ),
    )
    coverage_evidence: List[CoverageEvidence] = Field(
        ...,
        description=(
            "For attribute fragments: one entry per category-"
            "grounded atom this fragment contains. Simple one-axis "
            "fragments produce one entry; compound fragments "
            "produce multiple entries (one per implied atom). "
            "For non-attribute fragments (selection_rule, "
            "role_marker, polarity_modifier), this list is empty — "
            "those fragments modify adjacent attributes rather "
            "than standing alone in the taxonomy."
        ),
    )
    full_rewrite: str = Field(
        ...,
        description=(
            "The fragment rewritten as the sum of its "
            "atomic_rewrites, smoothed into one readable phrase. "
            "When coverage_evidence is empty (non-attribute "
            "fragments), use query_text verbatim or minimally "
            "smoothed. Every atomic_rewrite from an entry with "
            "fit_quality in ('clean', 'partial') must appear as a "
            "recognizable part of full_rewrite — do not drop atoms. "
            "Entries with fit_quality='no_fit' are discardable and "
            "do NOT need to appear in full_rewrite."
        ),
    )


class Step2Response(BaseModel):
    """Structured output for the step-2 pre-pass categorization step."""

    overall_query_intention_exploration: str = Field(
        ...,
        description=(
            "2-4 sentences describing what the query as a whole is "
            "asking for, including any overarching framing "
            "(occasion, audience, mood) that colors the specific "
            "requirements."
        ),
    )
    requirements: List[RequirementFragment] = Field(
        ...,
        description=(
            "Every requirement-bearing chunk of the query, with "
            "original wording preserved. Ignore pure filler "
            "('movies', 'films', 'help me find'). Role markers "
            "('starring', 'directed by', 'about') and polarity "
            "modifiers ('not', 'not too', 'without') get their OWN "
            "fragments — they are signal, not filler."
        ),
    )
    rewritten_query: str = Field(
        ...,
        description=(
            "The full query rewritten as the smoothed composition "
            "of all fragments' full_rewrites. Preserves original "
            "specificity. Adds no attributes, selection rules, "
            "roles, or polarity modifiers beyond what appears in "
            "the fragments."
        ),
    )
