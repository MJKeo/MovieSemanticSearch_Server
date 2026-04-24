# Search V2 — Step 2 (Query Pre-pass) output schema.
#
# Every requirement fragment is an attribute: a content-bearing chunk
# of the query that carries one or more category-grounded atoms.
# Polarity modifiers ("not", "not too", "preferably") and role
# markers ("starring", "directed by", "about") are NOT their own
# fragments — they are nested inside the attribute fragment they bind
# to, as entries in the modifiers list. This keeps each requirement
# self-contained so it can be dispatched to its category handler
# without cross-fragment reassembly.
#
# Field order within CoverageEvidence is observations-first:
#   1. captured_meaning — what atom is being observed (before
#      committing to a category label).
#   2. category_name — the category that covers the observed atom.
#   3. fit_quality — clean vs partial vs no_fit.
#   4. atomic_rewrite — the atom expressed as a category-grounded
#      request, preserving specifics from the original.

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from schemas.enums import CategoryName, FitQuality, LanguageType


class Modifier(BaseModel):
    """A polarity or role-marker phrase that attaches to an attribute
    fragment and shapes how that attribute is interpreted.
    """

    original_text: str = Field(
        ...,
        description=(
            "The verbatim span from the query that produced this "
            "modifier (e.g. 'starring', 'not too', 'directed by', "
            "'preferably'). Preserve wording and typos."
        ),
    )
    effect: str = Field(
        ...,
        description=(
            "One brief phrase stating how this modifier changes the "
            "adjacent attribute — the binding it creates or the "
            "sign/strength shift it applies. Written as a terse note "
            "for the downstream LLM, not a full sentence."
        ),
    )
    type: LanguageType = Field(
        ...,
        description=(
            "POLARITY_MODIFIER if this modifier flips or modulates "
            "the sign/strength of the attribute ('not', 'not too', "
            "'without', 'preferably', 'ideally'). ROLE_MARKER if it "
            "binds the attribute to a specific role or dimension "
            "('starring', 'directed by', 'about', 'set in', 'based "
            "on')."
        ),
    )


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
    """A contiguous chunk of the query that conveys one attribute of
    the desired results. Every fragment is an attribute; polarity
    modifiers and role markers that bind to it live in the modifiers
    list. Preserves the user's exact wording.
    """

    query_text: str = Field(
        ...,
        description=(
            "The attribute span exactly as it appears in the query, "
            "preserving wording and typos. Do not paraphrase. Any "
            "adjacent role markers or polarity modifiers are NOT "
            "included here — they go in the modifiers list."
        ),
    )
    description: str = Field(
        ...,
        description=(
            "One sentence describing what this fragment contributes "
            "to the query."
        ),
    )
    modifiers: List[Modifier] = Field(
        ...,
        description=(
            "Polarity modifiers and role markers that attach to "
            "this attribute. Empty list when the fragment has none. "
            "Each entry carries the verbatim span, a brief effect "
            "note, and the modifier type (POLARITY_MODIFIER or "
            "ROLE_MARKER)."
        ),
    )
    coverage_evidence: List[CoverageEvidence] = Field(
        ...,
        description=(
            "One entry per category-grounded atom this fragment "
            "contains. Simple one-axis fragments produce one entry; "
            "compound descriptors and multi-dimension entities "
            "produce multiple entries (one per implied atom)."
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
            "Every attribute-bearing chunk of the query, with "
            "original wording preserved. Ignore pure filler "
            "('movies', 'films', 'help me find'). Role markers "
            "('starring', 'directed by') and polarity modifiers "
            "('not', 'not too') are NOT their own fragments — they "
            "attach to the adjacent attribute fragment as entries "
            "in its modifiers list."
        ),
    )
