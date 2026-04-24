# Search V2 — Step 1 (Spin Generation) output schema.
#
# Step 1 runs in parallel with step 0 on the raw user query. It
# treats every input as if it will run through the standard flow
# and produces two creative "spins" — adjacent but distinct
# searches the user might find interesting but wouldn't have typed
# themselves — plus a short UI label for the original query.
#
# Schema is observations-first: the three decomposition fields
# (hard_commitments, soft_interpretations, open_dimensions) are
# populated before any spin is committed to, so spins are grounded
# in a structured read of the query rather than free-associated.
#
# Field order within Spin is also observations-first:
#   1. branching_opportunity — the lever being pulled and why it's
#      worth exploring.
#   2. distinctness — how the resulting search differs from the
#      original AND the sibling spin, in retrieval terms.
#   3. query — the full spin expressed as a search phrase.
#   4. ui_label — short title-case UI label.

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class Spin(BaseModel):
    """One creative spin on the original query."""

    branching_opportunity: str = Field(
        ...,
        description=(
            "Names the specific lever this spin pulls — one item "
            "from either soft_interpretations or open_dimensions — "
            "and why that lever yields a branch worth showing the "
            "user. One or two sentences. Commit to exactly one "
            "lever; if the description bundles two, split into "
            "one spin each or drop one."
        ),
    )
    distinctness: str = Field(
        ...,
        description=(
            "How this spin's result set will differ from the "
            "original query AND from the sibling spin, in concrete "
            "retrieval terms (what movies appear here that would "
            "not appear in the others). Must pull on a different "
            "lever than the sibling. If both spins would retrieve "
            "largely overlapping lists, redesign one."
        ),
    )
    query: str = Field(
        ...,
        description=(
            "The spin expressed as a full search phrase, natural "
            "enough for step 2 to decompose. Preserve every item "
            "from hard_commitments verbatim; only the lever named "
            "in branching_opportunity's content changes."
        ),
    )
    ui_label: str = Field(
        ...,
        description=(
            "2-5 word Title Case label describing what this branch "
            "is about. Lean into the distinguishing lever so the "
            "label makes this spin's angle visible at a glance."
        ),
    )


class Step1Response(BaseModel):
    """Structured output for the step-1 spin generation step."""

    hard_commitments: List[str] = Field(
        ...,
        description=(
            "Things the user explicitly named that any faithful "
            "spin must preserve: actors, directors, franchises, "
            "studios, characters, explicit genres or formats, "
            "explicit eras, explicit platforms or constraints. "
            "Empty list if the query names nothing concrete."
        ),
    )
    soft_interpretations: List[str] = Field(
        ...,
        description=(
            "Words or phrases the user wrote that carry "
            "inferential lifting — evaluative words ('classics', "
            "'best'), mood/tone descriptors ('feel-good', 'cozy'), "
            "occasion/audience framings ('date night', 'millennial "
            "favorites'), vague scope words ('epic', "
            "'underrated'). Each is a candidate lever a spin can "
            "reinterpret. Empty list if none exist."
        ),
    )
    open_dimensions: List[str] = Field(
        ...,
        description=(
            "Axes the query does NOT touch at all — sub-genre, "
            "lead actor, decade, mood, tempo, format. Each is a "
            "candidate lever a spin can narrow along. Name each "
            "dimension concretely (not 'a genre' — 'sub-genre like "
            "thrillers'). Only list dimensions that would produce "
            "a useful sub-angle; 0-4 entries typical."
        ),
    )
    original_query_label: str = Field(
        ...,
        description=(
            "2-5 word Title Case UI label for the user's raw "
            "query. A clean, human-friendly summary of what the "
            "user literally asked for."
        ),
    )
    spins: List[Spin] = Field(
        ...,
        description=(
            "Exactly two spins. Each must pull on a different "
            "lever than the other — two narrowings along the same "
            "axis, or two near-synonymous reinterpretations of the "
            "same soft term, are design errors. The pair should "
            "together open up two visibly different browsing "
            "directions."
        ),
        min_length=2,
        max_length=2,
    )
