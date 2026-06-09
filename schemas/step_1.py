# Search V2 — Step 1 (Spin Generation) output schema.
#
# Step 1 runs in parallel with step 0 on the raw user query. It
# treats every input as if it will run through the standard flow
# and produces two creative "spins" — alternative queries that
# broaden the user's browsing without straying from what they meant.
#
# The schema is minimal but cognitively-scaffolded: the model
# first surfaces its reasoning visibly in an `exploration` field
# (a freeform brainstorm of alternative search directions worth
# considering for this query), then commits to the two spins.
# Earlier iterations forced the model to dissect the query into
# pre-defined slots (hard commitments / soft interpretations /
# open dimensions) before reconstructing spins, which biased it
# toward single-token tweaks that collapsed back onto the
# original's result set. Replacing the slotted decomposition with
# a freeform pre-generation scratchpad lets the model interpret
# language holistically.
#
# Per the "lean fields + system prompt teaches" convention, field
# descriptions stay compact: the how-to-think guidance lives in
# search_v2/step_1.py's system prompt, not duplicated here.

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


# One alternative search the user didn't think to type. Carries
# the spin's query text and the UI label that surfaces it in the
# browsing interface.
class Spin(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Full natural-language search phrase, the kind of "
            "thing the user could have typed themselves. Never "
            "names specific movie or show titles. Brand, studio, "
            "director, or actor names are allowed only when they "
            "are the central pivot of the search, never as an "
            "enumeration of examples. Max 150 characters."
        ),
        max_length=150,
    )
    ui_label: str = Field(
        ...,
        description=(
            "Short Title Case label for the browsing UI. Pithy "
            "enough to read at a glance but not so compressed it "
            "loses meaning. Max 50 characters."
        ),
        max_length=50,
    )


# Structured output for the step-1 spin generation step. The
# exploration field is a freeform brainstorm that scaffolds the
# two committed spins below it.
class Step1Response(BaseModel):
    exploration: str = Field(
        ...,
        description=(
            "Reasoning scratchpad: identify what the user is "
            "really after, the adjacent searches that would also "
            "serve them, and which candidates avoid overlap with "
            "the original query's result set. 2-3 compact, "
            "telegraphic sentences."
        ),
    )
    spins: List[Spin] = Field(
        ...,
        description=(
            "Exactly two alternative searches, each refining one "
            "of the candidate angles surfaced in exploration. The "
            "two spins must surface visibly different result sets "
            "from each other and from the original query."
        ),
        min_length=2,
        max_length=2,
    )


# Structured output for the step-1 clarification-mode call. Fires
# only when the user has supplied a follow-up clarification on top
# of their original query. Adds a main_rewrite slot above the spins:
# main_rewrite is a faithful merge of original + clarification that
# replaces the verbatim-original slot in the branch plan; spins keep
# their creative role but now explore around the rewritten intent.
class Step1ClarificationResponse(BaseModel):
    exploration: str = Field(
        ...,
        description=(
            "Reasoning scratchpad. First read how the clarification "
            "reshapes the original — what it adds, retracts, or "
            "polarity-flips — then sketch the rewritten intent in "
            "plain words, then surface adjacent searches the same "
            "viewer might also want. 2-3 compact telegraphic "
            "sentences."
        ),
    )
    main_rewrite: Spin = Field(
        ...,
        description=(
            "The merged search representing the user's most likely "
            "intent given the original query plus the follow-up "
            "clarification, expressed as a natural-language search "
            "the user could have typed themselves."
        ),
    )
    spins: List[Spin] = Field(
        ...,
        description=(
            "Exactly two creative alternative searches that explore "
            "adjacent territory the rewritten intent (NOT the raw "
            "original) would otherwise miss. Each spin's result set "
            "must be visibly different from main_rewrite and from "
            "the sibling spin."
        ),
        min_length=2,
        max_length=2,
    )
