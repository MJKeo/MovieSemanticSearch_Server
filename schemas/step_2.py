# Query analysis output schema.
#
# A single LLM call produces two coupled outputs:
# 1. holistic_read: a faithful prose read of the query in the
#    user's own words, describing what they're asking for and how
#    the pieces of that ask affect each other.
# 2. atoms: the query's evaluative criteria, each with the
#    user's words for it (surface_text), every signal in the query
#    that shapes how it should be evaluated (modifying_signals),
#    and a concise prose statement of what evaluating it actually
#    means once that context is integrated (evaluative_intent).
#
# The prose read precedes the atoms and grounds them. surface_text
# and modifying_signals stay strictly descriptive — recording what
# is in the query, not committing to downstream interpretations
# (polarity, salience, category, search strategy). evaluative_intent
# is the one place where light inference is permitted, because the
# whole point of that field is consolidating context into meaning.
#
# Design principles:
# - Minimum context, maximum looseness. Queries are freeform; we
#   record raw signal and let the LLM consolidate per-criterion.
# - One unified place for modifier signals. Whether a modifier sat
#   adjacent to the criterion in the surface text or came from
#   another part of the query, it lands on the same list.
# - Effects are described, not categorized. surface_phrase + a
#   freeform effect string carries every modifier shape; no closed
#   enum to bucket-force into.
# - Intent is the load-bearing semantic field. Downstream consumes
#   evaluative_intent for evaluation; modifying_signals is provenance.
#
# This schema is the only documentation the LLM gets for the
# output shape — the system prompt does NOT duplicate this content.
# Field descriptions are micro-prompts.

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------
# Modifying signal
# ---------------------------------------------------------------------


class ModifyingSignal(BaseModel):
    """One signal from the query that shapes how an atom is
    evaluated. Adjacent qualifiers and cross-criterion modifiers
    live on the same list — conceptually the same thing
    (something-shaping-this-criterion's-meaning)."""

    surface_phrase: str = Field(
        ...,
        description=(
            "The user's exact words for this signal — verbatim "
            "substring of the query. For adjacent qualifiers "
            "(hedges, polarity words, role markers, range words), "
            "just that phrase. For signals from elsewhere in the "
            "query (scoping/transposing another criterion), the "
            "connecting language plus the reference, in the user's "
            "words. Never a positional pointer or index."
        ),
    )
    effect: str = Field(
        ...,
        description=(
            "Concise description of what this signal does to the "
            "atom's evaluation. A few words to a short phrase, not "
            "a sentence.\n"
            "\n"
            "DESCRIBE the effect; don't categorize the signal. "
            "Example flavors (NOT a closed list — use your own "
            "words when none fit):\n"
            "- 'softens the requirement' (hedge)\n"
            "- 'hardens the requirement' (intensifier)\n"
            "- 'flips polarity' (negation)\n"
            "- 'contrasts with the prior want' (pivot)\n"
            "- 'binds to director credit' / 'binds to acting credit'\n"
            "- 'calibrates upper bound' / 'narrows to early portion'\n"
            "- 'applies as comparison reference'\n"
            "- 'transposes setting to a period'\n"
            "- 'narrows to a subset'\n"
            "- 'scopes to a specific subject'\n"
            "- 'used as style reference, not credit'\n"
            "\n"
            "For modal language, the controlled vocabulary SOFTENS "
            "/ HARDENS / FLIPS POLARITY / CONTRASTS is the "
            "recommended phrasing. Not enforced — describe what "
            "the modifier actually does when it doesn't fit those."
        ),
    )


# ---------------------------------------------------------------------
# Atom
# ---------------------------------------------------------------------


class Atom(BaseModel):
    """One criterion the user wants movies scored against, at the
    granularity they phrased it, plus its consolidated meaning in
    the query's full context."""

    surface_text: str = Field(
        ...,
        description=(
            "Exact substring of the original query, with modifying "
            "language stripped (preserved in modifying_signals).\n"
            "\n"
            "NEVER paraphrase, expand named things, or substitute "
            "system-shaped synonyms. The reference itself is the "
            "signal; what it evokes belongs to evaluative_intent."
        ),
    )
    modifying_signals: list[ModifyingSignal] = Field(
        default_factory=list,
        description=(
            "Every signal in the query that shapes how this "
            "criterion is evaluated.\n"
            "\n"
            "Signals can come from anywhere in the query — adjacent "
            "qualifying language (hedges, role markers, range words, "
            "polarity setters), polarity language elsewhere that "
            "distributes onto this criterion, or content phrases "
            "elsewhere that fundamentally reshape its evaluation "
            "(transposing setting / period / medium / style, scoping "
            "to a subset, supplying counterfactual context, narrowing "
            "inside a known referent). When a content phrase reshapes "
            "another atom this deeply, it absorbs as a signal here — "
            "it does NOT also appear as a separate atom. See the "
            "atomicity guidance in the system prompt for the boundary "
            "call.\n"
            "\n"
            "Surface-order position is irrelevant; if a phrase shapes "
            "this atom's evaluation, it goes here.\n"
            "\n"
            "One entry per signal: verbatim surface_phrase plus a "
            "concise effect string. No directional graph thinking, "
            "no positional pointers, no kind enum.\n"
            "\n"
            "Empty list is fine and common when criteria are "
            "genuinely independent. Don't fabricate signals to make "
            "atoms look connected."
        ),
    )
    evaluative_intent: str = Field(
        ...,
        description=(
            "Find everything in the query that modifies this "
            "criterion's meaning, then state — in 1-2 sentences of "
            "plain prose — its true evaluative intention. What does "
            "scoring movies on this criterion actually mean once "
            "that context is integrated?\n"
            "\n"
            "This is the ONE field where light inference is "
            "permitted. surface_text and modifying_signals stay "
            "strictly descriptive; here you consolidate them into "
            "meaning. A hedge softens → say so in evaluative terms "
            "('prefer X but tolerate departures'). A negation flips "
            "polarity → reflect the avoid-direction. A transposing "
            "criterion → describe the transposed evaluation. A "
            "criterion used as reference rather than target → say so.\n"
            "\n"
            "NEVER:\n"
            "- CATEGORIZE. No system-shaped labels ('genre', "
            "'runtime', 'actor', 'tone'). The user didn't think in "
            "those buckets.\n"
            "- COMMIT TO POLARITY/SALIENCE NUMBERS. Describe "
            "direction and weight in words; downstream assigns "
            "concrete values.\n"
            "- EXPAND NAMED THINGS. Don't unpack what a name 'evokes' "
            "or 'typically means' — the reference itself is the "
            "signal.\n"
            "- TRANSLATE INTO SYSTEM VOCABULARY. Don't pick a "
            "downstream channel / vector / endpoint — that's the "
            "next stages' call.\n"
            "- PARAPHRASE SURFACE_TEXT WHILE IGNORING SIGNALS. If "
            "modifying_signals is non-empty, the intent MUST reflect "
            "each signal's effect. Test: would the intent change "
            "noticeably if I removed or altered this signal? If no, "
            "you haven't integrated it — revise. (When signals is "
            "empty, plain description is correct.)"
        ),
    )
    candidate_internal_split: str | None = Field(
        default=None,
        description=(
            "Populated ONLY when genuinely uncertain whether this "
            "is one combined criterion or two distinct ones. Format: "
            "'<piece A> | <piece B>' using exact substrings of "
            "surface_text. Downstream verifies. Leave null when "
            "confident — guessing adds noise."
        ),
    )


# ---------------------------------------------------------------------
# Top-level response
# ---------------------------------------------------------------------


class QueryAnalysis(BaseModel):
    """Combined output of the query analysis stage: a faithful
    prose read of the query plus per-criterion atoms with
    consolidated evaluative intent."""

    holistic_read: str = Field(
        ...,
        description=(
            "Faithfully describe what the user is asking for, in "
            "their own words. Describe how query pieces affect each "
            "other only as far as the user themselves implied. "
            "Downstream phases commit to structure; your job here "
            "is description.\n"
            "\n"
            "NEVER:\n"
            "- CATEGORIZE. No system-shaped labels ('genre', "
            "'runtime', 'actor', 'tone'). The user didn't think in "
            "those buckets.\n"
            "- EXPAND NAMED THINGS. The reference stays as written. "
            "Don't unpack what a name 'evokes' or 'typically means' "
            "— the reference itself is the signal.\n"
            "- INFER beyond what the user said. No 'i.e.', 'such "
            "as', 'meaning', or parentheticals explaining what the "
            "user 'really' meant. Loose terms stay loose — looseness "
            "is part of what they said.\n"
            "- IMPOSE STRUCTURE THAT ISN'T THERE. Parallel wants are "
            "parallel — don't pick a 'primary'. Don't apply "
            "structural labels ('kept whole', 'anchor', 'hybrid', "
            "'cross') unless the user's phrasing puts that "
            "relationship on the page.\n"
            "\n"
            "DO:\n"
            "1. List the wants the user named, in exact phrasing. "
            "Every high-value semantic word appears unchanged.\n"
            "2. For each piece of modal language attached to a want, "
            "name its effect verbatim: SOFTENS (hedges: 'ideally', "
            "'kinda', 'maybe', 'preferably', 'a bit'), HARDENS "
            "(intensifiers: 'really', 'must', 'above all', 'need'), "
            "FLIPS POLARITY (negations: 'not', 'without', 'no', "
            "'avoid', 'skip'), CONTRASTS (pivots: 'but', 'though', "
            "'yet' turning to a contrasting want). The phrase alone "
            "isn't the signal; the named effect is.\n"
            "3. Describe how the wants relate, only as plainly as "
            "the user implied. Independent → say so. One modifies "
            "another → say which. Mutually dependent → say that. "
            "Use the user's framing; don't reach for vocabulary they "
            "didn't use.\n"
            "\n"
            "VOICE: plain prose, no enum-style labels. Length scales "
            "with the query's structure — single-want is one line, "
            "multi-want is more. Don't pad."
        ),
    )
    atoms: list[Atom] = Field(
        ...,
        description=(
            "The query's evaluative criteria. Each atom is one "
            "criterion the user wants scored, at the granularity "
            "they phrased it.\n"
            "\n"
            "WHEN ONE, WHEN MULTIPLE: a compound stays whole when "
            "the pieces aren't separately evaluable — when only the "
            "whole names what the user is judging on, and splitting "
            "loses their actual ask. Otherwise: distinct evaluative "
            "criteria → distinct atoms, even within one surface "
            "phrase.\n"
            "\n"
            "Each atom carries three load-bearing fields: "
            "surface_text (verbatim user words), modifying_signals "
            "(everything shaping this criterion's evaluation), "
            "evaluative_intent (consolidated meaning). See each "
            "field for discipline.\n"
            "\n"
            "NEVER:\n"
            "- PARAPHRASE surface_text. Exact substring of the "
            "query.\n"
            "- PROMOTE MODIFIERS TO ATOMS. Hedges, polarity setters, "
            "role markers, range words, comparison frames absorb "
            "into their host atom as modifying_signals.\n"
            "- MERGE INDEPENDENTLY-EVALUABLE CRITERIA. When parts "
            "each name real populations whose intersection is the "
            "user's intent, they're distinct atoms. (Opposite case "
            "— pieces that don't retrieve independently — collapses "
            "to ONE atom under the atomicity test in the system "
            "prompt, with the dependent part absorbed as a "
            "modifying_signal.)\n"
            "- LET ABSORBED MATERIAL APPEAR TWICE. When a phrase is "
            "recorded as a modifying_signal, both the phrase AND "
            "the concept inside it (setting, period, medium, style, "
            "named referent, mood) are claimed by that signal. "
            "Neither the phrase nor any bare concept word from it "
            "becomes a separate atom. Consolidated meaning lives on "
            "the host atom's evaluative_intent.\n"
            "- COMMIT to category, polarity, salience, search "
            "strategy, or weight at the structural level. Light "
            "inference allowed only inside evaluative_intent.\n"
            "\n"
            "ORDERING: atoms appear in surface-text order from the "
            "original query. Order is load-bearing downstream."
        ),
    )
