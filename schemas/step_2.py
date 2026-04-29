# Query analysis output schema.
#
# A single LLM call produces three coupled outputs:
# 1. holistic_read: a faithful prose read of the query in the
#    user's own words.
# 2. atoms: the descriptive layer — the query's evaluative criteria
#    as phrased, with the signals shaping each criterion's meaning
#    and a consolidated 1-2 sentence intent. Atoms record.
# 3. traits: the committed layer — search-ready units produced by
#    splitting / deduping atoms and assigning role, polarity, and
#    salience. Traits commit. Step 3 consumes traits, not atoms.
#
# The split keeps each layer's job tight and stops the model
# conflating description with prescription. surface_text and
# modifying_signals stay strictly descriptive on atoms;
# evaluative_intent is the one place light inference is permitted,
# because that field's whole purpose is consolidating context into
# meaning. Traits inherit intent from their source atom(s) and add
# the prescriptive commitments (role / polarity / salience).
#
# Design principles:
# - Minimum context, maximum looseness. Queries are freeform; we
#   record raw signal and let the LLM consolidate per-criterion.
# - One unified place for modifier signals. Adjacent and
#   cross-criterion modifiers land on the same list.
# - Effects are described, not categorized. surface_phrase + a
#   freeform effect string carries every modifier shape; controlled
#   modal vocabulary is recommended where downstream parses tokens.
# - Intent is the load-bearing semantic field. Downstream consumes
#   evaluative_intent for evaluation; modifying_signals is provenance.
# - Atoms describe; traits commit. Role/polarity/salience never
#   appear on atoms.
#
# This schema is the only documentation the LLM gets for the
# output shape — the system prompt does NOT duplicate field
# context. Field descriptions are micro-prompts: compact,
# information-dense, with NEVER lists where downstream depends on
# discipline.

from __future__ import annotations

from typing import Literal

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
            "Verbatim substring of the query. For adjacent qualifiers "
            "(hedges, polarity words, role markers, range words), "
            "just that phrase. For signals from elsewhere "
            "(scoping/transposing another criterion), the connecting "
            "language plus the reference, in the user's words. Never "
            "a positional pointer or index."
        ),
    )
    effect: str = Field(
        ...,
        description=(
            "What this signal does to the atom's evaluation, in a few "
            "words to a short phrase. Describe the effect; don't "
            "categorize the signal.\n"
            "\n"
            "Example flavors (NOT closed): 'softens the requirement', "
            "'hardens the requirement', 'flips polarity', 'contrasts "
            "with the prior want', 'binds to director credit', "
            "'calibrates upper bound', 'applies as comparison "
            "reference', 'transposes setting to a period', 'narrows "
            "to a subset', 'used as style reference, not credit'.\n"
            "\n"
            "For modal language, controlled vocabulary SOFTENS / "
            "HARDENS / FLIPS POLARITY / CONTRASTS is the recommended "
            "phrasing — the commit phase parses these tokens to "
            "assign polarity and salience. Not enforced; describe in "
            "plain words when no controlled term fits."
        ),
    )


# ---------------------------------------------------------------------
# Atom — descriptive layer
# ---------------------------------------------------------------------


class Atom(BaseModel):
    """One criterion the user wants movies scored against, at the
    granularity they phrased it, plus its consolidated meaning in
    the query's full context. Descriptive layer — atoms record;
    they do not commit to role / polarity / salience."""

    surface_text: str = Field(
        ...,
        description=(
            "Exact substring of the original query, with modifying "
            "language stripped (preserved in modifying_signals). "
            "NEVER paraphrase, expand named things, or substitute "
            "system-shaped synonyms. The reference itself is the "
            "signal; what it evokes belongs to evaluative_intent."
        ),
    )
    modifying_signals: list[ModifyingSignal] = Field(
        default_factory=list,
        description=(
            "Every signal in the query that shapes how this criterion "
            "is evaluated — adjacent qualifying language, polarity "
            "elsewhere distributing onto this criterion, or content "
            "phrases elsewhere that fundamentally reshape its "
            "evaluation (transposing setting/period/medium/style, "
            "scoping to a subset, supplying counterfactual context, "
            "narrowing inside a known referent).\n"
            "\n"
            "Surface-order position is irrelevant. When a content "
            "phrase reshapes another atom this deeply, it absorbs as "
            "a signal here — it does NOT also appear as a separate "
            "atom (see atomicity in the system prompt).\n"
            "\n"
            "Empty list is fine when criteria are genuinely "
            "independent. Don't fabricate signals to make atoms look "
            "connected."
        ),
    )
    evaluative_intent: str = Field(
        ...,
        description=(
            "1-2 sentences of plain prose stating this criterion's "
            "true evaluative intention once every modifying_signal is "
            "integrated. What does scoring movies on this criterion "
            "actually mean in the query's full context?\n"
            "\n"
            "The ONE field where light inference is permitted. A "
            "hedge softens → say so ('prefer X but tolerate "
            "departures'). A negation flips polarity → reflect the "
            "avoid-direction. A transposing modifier → describe the "
            "transposed evaluation. A reference (not target) → say so.\n"
            "\n"
            "NEVER:\n"
            "- CATEGORIZE. No system labels ('genre', 'runtime', "
            "'actor', 'tone').\n"
            "- COMMIT POLARITY/SALIENCE VALUES. Describe direction "
            "and weight in words; commitments live on traits.\n"
            "- EXPAND NAMED THINGS. The reference itself is the "
            "signal.\n"
            "- TRANSLATE INTO SYSTEM VOCABULARY. No downstream "
            "channel/vector/endpoint.\n"
            "- PARAPHRASE SURFACE_TEXT WHILE IGNORING SIGNALS. If "
            "modifying_signals is non-empty, intent MUST reflect each "
            "signal's effect. Test: would the intent change "
            "noticeably if I removed or altered this signal? If no, "
            "you haven't integrated it — revise. (Empty signals → "
            "plain description is correct.)"
        ),
    )
    split_exploration: str = Field(
        ...,
        description=(
            "Always populated. Pure evidence gathering — no verdict.\n"
            "\n"
            "Walk through whether this atom could be subdivided into "
            "smaller pieces, each retrievable independently. For each "
            "plausible subdivision, describe what each piece would "
            "retrieve on its own and whether the combined retrieval "
            "(intersection / joint scoring) would capture what the "
            "user is asking for at this atom's granularity. Describe "
            "the analysis only; the commit phase reads this and "
            "decides whether to split into traits.\n"
            "\n"
            "NEVER:\n"
            "- WRITE A VERDICT. No 'keep whole' / 'split into A and "
            "B' / 'not split because...' framings. Describe the "
            "retrieval shapes; the commit phase decides.\n"
            "- DISMISS THE QUESTION. Even atoms that obviously stay "
            "whole get analyzed (e.g. single-concept word: state that "
            "no smaller meaningful unit exists and why)."
        ),
    )
    standalone_check: str = Field(
        ...,
        description=(
            "Always populated. Pure evidence gathering — no verdict.\n"
            "\n"
            "Compare this atom's evaluative_intent against the user's "
            "articulated ask in the holistic_read. Describe HOW (not "
            "if) retrieving this atom standalone — alone, ignoring "
            "the other atoms — would relate to the user's articulated "
            "intent for its part of the query. Always describe; never "
            "dismiss with 'this is the only criterion' / 'no other "
            "atom captures this' / 'first mention of X'.\n"
            "\n"
            "Walk through:\n"
            "- What population would standalone retrieval of this "
            "atom return?\n"
            "- Does that population correspond to a constraint the "
            "user articulated as standalone-able, or does the "
            "standalone reading shift the meaning (introduce a hard "
            "requirement the user didn't ask for, lose a coupling the "
            "user did imply, narrow what the user kept loose)?\n"
            "- When this atom's evaluative_intent integrates context "
            "from another atom, is that context preserved in the "
            "standalone retrieval, or does it fall away?\n"
            "\n"
            "Reference other atoms by their surface_text when "
            "describing couplings.\n"
            "\n"
            "NEVER:\n"
            "- WRITE A VERDICT. No 'redundant given X' / 'not "
            "redundant' / 'standalone is fine'. Describe the "
            "relationship between standalone retrieval and "
            "user-articulated intent; the commit phase decides "
            "whether to merge.\n"
            "- SHORT-CIRCUIT WITH UNIQUENESS CHECKS. 'Primary subject "
            "of the query' / 'first mention' / 'no other atom "
            "captures this domain' / 'distinct concept' are not "
            "analyses — they're dismissals. Walk through what "
            "standalone retrieval would actually return.\n"
            "- APPEAL TO INDEPENDENT RETRIEVABILITY AS A VIRTUE. "
            "Standalone retrievability is not the goal — fidelity to "
            "user-articulated intent is. An atom that retrieves "
            "cleanly on its own can still distort meaning when its "
            "standalone population isn't what the user asked for.\n"
            "- USE 'WHILE [COUPLING ACKNOWLEDGED] BUT [STANDALONE "
            "VALUE]' PATTERNS. If the coupling exists, describe it; "
            "the existence of independent-retrieval value doesn't "
            "negate the meaning shift."
        ),
    )


# ---------------------------------------------------------------------
# Trait — committed layer
# ---------------------------------------------------------------------


class Trait(BaseModel):
    """One search-ready unit. Produced by the commit phase from
    atoms — splits resolved, redundancies deduped, role / polarity /
    salience committed. Step 3 consumes traits, not atoms."""

    surface_text: str = Field(
        ...,
        description=(
            "Verbatim phrase for this trait. Clean carry-over: the "
            "source atom's surface_text. After a split: the split "
            "substring. After a merge of duplicates: the clearer of "
            "the two source phrasings. Never paraphrase or expand."
        ),
    )
    evaluative_intent: str = Field(
        ...,
        description=(
            "1-2 sentence consolidated meaning, carried from the "
            "source atom(s). Clean carry-over copies the atom's "
            "intent verbatim. Split: re-state the relevant slice. "
            "Merge of duplicates: a single intent reflecting both "
            "sources' signals. Same NEVER list as "
            "Atom.evaluative_intent — no categorization, no system "
            "vocabulary, no concrete polarity / salience numbers."
        ),
    )
    role: Literal["carver", "qualifier"] = Field(
        ...,
        description=(
            "Read from the source atom's evaluative_intent shape. "
            "CARVER = population-defining; intent names a population "
            "the user wants narrowed to. QUALIFIER = shaping / "
            "reference; intent reads candidates against something "
            "rather than defining them.\n"
            "\n"
            "Operational test: if removing this trait from the "
            "candidate set would FILTER movies (yes/no exclusion), "
            "it's a carver. If it would DOWNRANK them (continuous, "
            "low-X movies still valid), it's a qualifier."
        ),
    )
    polarity: Literal["positive", "negative"] = Field(
        ...,
        description=(
            "Read source atom's modifying_signals. Any signal whose "
            "effect contains FLIPS POLARITY or recognizable negation "
            "language → negative. Otherwise positive. Hedges and "
            "intensifiers do NOT change polarity — they affect "
            "salience."
        ),
    )
    relevance_to_query: str = Field(
        ...,
        description=(
            "Reasoning step before salience. 1-2 sentences walking "
            "through how this trait sits in the query as a whole: "
            "hedges or intensifiers on the source atom, position in "
            "surface order (early/headline vs trailing), how many "
            "words the user spent on it, whether removing it would "
            "meaningfully change the ask.\n"
            "\n"
            "Read holistically. Modal effect tokens (SOFTENS, "
            "HARDENS) on modifying_signals are one signal but not "
            "the whole picture — within-query position and "
            "structural prominence contribute too. Salience commits "
            "as the natural conclusion of this reasoning.\n"
            "\n"
            "No system vocabulary; no concrete numbers."
        ),
    )
    salience: Literal["central", "supporting"] = Field(
        ...,
        description=(
            "Natural conclusion from relevance_to_query. CENTRAL = "
            "headline want; the query feels fundamentally different "
            "without this trait. SUPPORTING = meaningful but rounds "
            "out an already-defined ask rather than load-bearing.\n"
            "\n"
            "Applies to all traits regardless of role. A non-central "
            "carver acts as a lenient filter — the trait still "
            "defines its own pool but with softer boundaries; "
            "downstream code reads salience and adjusts."
        ),
    )


# ---------------------------------------------------------------------
# Top-level response
# ---------------------------------------------------------------------


class QueryAnalysis(BaseModel):
    """Combined output of the query analysis stage: a faithful prose
    read, descriptive atoms with consolidated evaluative intent,
    and committed search-ready traits."""

    holistic_read: str = Field(
        ...,
        description=(
            "Faithfully describe what the user is asking for, in "
            "their own words. Describe how query pieces affect each "
            "other only as far as the user themselves implied. "
            "Downstream phases commit; your job here is description.\n"
            "\n"
            "NEVER:\n"
            "- CATEGORIZE. No system labels ('genre', 'runtime', "
            "'actor', 'tone').\n"
            "- EXPAND NAMED THINGS. The reference stays as written; "
            "don't unpack what a name 'evokes' or 'typically means'.\n"
            "- INFER beyond what the user said. No 'i.e.', 'such as', "
            "'meaning', or parentheticals explaining what they "
            "'really' meant. Loose terms stay loose.\n"
            "- IMPOSE STRUCTURE THAT ISN'T THERE. Parallel wants are "
            "parallel — don't pick a 'primary'. No structural labels "
            "('kept whole', 'anchor', 'hybrid', 'cross') unless the "
            "user's phrasing puts that on the page.\n"
            "\n"
            "DO:\n"
            "1. List the wants in exact phrasing. Every high-value "
            "semantic word appears unchanged.\n"
            "2. For each piece of modal language attached to a want, "
            "name its effect verbatim: SOFTENS (hedges: 'ideally', "
            "'kinda', 'maybe', 'preferably', 'a bit'), HARDENS "
            "(intensifiers: 'really', 'must', 'above all', 'need'), "
            "FLIPS POLARITY (negations: 'not', 'without', 'no', "
            "'avoid', 'skip'), CONTRASTS (pivots: 'but', 'though', "
            "'yet' turning to a contrasting want).\n"
            "3. Describe how the wants relate, only as plainly as "
            "the user implied. Use the user's framing.\n"
            "\n"
            "VOICE: plain prose, no enum-style labels. Length scales "
            "with query structure. Don't pad."
        ),
    )
    atoms: list[Atom] = Field(
        ...,
        description=(
            "The query's evaluative criteria as the user phrased "
            "them — descriptive layer. One atom per distinct "
            "criterion at the granularity the user asked for.\n"
            "\n"
            "WHEN ONE, WHEN MULTIPLE: a compound stays whole when "
            "the pieces aren't separately evaluable. Otherwise: "
            "distinct evaluative criteria → distinct atoms.\n"
            "\n"
            "NEVER:\n"
            "- PARAPHRASE surface_text. Exact substring.\n"
            "- PROMOTE MODIFIERS TO ATOMS. Hedges, polarity setters, "
            "role markers, range words, comparison frames absorb as "
            "modifying_signals.\n"
            "- MERGE INDEPENDENTLY-EVALUABLE CRITERIA. Parts naming "
            "real populations whose intersection is the user's intent "
            "are distinct atoms. (Pieces that don't retrieve "
            "independently → ONE atom, dependent part absorbed as a "
            "modifying_signal.)\n"
            "- LET ABSORBED MATERIAL APPEAR TWICE. A phrase recorded "
            "as a modifying_signal claims both the phrase AND the "
            "concept inside it (setting, period, medium, style, "
            "named referent, mood). Neither becomes a separate atom.\n"
            "- COMMIT to category, polarity, salience, search "
            "strategy, or weight. Light inference allowed only "
            "inside evaluative_intent. Role / polarity / salience "
            "commitments belong on traits.\n"
            "\n"
            "ORDERING: surface-text order from the original query. "
            "Order is load-bearing downstream."
        ),
    )
    traits: list[Trait] = Field(
        ...,
        description=(
            "Committed search-ready units. Produced by the commit "
            "phase from atoms; Step 3 consumes this list.\n"
            "\n"
            "Construction (see commit-phase section in system "
            "prompt for full discipline):\n"
            "1. Resolve each atom's split_note → split or keep "
            "whole.\n"
            "2. Resolve redundancy_note plus catch any forward "
            "redundancies → dedupe.\n"
            "3. Assign role from intent shape: population-defining "
            "→ carver; reference / shaping → qualifier.\n"
            "4. Assign polarity from effect tokens: any FLIPS "
            "POLARITY → negative; otherwise positive.\n"
            "5. Assign salience from effect tokens: HARDENS or no "
            "modal → central; SOFTENS → supporting.\n"
            "\n"
            "NEVER:\n"
            "- INVENT TRAITS NOT GROUNDED IN ATOMS. Every trait "
            "traces to one or more source atoms.\n"
            "- DROP A NON-REDUNDANT ATOM. Splits add traits; merges "
            "combine. Genuine criteria don't disappear.\n"
            "- PARAPHRASE during transfer. Carry surface_text and "
            "evaluative_intent through faithfully; merges pick the "
            "clearer of the source phrasings.\n"
            "- COMMIT a category, endpoint, or concrete weight. "
            "Role / polarity / salience are the only commitments at "
            "this layer.\n"
            "\n"
            "ORDERING: traits appear in the order their source atoms "
            "appeared. Splits follow source-atom position; merges "
            "take the earlier source's slot."
        ),
    )
