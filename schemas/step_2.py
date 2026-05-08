# Query analysis output schema.
#
# Three coupled outputs:
# 1. intent_exploration — query-level exploratory analysis. Surface
#    plausible high-level intents in concrete terms and weigh which
#    is more likely. No verdict; no commitment. Sets up the atom
#    phase by perceiving the query's structural shape.
# 2. atoms — descriptive layer. surface_text + modifying_signals
#    (raw signal from anywhere in the query) + evaluative_intent
#    (consolidated meaning; the one place light inference is
#    permitted). Plus split_exploration / standalone_check —
#    evidence the commit phase reads. Atoms record.
# 3. traits — committed layer. Splits / dedupes resolved; polarity
#    and commitment committed. Step 3 consumes traits.
#
# Design principles:
# - Atoms describe; traits commit. Polarity / commitment never
#   appear on atoms.
# - One list for modifier signals (adjacent or cross-criterion —
#   conceptually the same thing).
# - Effects described, not categorized. Controlled modal tokens
#   (SOFTENS / HARDENS / FLIPS POLARITY / CONTRASTS) recommended
#   where downstream parses them; freeform elsewhere.
# - Schema = micro-prompts. System prompt is procedural and does
#   NOT duplicate field shape. NEVER lists where downstream
#   depends on discipline.

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from schemas.enums import Polarity, TraitRelationshipRole


# ---------------------------------------------------------------------
# Modifying signal
# ---------------------------------------------------------------------
#
# One signal from the query that shapes how an atom is evaluated.
# Adjacent qualifiers and cross-criterion modifiers live on the same
# list.


class ModifyingSignal(BaseModel):
    surface_phrase: str = Field(
        ...,
        description=(
            "Verbatim substring of the query. For adjacent qualifiers "
            "(hedges, polarity words, role markers, range words), "
            "just the phrase. For signals from elsewhere, the "
            "connecting language plus the reference, in the user's "
            "words. Never a positional pointer or index."
        ),
    )
    effect: str = Field(
        ...,
        description=(
            "What this signal does to the atom's evaluation, in a few "
            "words specific to this signal. Describe what it DOES; "
            "don't slot it into a closed bucket.\n"
            "\n"
            "Modal-language signals use controlled vocabulary so the "
            "commit phase can parse polarity / commitment: SOFTENS, "
            "HARDENS, FLIPS POLARITY, CONTRASTS. Recommended where "
            "they fit; freeform otherwise.\n"
            "\n"
            "When a cross-criterion content phrase shapes THIS atom's "
            "IDENTITY (not just its evaluation) — i.e. removing the "
            "other atom's content collapses what the user is asking "
            "for here into something different — additionally include "
            "the controlled token IDENTITY-SHAPING in the effect text. "
            "This is the signal the commit phase reads to decide "
            "whether two atoms should fuse into one compound trait. "
            "Use sparingly: most cross-modifications shape evaluation "
            "(scope, qualifier, transposition) without reshaping "
            "identity.\n"
            "\n"
            "NEVER:\n"
            "- CATEGORIZE. Describe the effect on evaluation, not "
            "what bucket the signal belongs to.\n"
            "- USE SYSTEM VOCABULARY. No category / endpoint / "
            "channel names.\n"
            "- USE IDENTITY-SHAPING ON SCOPING / QUALIFIER SIGNALS. "
            "If the other atom merely changes WHICH instance to score "
            "(\"shitty shark\" — shitty doesn't reshape what \"shark\" "
            "means; shark doesn't reshape what \"shitty\" means), the "
            "shaping is on EVALUATION, not identity. Reserve "
            "IDENTITY-SHAPING for compounds where each piece's "
            "meaning depends on the other being present (\"elevated "
            "horror\", \"godfather but with cowboys\")."
        ),
    )


# ---------------------------------------------------------------------
# Atom — descriptive layer
# ---------------------------------------------------------------------


# One criterion the user wants movies scored against, at the
# granularity they phrased it, plus its consolidated meaning in the
# query's full context. Descriptive — atoms record; they do not
# commit polarity / commitment.


class Atom(BaseModel):
    surface_text: str = Field(
        ...,
        description=(
            "Exact substring of the original query, with modifying "
            "language stripped (preserved in modifying_signals). "
            "NEVER paraphrase, expand named things, or substitute "
            "synonyms. The reference is the signal; what it evokes "
            "belongs to evaluative_intent."
        ),
    )
    modifying_signals: list[ModifyingSignal] = Field(
        default_factory=list,
        description=(
            "Every signal that shapes how this criterion is "
            "evaluated — adjacent qualifying language, polarity "
            "elsewhere distributing onto this criterion, and "
            "cross-criterion phrases that reshape this atom's "
            "evaluation surface. Surface-order position is "
            "irrelevant; if it shapes the meaning, it lands here.\n"
            "\n"
            "Modifier-only language (hedges, intensifiers, polarity "
            "setters, role markers, range words, structural binders) "
            "absorbs here — these have no standalone population on "
            "their own, so their meaning lives entirely as a shaping "
            "signal on the atom they attach to.\n"
            "\n"
            "Cross-atom relationships (one atom shaping another's "
            "evaluation: transposition, comparison, scoping, "
            "qualifying) ALSO record here as signals — once on each "
            "atom involved, in the user's words. The two atoms BOTH "
            "remain peer atoms; the signal describes how they relate "
            "without collapsing one into the other. Atomicity in the "
            "system prompt covers when a phrase has a standalone "
            "population (peer atom) versus when it is purely "
            "operator language (modifier-only).\n"
            "\n"
            "Empty list is the COMMON case for parallel-filter "
            "queries where criteria are genuinely independent. Don't "
            "fabricate signals to make atoms look connected."
        ),
    )
    evaluative_intent: str = Field(
        ...,
        description=(
            "1-2 sentences stating this criterion's true evaluative "
            "intention once every modifying_signal is integrated. "
            "What does scoring movies on this criterion actually "
            "mean in the query's full context?\n"
            "\n"
            "The ONE field where light inference is permitted. A "
            "hedge softens → say so. A negation flips polarity → "
            "reflect the avoid-direction. A transposing modifier → "
            "describe the transposed evaluation. A reference (not "
            "target) → say so.\n"
            "\n"
            "NEVER:\n"
            "- CATEGORIZE. No system labels ('genre', 'runtime', "
            "'actor', 'tone').\n"
            "- COMMIT POLARITY/SALIENCE VALUES. Describe direction "
            "and weight in words; commitments live on traits.\n"
            "- EXPAND NAMED THINGS. The reference is the signal.\n"
            "- TRANSLATE INTO SYSTEM VOCABULARY. No channel/vector/"
            "endpoint.\n"
            "- PARAPHRASE SURFACE_TEXT WHILE IGNORING SIGNALS. Test: "
            "would the intent change noticeably if I removed or "
            "altered this signal? If no, you haven't integrated it. "
            "(Empty signals → plain description is correct.)"
        ),
    )
    split_exploration: str = Field(
        ...,
        description=(
            "Always populated. Evidence gathering, no verdict.\n"
            "\n"
            "Two checks, both walked from evaluative_intent (not "
            "surface_text alone):\n"
            "\n"
            "(1) FORWARD — could the atom's intent be subdivided into "
            "smaller pieces, each retrievable independently? For each "
            "plausible subdivision, describe what each piece would "
            "retrieve standalone and whether the combined retrieval "
            "captures what the user is asking for at this atom's "
            "granularity.\n"
            "\n"
            "(2) INVERSE — for each modifying_signal recorded on this "
            "atom, ask whether its content phrase (stripped of "
            "connective language) names a kind-of-movie a user could "
            "ask for as a standalone search. If yes, that signal's "
            "content is itself a population — describe what splitting "
            "it back into a peer atom would look like and what each "
            "would retrieve. If no (the signal is purely operator "
            "language with no population of its own), describe why "
            "absorption is the only sensible read.\n"
            "\n"
            "The inverse check matters whenever a modifying_signal "
            "carries a content phrase rather than pure operator "
            "language. Skipping it is how population-bearing phrases "
            "get silently absorbed into anchors they shouldn't be "
            "absorbed into.\n"
            "\n"
            "NEVER:\n"
            "- WRITE A VERDICT ('keep whole' / 'split into A and "
            "B'). Describe retrieval shapes; commit phase decides.\n"
            "- DISMISS THE QUESTION. Even atoms that obviously stay "
            "whole get analyzed — state why no smaller meaningful "
            "unit exists, and (for the inverse check) why each "
            "absorbed signal is operator-only rather than population-"
            "bearing.\n"
            "- READ ONLY surface_text. Walk evaluative_intent and the "
            "modifying_signals; that is where the absorbed content "
            "lives."
        ),
    )
    standalone_check: str = Field(
        ...,
        description=(
            "Always populated. Evidence gathering, no verdict.\n"
            "\n"
            "Compare this atom's evaluative_intent against the "
            "intents surfaced in intent_exploration. Describe HOW "
            "(not if) retrieving this atom standalone — alone, "
            "ignoring the other atoms — would relate to the user's "
            "articulated intent for its part of the query.\n"
            "\n"
            "Walk through:\n"
            "- What population would standalone retrieval return?\n"
            "- Does that population fit naturally under the more-"
            "likely intent(s) surfaced in intent_exploration, or "
            "would it implicitly commit the search to a less-likely "
            "intent (introducing a hard requirement the user didn't "
            "ask for, losing a coupling the user did imply, "
            "narrowing what the user kept loose)?\n"
            "- When this atom integrates context from another atom, "
            "is that context preserved standalone, or does it fall "
            "away?\n"
            "\n"
            "Reference other atoms by their surface_text.\n"
            "\n"
            "NEVER:\n"
            "- WRITE A VERDICT ('redundant given X' / 'standalone "
            "is fine'). Describe the relationship; commit phase "
            "decides.\n"
            "- SHORT-CIRCUIT WITH UNIQUENESS. 'Primary subject' / "
            "'first mention' / 'distinct concept' are dismissals, "
            "not analyses. Walk through what standalone retrieval "
            "would actually return.\n"
            "- APPEAL TO INDEPENDENT RETRIEVABILITY AS A VIRTUE. "
            "The goal is fidelity to user-articulated intent, not "
            "standalone retrievability.\n"
            "- USE 'WHILE [COUPLING] BUT [STANDALONE VALUE]' "
            "PATTERNS. If the coupling exists, describe it; "
            "independent-retrieval value doesn't negate the meaning "
            "shift."
        ),
    )


# ---------------------------------------------------------------------
# Trait — committed layer
# ---------------------------------------------------------------------


# One search-ready unit. Produced by the commit phase from atoms —
# splits resolved, redundancies deduped, polarity and commitment
# committed. Step 3 consumes traits, not atoms.


class Trait(BaseModel):
    surface_text: str = Field(
        ...,
        description=(
            "Verbatim phrase. Clean carry-over: source atom's "
            "surface_text. After a split: the split substring. "
            "After a merge: the clearer of the source phrasings. "
            "Never paraphrase or expand."
        ),
    )
    evaluative_intent: str = Field(
        ...,
        description=(
            "1-2 sentences carried from the source atom(s). Clean "
            "carry-over copies the atom's intent verbatim. Split: "
            "re-state the relevant slice. Merge: a single intent "
            "reflecting both sources' signals. Same NEVER list as "
            "Atom.evaluative_intent."
        ),
    )
    qualifier_relation: str = Field(
        ...,
        description=(
            "Freeform prose describing how this trait positions "
            "against the rest of the query — what role it plays "
            "(reference, anchor, comparator, transposition target, "
            "etc.) AND the operational meaning of that role (what "
            "Step 3 should treat as the dimensions: a measurable "
            "axis the candidates must clear / stay under, an "
            "archetype to position against, a setting to evaluate "
            "inside, etc.).\n"
            "\n"
            "Read mechanically off the source atom's "
            "modifying_signals — do not re-derive from "
            "evaluative_intent. The phrasing is yours; the substance "
            "comes from what the signals already recorded. Step 3 "
            "consumes this prose directly to constrain its dimension "
            "scope, so be specific enough that a fresh reader could "
            "tell what kinds of dimensions belong and which don't.\n"
            "\n"
            "When no qualifier-style relationship exists in the "
            "source atom's modifying_signals, write the literal "
            "string \"n/a\". Step 3 reads \"n/a\" as an explicit "
            "no-relation signal.\n"
            "\n"
            "NEVER:\n"
            "- SLOT INTO A FIXED VOCABULARY. There is no closed list "
            "of relation types. Describe the relation specific to "
            "this query.\n"
            "- INVENT a relation absent from modifying_signals. If "
            "the signal isn't there, this is \"n/a\".\n"
            "- DUPLICATE evaluative_intent. Intent describes "
            "meaning; this names the relationship and its "
            "operational implication.\n"
            "- LEAVE BLANK. Substantive description or literal "
            "\"n/a\"."
        ),
    )
    relationship_role: TraitRelationshipRole = Field(
        ...,
        description=(
            "Closed-enum commit of how this trait relates to the rest "
            "of the query. Read AFTER qualifier_relation — the prose "
            "you just wrote either describes a positioning relation "
            "(reference / qualifier) or it does not.\n"
            "\n"
            "Three operational shapes:\n"
            "\n"
            "INDEPENDENT — the trait stands on its own for retrieval "
            "and scoring. Covers BOTH parallel filters (criteria the "
            "user wants intersected) AND qualifier-on-population "
            "relations (one trait modifies how another's population "
            "is scored, but each is independently scorable). The "
            "test: does this trait need ANY information from a "
            "sibling for Step 3 to decompose it correctly? If no, "
            "INDEPENDENT — even when sibling traits exist.\n"
            "\n"
            "POSITIONING_REFERENCE — the trait names an anchor a "
            "sibling is comparing, transposing, or scoping against. "
            "The trait's identity is being used as a TEMPLATE; "
            "specific axes of that template may be replaced by "
            "siblings. The user is not asking for the reference "
            "itself; they are asking for things that match the "
            "reference along the kept axes. \"like X\", \"X-style\", "
            "\"X but Y\", \"X meets Y\" patterns place this trait on "
            "the X side.\n"
            "\n"
            "POSITIONING_QUALIFIER — the trait names a substitute "
            "for some axis on a sibling reference. The qualifier is "
            "independently scorable, but its meaning in this query "
            "is SUBSTITUTION on the reference. The Y in \"X but Y\" "
            "/ \"X with Y\" / \"X meets Y\" patterns is here.\n"
            "\n"
            "Recognize the FUNCTION the language plays, not the "
            "specific connective. \"with\", \"but\", \"-style\", \"like\" "
            "carry independent or positioning relations depending "
            "on the content phrases they join. The connective is "
            "evidence; the role is what the trait is DOING in the "
            "query.\n"
            "\n"
            "OPERATIONAL TEST: read this commit back. \"If I asked a "
            "fresh reader to decompose this trait standalone, would "
            "they need to know about a sibling trait to do it "
            "correctly?\" If no → INDEPENDENT. If yes and this trait "
            "is the anchor → POSITIONING_REFERENCE. If yes and this "
            "trait is the modifier → POSITIONING_QUALIFIER.\n"
            "\n"
            "NEVER:\n"
            "- COMMIT BY CONNECTIVE SURFACE. \"with X\" is sometimes "
            "an independent qualifier, sometimes a positioning "
            "qualifier — depends on whether X replaces an axis of a "
            "sibling reference.\n"
            "- COMMIT POSITIONING_REFERENCE WITHOUT A SIBLING "
            "QUALIFIER. The role is reciprocal; if no sibling "
            "qualifies this trait, it is INDEPENDENT.\n"
            "- COMMIT POSITIONING_QUALIFIER WITHOUT A REPLACES_AXIS. "
            "If the trait isn't substituting on some axis of a "
            "sibling, it isn't a positioning qualifier."
        ),
    )
    replaces_axis: str | None = Field(
        ...,
        description=(
            "Required when relationship_role is POSITIONING_QUALIFIER; "
            "must be None for INDEPENDENT and POSITIONING_REFERENCE.\n"
            "\n"
            "A short user-vocabulary noun-phrase naming the AXIS on "
            "the sibling reference that this qualifier substitutes "
            "for. The axis is the dimension of evaluation; the "
            "substitute is the value this trait provides on that "
            "dimension.\n"
            "\n"
            "OPERATIONAL TEST: \"does this name a DIMENSION of "
            "evaluation, or a VALUE on that dimension?\" Dimension "
            "→ correct (\"setting\", \"tone\", \"genre\", \"emotional "
            "register\"). Value → wrong (\"jungle setting\" → use "
            "\"setting\"; \"comedic tone\" → use \"tone\"; "
            "\"sci-fi genre\" → use \"genre\").\n"
            "\n"
            "Phrasing is in user vocabulary, not category names. "
            "Multi-axis substitution (the qualifier replaces more "
            "than one dimension at once) → write a slash-joined "
            "phrase (\"genre/setting\") rather than emitting two "
            "qualifier traits.\n"
            "\n"
            "NEVER:\n"
            "- USE CATEGORY / ENDPOINT VOCABULARY. No "
            "\"NARRATIVE_SETTING\", \"GENRE\", \"EMOTIONAL_EXPERIENTIAL\".\n"
            "- NAME THE SUBSTITUTE INSTEAD OF THE AXIS. The axis is "
            "the thing being replaced; the substitute is what THIS "
            "trait offers in its place.\n"
            "- LEAVE NON-NULL FOR NON-QUALIFIER ROLES. INDEPENDENT "
            "and POSITIONING_REFERENCE traits commit None."
        ),
    )
    axes_replaced_by_siblings: list[str] = Field(
        default_factory=list,
        description=(
            "Populated when relationship_role is POSITIONING_REFERENCE; "
            "empty list otherwise.\n"
            "\n"
            "Copies VERBATIM the replaces_axis values committed by "
            "every sibling trait whose relationship_role is "
            "POSITIONING_QUALIFIER. The reference does not invent or "
            "paraphrase replacements — it inherits them from the "
            "siblings that committed them. Step 2 is the only stage "
            "that sees the whole query, so this list is where the "
            "cross-trait reasoning lands.\n"
            "\n"
            "Step 3 reads this list mechanically when decomposing "
            "this reference trait: every aspect whose user-"
            "vocabulary phrasing matches a listed axis is dropped "
            "from the decomposition (the sibling will handle that "
            "axis).\n"
            "\n"
            "OPERATIONAL TEST: \"for each entry in this list, does a "
            "sibling trait commit replaces_axis equal to this exact "
            "phrase?\" If no, the entry was invented — drop it.\n"
            "\n"
            "NEVER:\n"
            "- INVENT REPLACEMENTS not present on a sibling.\n"
            "- PARAPHRASE the sibling's phrase. Verbatim copy.\n"
            "- POPULATE FOR NON-REFERENCE ROLES. INDEPENDENT and "
            "POSITIONING_QUALIFIER traits commit empty list."
        ),
    )
    anchor_reference: str = Field(
        ...,
        description=(
            "Surface phrase from the original query (or "
            "modifying_signals) that names the modifier or operator "
            "acting on this trait. Verbatim where possible — "
            "identity is by surface phrase, never by index or "
            "positional pointer.\n"
            "\n"
            "When no modifier acts on this trait, write the literal "
            "string \"n/a\".\n"
            "\n"
            "NEVER:\n"
            "- PARAPHRASE. Carry verbatim from "
            "modifying_signals.surface_phrase.\n"
            "- POINT BY POSITION (\"the previous trait\", \"trait "
            "#2\").\n"
            "- LEAVE BLANK. Substantive phrase or literal \"n/a\"."
        ),
    )
    polarity: Polarity = Field(
        ...,
        description=(
            "Read source atom's modifying_signals. Any signal whose "
            "effect contains FLIPS POLARITY or recognizable "
            "negation language → negative. Otherwise positive. "
            "Hedges and intensifiers do NOT change polarity — they "
            "affect commitment."
        ),
    )
    commitment_evidence: str = Field(
        ...,
        description=(
            "Always populated. Evidence gathering, no verdict. "
            "Surveys two signal channels so `commitment` below "
            "commits as the natural conclusion rather than a "
            "default-fill.\n"
            "\n"
            "(1) EXPLICIT signals — walk the source atom's "
            "modifying_signals for language whose function is to "
            "fix the trait's strength. Strong-assertion language "
            "takes several recognizable shapes: phrasing that names "
            "an inviolable constraint or non-negotiable; phrasing "
            "that frames the trait as a precondition for the "
            "candidate being a viable watch at all (access, "
            "language, format, or viewer-fit gates) rather than a "
            "preference within the space of viable watches; "
            "phrasing that asserts an exclusion (the trait names "
            "something out of scope) rather than expressing a "
            "preference against (something to be downranked). "
            "Soft-framing language takes one shape: phrasing whose "
            "function is to soften the trait's claim on the result "
            "and invite the system to set it aside in exchange for "
            "matches on other axes. Recognize the FUNCTION of the "
            "language; specific surface tokens vary by query. "
            "Polarity is committed above; reading it here helps "
            "tell exclusion-assertion from preference-against.\n"
            "\n"
            "(2) STRUCTURAL signals — walk surface position "
            "(headline / leading vs. trailing), content load (bulk "
            "of the query's words vs. modest), positioning per "
            "qualifier_relation and anchor_reference (does this "
            "trait name the population the query is asking for, "
            "refine a population a peer defines, or sit coordinate "
            "with peers), and the removability test against "
            "intent_exploration's most-likely interpretation (would "
            "removing the trait collapse the structural ask, narrow "
            "it without collapse, or leave a refinement falling "
            "away).\n"
            "\n"
            "CHANNEL PRECEDENCE. The explicit channel dominates. "
            "When it fires, the trait commits at the level the "
            "explicit signal names (REQUIRED for strong assertion, "
            "DIMINISHED for soft framing) regardless of where "
            "structural prominence sits. The structural channel "
            "sets the level only when the explicit channel is "
            "silent. Note explicitly which channel is doing the "
            "work for this trait so `commitment` reads cleanly.\n"
            "\n"
            "NEVER:\n"
            "- WRITE A VERDICT. Survey the channels and stop. "
            "Picking the level is `commitment`'s job.\n"
            "- TREAT EXPLICIT-SIGNAL LANGUAGE AS A CLOSED LIST. "
            "Recognize the FUNCTION (asserting / softening); "
            "specific surface tokens vary by query.\n"
            "- DEFAULT-FILL. If a channel is silent, say so "
            "explicitly — silence is itself evidence the next "
            "channel must speak to.\n"
            "- USE SYSTEM VOCABULARY. No category / endpoint / "
            "channel names."
        ),
    )
    commitment: Literal[
        "required", "elevated", "neutral", "supporting", "diminished"
    ] = Field(
        ...,
        description=(
            "Natural conclusion of `commitment_evidence` above. "
            "Five levels on a single importance axis. The extreme "
            "levels (REQUIRED, DIMINISHED) commit only when the "
            "explicit channel fires — the user has said something "
            "out loud about the trait's strength. The middle three "
            "levels (ELEVATED, NEUTRAL, SUPPORTING) commit on the "
            "structural channel when the explicit channel is "
            "silent.\n"
            "\n"
            "REQUIRED — the explicit channel reports strong-"
            "assertion language: an inviolable constraint, a "
            "precondition for the candidate being a viable watch "
            "at all, or an asserted exclusion. Carries the largest "
            "weight in final score and reranking; reward or "
            "penalty direction is set by polarity.\n"
            "\n"
            "ELEVATED — explicit channel silent. The query's "
            "structure presents the trait as the load-bearing axis "
            "the search is fundamentally about; removing it would "
            "change what kind of movie is being asked for rather "
            "than how that movie is qualified.\n"
            "\n"
            "NEUTRAL — explicit channel silent and structural "
            "prominence reads balanced. Co-equal criterion among "
            "peers; the query would be narrower without it but "
            "would not collapse.\n"
            "\n"
            "SUPPORTING — explicit channel silent. Structural "
            "prominence reads as a refinement on a population other "
            "traits define; trailing position, modest content load.\n"
            "\n"
            "DIMINISHED — the explicit channel reports soft-framing "
            "language: phrasing that softens the trait's claim and "
            "invites the system to set it aside in exchange for "
            "matches on other axes. Includes the \"not too X\" "
            "special case (explicit dampener on a negative-polarity "
            "direction commits diminished + negative — softened "
            "preference against, not assertion against).\n"
            "\n"
            "NEVER:\n"
            "- COMMIT REQUIRED OR DIMINISHED WITHOUT AN EXPLICIT "
            "SIGNAL. Both extremes require the user to have said "
            "something out loud about the trait's strength. Strong "
            "structural prominence alone commits ELEVATED, not "
            "REQUIRED. Structural triviality alone commits "
            "SUPPORTING, not DIMINISHED.\n"
            "- DEFAULT-FILL TO NEUTRAL. Neutral commits when the "
            "explicit channel is silent and structural prominence "
            "reads balanced — not when the explicit channel is "
            "silent and the structural channel pointed elsewhere.\n"
            "- LET STRUCTURAL PROMINENCE OVERRIDE EXPLICIT FRAMING. "
            "When the explicit channel fires, the level it names "
            "commits regardless of where the structural channel "
            "would have landed."
        ),
    )
    contextualized_phrase: str = Field(
        ...,
        description=(
            "One short phrase restating this trait with its "
            "modifying signals folded in, in the user's voice. Step "
            "3 reads this as the headline trait identity ahead of "
            "surface_text. A bare named-entity surface phrase, "
            "stripped of its query context, looks like a literal "
            "lookup target even when the surrounding modifier was "
            "positioning the entity rather than naming it as a "
            "retrieval target — this field re-attaches the context "
            "so Step 3's routing can read the modifier's role.\n"
            "\n"
            "Construction:\n"
            "- Start from surface_text.\n"
            "- Fold in anchor_reference and any other "
            "modifying_signals whose effect changes WHAT KIND of "
            "movie the trait is asking for (transposition, "
            "comparison framing, qualifier role, etc.).\n"
            "- Result is one short phrase. Faithful restatement, "
            "not interpretation.\n"
            "- When no meaning-shaping modifier acts on the trait, "
            "copy surface_text verbatim.\n"
            "\n"
            "TEST: read the phrase aloud out of query context. Can "
            "a fresh reader recover what the trait is asking for? "
            "If yes, keep. If no, the modifier wasn't folded in "
            "clearly enough.\n"
            "\n"
            "NEVER:\n"
            "- DECOMPOSE into multiple phrases. One per trait.\n"
            "- EXPAND PARAMETRICALLY. No 'such as' / 'like X, Y, "
            "Z'.\n"
            "- ADD details the original query did not contain.\n"
            "- DROP details the original query did contain.\n"
            "- TRANSLATE INTO SYSTEM VOCABULARY."
        ),
    )


# ---------------------------------------------------------------------
# Top-level response
# ---------------------------------------------------------------------
#
# Combined output: query-level intent exploration, descriptive atoms
# with consolidated evaluative intent, and committed search-ready
# traits.


class QueryAnalysis(BaseModel):
    intent_exploration: str = Field(
        ...,
        description=(
            "Exploratory analysis, no verdict. Walk the query and "
            "describe the plausible high-level intents it could be "
            "expressing, in concrete terms — what kind of movie "
            "would satisfy each intent, and how the query's pieces "
            "relate to that kind (which piece names the population, "
            "which pieces narrow or qualify it).\n"
            "\n"
            "When the query genuinely admits more than one plausible "
            "read, surface each one and reason about which is more "
            "likely using context cues from the query.\n"
            "\n"
            "When the query admits one obvious read, one is enough "
            "— don't manufacture alternatives. Length scales with "
            "the ambiguity actually present.\n"
            "\n"
            "Concrete means describing what KIND of movie a viewer "
            "would actually be watching if the intent were "
            "satisfied — the population in user vocabulary, not "
            "abstract labels naming what type of intent it is.\n"
            "\n"
            "NEVER:\n"
            "- COMMIT TO ONE READING WHEN MULTIPLE ARE PLAUSIBLE. "
            "Commitment happens in atoms and traits; here, surface "
            "and weigh.\n"
            "- LIST INTENTS WITHOUT WEIGHING THEM. When you surface "
            "more than one, say which is more likely and from what "
            "cues. A bare list is not exploration.\n"
            "- CATEGORIZE. A categorical label names the kind of "
            "attribute the system would key off; a concrete "
            "description names the watching experience the user "
            "wants. Describe the experience.\n"
            "- EXPAND NAMED THINGS. The reference is the signal; "
            "what it evokes belongs to atom evaluative_intent.\n"
            "- INVENT DETAILS the query did not contain."
        ),
    )
    atoms: list[Atom] = Field(
        ...,
        description=(
            "The query's evaluative criteria as the user phrased "
            "them. One atom per distinct criterion at the user's "
            "granularity.\n"
            "\n"
            "WHEN ONE, WHEN MULTIPLE: a compound stays whole when "
            "the pieces aren't separately evaluable. Otherwise: "
            "distinct evaluable criteria → distinct atoms.\n"
            "\n"
            "NEVER:\n"
            "- PARAPHRASE surface_text. Exact substring.\n"
            "- PROMOTE MODIFIER-ONLY LANGUAGE TO ATOMS. Hedges, "
            "intensifiers, polarity setters, role markers, range "
            "words, structural binders, and pure comparison "
            "operators have no standalone population — they absorb "
            "as modifying_signals.\n"
            "- MERGE INDEPENDENTLY-EVALUABLE CRITERIA. Pieces that "
            "each name a population a user could ask for as a "
            "standalone search are distinct atoms — even when "
            "grammatically positioned as one operating on the "
            "other. Their relationship records as a "
            "modifying_signal on each peer atom; both peers "
            "survive.\n"
            "- DOUBLE-EMIT. Once atomicity decides a phrase is "
            "absorbed (modifier-only) or peer (population-bearing), "
            "don't also emit it in the other slot. A modifier-only "
            "phrase doesn't get a peer-atom version; a peer-atom "
            "phrase doesn't get a duplicate inside another atom's "
            "modifying_signals as if it were absorbed.\n"
            "- COMMIT category, polarity, commitment, search "
            "strategy, or weight. Light inference allowed only "
            "inside evaluative_intent. Polarity / commitment belong "
            "on traits.\n"
            "\n"
            "ORDERING: surface-text order from the original query. "
            "Order is load-bearing downstream."
        ),
    )
    traits: list[Trait] = Field(
        ...,
        description=(
            "Committed search-ready units produced by the commit "
            "phase from atoms. Step 3 consumes this list. See the "
            "commit-phase section in the system prompt for the full "
            "construction discipline.\n"
            "\n"
            "NEVER:\n"
            "- INVENT TRAITS not grounded in atoms. Every trait "
            "traces to one or more source atoms.\n"
            "- DROP a non-redundant atom. Splits add traits; merges "
            "combine. Genuine criteria don't disappear.\n"
            "- PARAPHRASE during transfer. Carry surface_text and "
            "evaluative_intent through faithfully; merges pick the "
            "clearer of the source phrasings.\n"
            "- COMMIT a category, endpoint, or concrete weight. "
            "Polarity / commitment are the only commitments at "
            "this layer.\n"
            "\n"
            "ORDERING: traits appear in the order their source "
            "atoms appeared. Splits inherit position; merges take "
            "the earlier source's slot."
        ),
    )

    # Belt-and-suspenders enforcement of the relationship-role
    # typology's structural invariants. The Step 2 prompt teaches
    # these rules, but Pydantic doesn't validate cross-field
    # consistency by default — without this validator, an LLM that
    # commits POSITIONING_QUALIFIER while the sibling reference
    # forgets to populate axes_replaced_by_siblings would silently
    # corrupt Step 3's axis-drop logic. Catching the malformed state
    # at validation time triggers the orchestrator's retry rather
    # than letting it through.
    @model_validator(mode="after")
    def _validate_relationship_roles(self) -> "QueryAnalysis":
        traits = self.traits

        # Self-healing pre-pass: degrade noisy positioning commits to
        # INDEPENDENT rather than rejecting the whole query. Two cases
        # the LLM occasionally produces with no semantic content:
        #   (a) POSITIONING_REFERENCE with empty axes_replaced_by_
        #       siblings — there is no sibling axis to drop, so the
        #       trait would behave identically to INDEPENDENT under
        #       Step 3 anyway.
        #   (b) POSITIONING_QUALIFIER with empty replaces_axis — the
        #       trait has no axis to substitute, so it cannot do the
        #       qualifier's job. Treating it as INDEPENDENT loses
        #       nothing the qualifier role would have added.
        # After the per-trait fix, recheck reciprocity: if all refs
        # (or all quals) on one side just collapsed, the surviving
        # orphans on the other side are also INDEPENDENT in effect.
        # Strict axis-bookkeeping checks below still run — those
        # detect non-coercible LLM errors (axis names that don't
        # match between sibling commits) that would silently corrupt
        # Step 3's drop logic if let through.
        for t in traits:
            role = t.relationship_role
            if (
                role is TraitRelationshipRole.POSITIONING_REFERENCE
                and not t.axes_replaced_by_siblings
            ):
                t.relationship_role = TraitRelationshipRole.INDEPENDENT
                t.replaces_axis = None
            elif (
                role is TraitRelationshipRole.POSITIONING_QUALIFIER
                and not t.replaces_axis
            ):
                t.relationship_role = TraitRelationshipRole.INDEPENDENT
                t.axes_replaced_by_siblings = []

        has_ref = any(
            t.relationship_role is TraitRelationshipRole.POSITIONING_REFERENCE
            for t in traits
        )
        has_qual = any(
            t.relationship_role is TraitRelationshipRole.POSITIONING_QUALIFIER
            for t in traits
        )
        if has_ref and not has_qual:
            for t in traits:
                if t.relationship_role is TraitRelationshipRole.POSITIONING_REFERENCE:
                    t.relationship_role = TraitRelationshipRole.INDEPENDENT
                    t.replaces_axis = None
                    t.axes_replaced_by_siblings = []
        elif has_qual and not has_ref:
            for t in traits:
                if t.relationship_role is TraitRelationshipRole.POSITIONING_QUALIFIER:
                    t.relationship_role = TraitRelationshipRole.INDEPENDENT
                    t.replaces_axis = None
                    t.axes_replaced_by_siblings = []

        # Per-trait field consistency. Each role has one valid
        # combination of (replaces_axis, axes_replaced_by_siblings).
        for i, t in enumerate(traits):
            role = t.relationship_role
            if role is TraitRelationshipRole.INDEPENDENT:
                if t.replaces_axis is not None:
                    raise ValueError(
                        f"trait[{i}] role=INDEPENDENT but "
                        f"replaces_axis={t.replaces_axis!r}; expected None"
                    )
                if t.axes_replaced_by_siblings:
                    raise ValueError(
                        f"trait[{i}] role=INDEPENDENT but "
                        f"axes_replaced_by_siblings="
                        f"{t.axes_replaced_by_siblings}; expected []"
                    )
            elif role is TraitRelationshipRole.POSITIONING_REFERENCE:
                if t.replaces_axis is not None:
                    raise ValueError(
                        f"trait[{i}] role=POSITIONING_REFERENCE but "
                        f"replaces_axis={t.replaces_axis!r}; expected None"
                    )
                if not t.axes_replaced_by_siblings:
                    raise ValueError(
                        f"trait[{i}] role=POSITIONING_REFERENCE but "
                        "axes_replaced_by_siblings is empty; the "
                        "reference must inherit at least one axis "
                        "from a sibling qualifier (without one, the "
                        "role is INDEPENDENT)"
                    )
            elif role is TraitRelationshipRole.POSITIONING_QUALIFIER:
                if not t.replaces_axis:
                    raise ValueError(
                        f"trait[{i}] role=POSITIONING_QUALIFIER but "
                        f"replaces_axis={t.replaces_axis!r}; the "
                        "qualifier must name the axis it substitutes "
                        "on the sibling reference"
                    )
                if t.axes_replaced_by_siblings:
                    raise ValueError(
                        f"trait[{i}] role=POSITIONING_QUALIFIER but "
                        f"axes_replaced_by_siblings="
                        f"{t.axes_replaced_by_siblings}; expected [] "
                        "(only references inherit axis substitutions)"
                    )

        # Cross-trait reciprocity. Positioning is a paired relation —
        # a reference without any qualifier (or vice versa) is
        # malformed; the orphaned trait should have committed
        # INDEPENDENT.
        refs = [
            t for t in traits
            if t.relationship_role is TraitRelationshipRole.POSITIONING_REFERENCE
        ]
        quals = [
            t for t in traits
            if t.relationship_role is TraitRelationshipRole.POSITIONING_QUALIFIER
        ]
        if refs and not quals:
            raise ValueError(
                "POSITIONING_REFERENCE trait(s) committed but no "
                "POSITIONING_QUALIFIER sibling exists; references "
                "commit only when a qualifier targets them"
            )
        if quals and not refs:
            raise ValueError(
                "POSITIONING_QUALIFIER trait(s) committed but no "
                "POSITIONING_REFERENCE sibling exists; qualifiers "
                "substitute on a sibling reference's axis — without "
                "a reference the role should be INDEPENDENT"
            )

        # Axis-bookkeeping reciprocity. Every qualifier's
        # replaces_axis must be inherited by some reference, and
        # every entry in any reference's axes_replaced_by_siblings
        # must trace to a sibling qualifier. Verbatim string match
        # per the schema description ("VERBATIM copy").
        qual_axes: set[str] = {q.replaces_axis for q in quals if q.replaces_axis}
        ref_inherit_axes: set[str] = set()
        for r in refs:
            ref_inherit_axes.update(r.axes_replaced_by_siblings)
        missing_on_refs = qual_axes - ref_inherit_axes
        if missing_on_refs:
            raise ValueError(
                "POSITIONING_QUALIFIER replaces_axis values not "
                "inherited by any sibling reference's "
                f"axes_replaced_by_siblings: {sorted(missing_on_refs)}"
            )
        invented_on_refs = ref_inherit_axes - qual_axes
        if invented_on_refs:
            raise ValueError(
                "POSITIONING_REFERENCE axes_replaced_by_siblings "
                "entries not committed by any sibling qualifier: "
                f"{sorted(invented_on_refs)}"
            )
        return self
