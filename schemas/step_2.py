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
# 3. traits — committed layer. Splits / dedupes resolved; role,
#    polarity, salience committed. Step 3 consumes traits.
#
# Design principles:
# - Atoms describe; traits commit. Role / polarity / salience
#   never appear on atoms.
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

from pydantic import BaseModel, Field

from schemas.enums import Polarity, Role


# ---------------------------------------------------------------------
# Modifying signal
# ---------------------------------------------------------------------


class ModifyingSignal(BaseModel):
    """One signal from the query that shapes how an atom is
    evaluated. Adjacent qualifiers and cross-criterion modifiers
    live on the same list."""

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
            "commit phase can parse polarity / salience: SOFTENS, "
            "HARDENS, FLIPS POLARITY, CONTRASTS. Recommended where "
            "they fit; freeform otherwise.\n"
            "\n"
            "NEVER:\n"
            "- CATEGORIZE. Describe the effect on evaluation, not "
            "what bucket the signal belongs to.\n"
            "- USE SYSTEM VOCABULARY. No category / endpoint / "
            "channel names."
        ),
    )


# ---------------------------------------------------------------------
# Atom — descriptive layer
# ---------------------------------------------------------------------


class Atom(BaseModel):
    """One criterion the user wants movies scored against, at the
    granularity they phrased it, plus its consolidated meaning in
    the query's full context. Descriptive — atoms record; they do
    not commit role / polarity / salience."""

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


class Trait(BaseModel):
    """One search-ready unit. Produced by the commit phase from
    atoms — splits resolved, redundancies deduped, role / polarity /
    salience committed. Step 3 consumes traits, not atoms."""

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
            "When role=carver and no qualifier-style relationship "
            "exists in the signals, write the literal string \"n/a\". "
            "Step 3 reads \"n/a\" as an explicit no-relation signal.\n"
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
    role_evidence: str = Field(
        ...,
        description=(
            "One sentence. Read intent_exploration's most-likely "
            "interpretation as the primary source — it has already "
            "identified which piece of the query gates the "
            "population vs which refines. Use qualifier_relation "
            "(committed above) and the other atoms / traits in the "
            "query as contextual grounding for that frame. Against "
            "this primary frame, can this trait on its own "
            "definitively include or exclude films from eligibility "
            "(→ carver), or does it qualify because (a) it can only "
            "be evaluated as a continuous score rather than a "
            "yes/no membership, (b) it is used as a comparison "
            "reference rather than naming the population the user "
            "wants, or (c) another atom or trait in the query "
            "already gates the population this one would only "
            "refine?"
        ),
    )
    role: Role = Field(
        ...,
        description=(
            "Conclusion of role_evidence above. CARVER: the trait "
            "definitively gates eligibility — its presence or "
            "absence determines whether a film qualifies as a "
            "candidate. QUALIFIER: the trait scores or refines "
            "within a population other traits gate, OR is itself a "
            "comparison reference rather than the population the "
            "user wants."
        ),
    )
    polarity: Polarity = Field(
        ...,
        description=(
            "Read source atom's modifying_signals. Any signal whose "
            "effect contains FLIPS POLARITY or recognizable "
            "negation language → negative. Otherwise positive. "
            "Hedges and intensifiers do NOT change polarity — they "
            "affect salience."
        ),
    )
    relevance_to_query: str = Field(
        ...,
        description=(
            "1-2 sentences walking through how this trait sits in "
            "the query: hedges or intensifiers attached, position "
            "in surface order (early/headline vs trailing), words "
            "spent, whether removing it would meaningfully change "
            "the ask. Modal effect tokens (SOFTENS, HARDENS) are "
            "one signal but not the whole picture — within-query "
            "position and structural prominence contribute too. "
            "Salience commits as the natural conclusion. No system "
            "vocabulary; no concrete numbers."
        ),
    )
    salience: Literal["central", "supporting"] = Field(
        ...,
        description=(
            "Natural conclusion from relevance_to_query. CENTRAL = "
            "headline want; query feels fundamentally different "
            "without it. SUPPORTING = meaningful but rounds out an "
            "already-defined ask.\n"
            "\n"
            "Applies to all traits regardless of role. A "
            "non-central carver acts as a lenient filter; "
            "downstream reads salience and adjusts."
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
            "- Carver traits with no meaning-shaping modifier copy "
            "surface_text verbatim.\n"
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


class QueryAnalysis(BaseModel):
    """Combined output: query-level intent exploration, descriptive
    atoms with consolidated evaluative intent, and committed
    search-ready traits."""

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
            "- COMMIT category, polarity, salience, search "
            "strategy, or weight. Light inference allowed only "
            "inside evaluative_intent. Role / polarity / salience "
            "belong on traits.\n"
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
            "Role / polarity / salience are the only commitments at "
            "this layer.\n"
            "\n"
            "ORDERING: traits appear in the order their source "
            "atoms appeared. Splits inherit position; merges take "
            "the earlier source's slot."
        ),
    )
