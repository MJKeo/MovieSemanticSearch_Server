# Trait decomposition output schema (Step 3).
#
# One LLM call per trait emits a TraitDecomposition with three
# coupled layers:
#
#   1. Trait-role analysis — target_population + trait_role_analysis.
#      Reads role / qualifier_relation / anchor_reference (committed
#      by Step 2) and commits what those mean for what the dimensions
#      describe. Pre-dimension; prevents qualifier traits from
#      drifting back into population-defining decompositions.
#   2. Dimension inventory + per-dimension routing candidates. Each
#      dimension is a concrete database-vocabulary check; each
#      carries a freeform list of plausible categories with explicit
#      what's-covered / what's-missing prose.
#   3. Commitment layer — category_calls. Minimum number of
#      taxonomy-routed calls whose union reproduces the trait's
#      intent. One call per category; multi-expression calls when
#      one category cleanly covers several dimensions of one trait.
#      Step 4 consumes this list.
#
# Design principles:
# - Role-commitment first. trait_role_analysis precedes dimensions
#   so qualifier_relation / anchor_reference shape what dimensions
#   describe.
# - Dimensions are the smallest searchable unit (database-vocabulary).
# - Coverage analysis structured, not prose. Per-dimension candidate
#   list with what-covered / what-missed surfaces partial fits and
#   adjacency honestly.
# - Multi-expression calls are natural for figurative multi-faceted
#   traits whose dimensions all route to a single category.
# - Additive composition only. Calls combine by unweighted sum
#   across CALLS (not expressions). No per-call weighting.
# - Presence only. Polarity is upstream and applied at merge time —
#   even for negative traits, expressions describe presence of the
#   attribute being avoided.
# - Minimum SET. Most traits resolve to one call; padding dilutes
#   the trait's score sum.
# - Schema = micro-prompts. The system prompt is procedural and
#   does NOT duplicate field shape.

from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.enums import TraitCombineMode
from schemas.trait_category import CategoryName


# ---------------------------------------------------------------------
# CategoryCandidate — per-dimension routing analysis
# ---------------------------------------------------------------------


class CategoryCandidate(BaseModel):
    """One plausible category for a single dimension. Multiple per
    dimension is normal — surfaces partial fits and adjacency so the
    commitment step is grounded rather than back-rationalized."""

    category: CategoryName = Field(
        ...,
        description=(
            "The taxonomy category being analyzed for fit. "
            "Constrained to the closed set in the system prompt's "
            "CATEGORY TAXONOMY section."
        ),
    )
    what_this_covers: str = Field(
        ...,
        description=(
            "One short sentence stating which aspect of the "
            "dimension this category genuinely owns. Cite the "
            "category's description or boundary line when adjacency "
            "exists. Concrete, not generic.\n"
            "\n"
            "NEVER:\n"
            "- BACK-RATIONALIZE. If the only thing you can say is "
            "'this is plausible', drop the candidate.\n"
            "- GENERALIZE. Coverage is about THIS dimension."
        ),
    )
    what_this_misses: str = Field(
        ...,
        description=(
            "One short phrase or sentence stating what the "
            "dimension calls for that this category does NOT cover. "
            "\"Nothing\" is correct when the category is a clean "
            "fit and no adjacent category competes. Otherwise, name "
            "the specific aspect (a tonal sub-shade, a chronological "
            "ordering signal, a craft attribute) the boundary "
            "explicitly redirects elsewhere — and where.\n"
            "\n"
            "NEVER:\n"
            "- HEDGE WITHOUT NAMING. Either commit \"nothing\" or "
            "name what's specifically missing.\n"
            "- INVENT GAPS to look thorough."
        ),
    )


# ---------------------------------------------------------------------
# Dimension — analysis layer
# ---------------------------------------------------------------------


class Dimension(BaseModel):
    """One concrete searchable piece this trait calls for, in
    database-vocabulary, plus its routing candidates. Smallest
    unit — no abstraction up, no bundling, no category names in
    the expression itself."""

    expression: str = Field(
        ...,
        description=(
            "MUST be a verbatim copy of one of the strings in the "
            "trait's `aspects` list. Character-for-character. No "
            "rewriting, no tightening, no rewording, no \"close "
            "enough\" paraphrase, no merging two aspects into one "
            "phrase. Each dimension names the exact aspect being "
            "analyzed; the dimensions layer is where you SELECT "
            "which aspects to route and HOW (via category_candidates "
            "below), not where you re-author the aspects in different "
            "words.\n"
            "\n"
            "The translation work — turning user-vocabulary into a "
            "database-runnable check — happens in category_candidates "
            "and downstream in the endpoint handler. expression itself "
            "is purely a pointer back to the aspect this dimension "
            "covers. If the aspect string is awkward as a stand-alone "
            "check, fix it upstream by revising the aspect, not by "
            "rephrasing it here.\n"
            "\n"
            "Multiple dimensions MAY share the same aspect-string when "
            "the aspect routes through more than one category — each "
            "dimension still emits the aspect verbatim and uses "
            "category_candidates to record the different routing "
            "options. Multi-dimensional aspect coverage is the rare "
            "case; one-dimension-per-aspect is the default.\n"
            "\n"
            "NEVER:\n"
            "- REWRITE THE ASPECT. Even small edits (\"highly praised "
            "for performances\" → \"praised for performances\") "
            "violate the verbatim rule. Copy the aspect string as-is.\n"
            "- MERGE TWO ASPECTS INTO ONE EXPRESSION. If two aspects "
            "feel close enough to combine, emit two separate dimensions "
            "(each with its own verbatim aspect string) and let them "
            "merge later at the category_calls layer if they route to "
            "the same category.\n"
            "- NAME A CATEGORY. Categories belong to "
            "category_candidates and category_calls.\n"
            "- EXPRESS ABSENCE. Presence-only — polarity is upstream "
            "and applied at merge time.\n"
            "- INVENT AN EXPRESSION not present in aspects. Every "
            "dimension traces to exactly one aspect string."
        ),
    )
    category_candidates: list[CategoryCandidate] = Field(
        ...,
        description=(
            "Plausible categories that could own this dimension, "
            "with what each covers and misses. NO upper bound — "
            "list every category whose description, boundary, or "
            "edge_cases makes it a real candidate. One candidate is "
            "fine when fit is unambiguous; two or three is normal "
            "when adjacency exists; more when the dimension "
            "genuinely sits between several.\n"
            "\n"
            "Forces honest routing analysis. When a dimension has "
            "only one candidate AND adjacent categories could "
            "plausibly compete, you haven't done the audit honestly "
            "— re-read the boundary lines.\n"
            "\n"
            "Test: \"If I removed this candidate, would the commit "
            "step lose a real routing option, or was it padding?\" "
            "Padding → drop. Real option → keep, even if you "
            "ultimately won't commit to it.\n"
            "\n"
            "NEVER:\n"
            "- LIST ONLY ONE OUT OF HABIT. When the taxonomy has "
            "an adjacency, surface it.\n"
            "- PAD with categories that have no real coverage. "
            "what_this_covers must be substantive.\n"
            "- DUPLICATE CATEGORIES per dimension."
        ),
    )


# ---------------------------------------------------------------------
# CategoryCall — commitment layer
# ---------------------------------------------------------------------


class CategoryCall(BaseModel):
    """One committed routing call. Owns one or more expressions
    when a single category cleanly covers several dimensions of the
    same trait. Step 4 turns this into endpoint-specific structured
    queries."""

    category: CategoryName = Field(
        ...,
        description=(
            "The taxonomy category this call routes to. Must have "
            "appeared as a candidate on at least one of the "
            "dimensions whose expressions this call now owns. When "
            "adjacency was surfaced in candidates, the commitment "
            "must reflect the deciding factor (cite the boundary "
            "or edge_case text)."
        ),
    )
    expressions: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "One or more short searchable phrases — Step 4's seeds. "
            "Database-vocabulary, not user-vocabulary. No hedges, no "
            "polarity words, no comparison framing — those were "
            "upstream commitments. Each entry is a tight phrase, "
            "not a sentence.\n"
            "\n"
            "Cardinality follows from the dimension list: one entry "
            "per dimension this call owns. When the call owns one "
            "dimension, the list has one entry. When several "
            "dimensions of the same trait route to this category, "
            "the list carries one entry per dimension — that's the "
            "natural multi-expression shape, not separate calls.\n"
            "\n"
            "Each expression must trace back to exactly one "
            "Dimension.expression — verbatim, lightly tightened, or "
            "recognizably the same check.\n"
            "\n"
            "NEVER:\n"
            "- SPLIT INTO MULTIPLE CALLS WHEN ONE CATEGORY OWNS "
            "THEM ALL. Multi-expression is the natural shape; "
            "separate same-category calls dilute the trait's score "
            "sum.\n"
            "- PAD WITH NEAR-DUPLICATES. Two phrases that read as "
            "trivial rewordings of each other are not two "
            "expressions.\n"
            "- LEAVE EMPTY. Drop the call instead."
        ),
    )
    retrieval_intent: str = Field(
        ...,
        description=(
            "1-3 sentences. The handoff field for Step 4. Convey the "
            "context that the short `expressions` phrases cannot "
            "carry on their own: what exactly is being searched for, "
            "what shape the search should take, and what the call is "
            "trying to discriminate between. Step 4 reads ONLY this "
            "field plus expressions to build its query, so the "
            "context that lives here is the context Step 4 has — "
            "missing nuance here is missing nuance in the search.\n"
            "\n"
            "ALWAYS POPULATED. Both population-naming and "
            "reference-positioning traits produce calls; both calls "
            "need Step 4 to understand what's being retrieved. The "
            "positioning distinction is orchestrator-side and already "
            "committed — Step 4 does not branch on it.\n"
            "\n"
            "What to convey:\n"
            "- The dimension(s) this call owns, expressed in plain "
            "user/database vocabulary so Step 4 understands the "
            "attribute being searched.\n"
            "- The shape of the search: looking for an exact match \n"
            "of a named entity, scoring along a continuous "
            "experiential axis, evaluating presence of an archetype \n"
            "or iconography, satisfying a setting or temporal "
            "window, etc.\n"
            "- When the trait's qualifier_relation describes a "
            "positioning relationship, fold in the operational "
            "meaning from trait_role_analysis (reference being "
            "positioned against, threshold candidates must clear or "
            "stay under, archetype to satisfy, setting to evaluate "
            "inside, craft template to match) so Step 4 understands "
            "the positioning intent.\n"
            "- When the trait's qualifier_relation is \"n/a\", name "
            "the population the call gates on so Step 4 understands "
            "what to admit / reject.\n"
            "\n"
            "NEVER:\n"
            "- LEAVE THIN. \"Filter for X\" without context is the "
            "common failure mode. Expand to convey what `X` actually "
            "means in this query.\n"
            "- DESCRIBE ABSENCE. Polarity is upstream; this layer "
            "is presence-only.\n"
            "- WEIGHT THIS CALL AGAINST OTHERS. Composition is "
            "additive and unweighted; per-call weighting forbidden.\n"
            "- NAME ENDPOINTS, FIELDS, OR VECTOR SPACES. Step 4's "
            "job. Stay in user/database vocabulary."
        ),
    )


# ---------------------------------------------------------------------
# TraitDecomposition — top-level response
# ---------------------------------------------------------------------


class TraitDecomposition(BaseModel):
    """Combined output of Step 3 for one trait. Role analysis must
    precede the dimension inventory; candidates must precede
    category_calls."""

    trait_restatement: str = Field(
        ...,
        description=(
            "ANCHOR FIELD — produced FIRST, before any inference. "
            "Restate the trait's upstream commit by reproducing, "
            "verbatim and in order:\n"
            "(1) the contextualized_phrase exactly as received, in "
            "double quotes;\n"
            "(2) the evaluative_intent exactly as received, in "
            "double quotes;\n"
            "(3) the relationship_role value, in single quotes;\n"
            "(4) if non-empty: replaces_axis (single quotes) or "
            "axes_replaced_by_siblings (bracketed list).\n"
            "\n"
            "No paraphrasing. No summarization. No commentary. No "
            "additions. Copy the upstream strings character-for-"
            "character. This restatement is the anchor for every "
            "field below — target_population, trait_role_analysis, "
            "aspects, dimensions, and category_calls each describe "
            "the trait this restatement names, no more and no less. "
            "Content not present in this restatement is content that "
            "was not in the trait; do not introduce it downstream."
        ),
    )
    target_population: str = Field(
        ...,
        description=(
            "1-2 sentences. Restate, in your own words, the "
            "population of movies this trait wants. What do those "
            "movies share? What can vary freely? Seed for the "
            "dimension inventory.\n"
            "\n"
            "Source: the trait_restatement you just produced. Every "
            "clause of target_population must trace to a word inside "
            "the quoted strings in trait_restatement (the verbatim "
            "contextualized_phrase and evaluative_intent). Do not "
            "introduce content present nowhere in those quotes — no "
            "exemplars, no 'or'-clauses admitting neighboring "
            "criteria, no canonical instances, no sub-types from "
            "prior knowledge of what this kind of trait 'typically' "
            "covers. If a clause has no anchor in the restatement, "
            "remove it.\n"
            "\n"
            "NEVER:\n"
            "- NAME CATEGORIES OR ENDPOINTS.\n"
            "- COPY evaluative_intent VERBATIM (the restatement "
            "already does that). Restate in concrete population "
            "terms at the same width the trait carries.\n"
            "- BROADEN, NARROW, OR INVENT against trait_restatement. "
            "Same constraint, same granularity, same scope as the "
            "quoted upstream text."
        ),
    )
    trait_role_analysis: str = Field(
        ...,
        description=(
            "1-2 sentences committing what the trait's role + "
            "qualifier_relation + anchor_reference (read MECHANICALLY "
            "off the user prompt — the values were committed in "
            "Step 2 and are the source of truth) mean for what the "
            "dimensions list should describe.\n"
            "\n"
            "Source: trait_restatement and target_population, both "
            "above. The role analysis describes the trait those "
            "fields name — it does not introduce new content. Every "
            "clause must trace to a word in the quoted "
            "contextualized_phrase / evaluative_intent in "
            "trait_restatement, to target_population, or to the "
            "relationship_role / axis bookkeeping the restatement "
            "reproduced.\n"
            "\n"
            "Two questions to answer in your prose:\n"
            "(1) Does this trait name something to RETRIEVE, or "
            "something to POSITION retrieval AGAINST? Carver → "
            "retrieve; qualifier → position against. The role field "
            "tells you which.\n"
            "(2) If qualifier, what is the operational meaning of "
            "the positioning, drawn from what qualifier_relation "
            "says in user-vocabulary? Translate the relation prose "
            "into what KIND of dimensions belong (a measurable axis "
            "the candidates must clear / stay under, an archetype "
            "or iconography to satisfy, a setting / period to "
            "evaluate inside, a craft template to match, etc.) — "
            "without slotting into a fixed bucket. The relation is "
            "freeform; your analysis is freeform too.\n"
            "\n"
            "TEST: read this analysis back. Does it clearly "
            "constrain what the dimensions list should contain? "
            "Could a different reader, given only this analysis, "
            "write the same kind of dimensions? If no, revise.\n"
            "\n"
            "NEVER:\n"
            "- RE-INTERPRET qualifier_relation. Step 2 commit is "
            "the source of truth — read it, don't second-guess it.\n"
            "- DERIVE A DIFFERENT ROLE FROM evaluative_intent.\n"
            "- SLOT INTO A FIXED VOCABULARY of relation types. "
            "Describe what THIS query's relation means.\n"
            "- INTRODUCE CONTENT NOT IN trait_restatement / "
            "target_population. The role analysis tightens what the "
            "dimensions should describe; it does not enlarge the "
            "scope of the trait.\n"
            "- LEAVE GENERIC. \"This trait wants movies that match "
            "it\" is a non-analysis."
        ),
    )
    aspects: list[str] = Field(
        ...,
        description=(
            "Flat enumeration of every distinguishable axis the "
            "trait calls for. One short noun-phrase per entry, in "
            "user-vocabulary or natural English. This is the "
            "enumeration step; the dimensions step below translates "
            "each entry into a database-vocabulary check.\n"
            "\n"
            "Walk the target_population and trait_role_analysis you "
            "just committed. Every distinct axis those prose fields "
            "identified becomes one entry here. A population "
            "defined by one axis contributes one entry; a "
            "population defined by several simultaneous conditions "
            "contributes one entry per condition. Cardinality "
            "follows what the trait actually says — concrete "
            "single-axis traits resolve to one entry; figurative or "
            "compound traits resolve to several.\n"
            "\n"
            "Mode-shift: this list stays in the user's mental "
            "model. Database-side routing (which category owns each "
            "aspect) happens in the dimensions step that follows. "
            "Crucially, each dimension below carries this aspect's "
            "string VERBATIM in its expression field — the dimensions "
            "step routes; it does not re-author. That makes the "
            "aspects you write here load-bearing: every aspect "
            "string flows downstream character-for-character into "
            "the dimensions and category_calls layers, and any "
            "drift, awkwardness, or imprecision here propagates "
            "without correction.\n"
            "\n"
            "TEST: read this list back. Could each entry stand on "
            "its own as a clean expression string the routing step "
            "would copy verbatim? If an entry is awkward, vague, or "
            "would force the dimensions step to paraphrase, revise "
            "it here. The dimensions step is forbidden from "
            "rewriting these strings, so they must already be "
            "right.\n"
            "\n"
            "NEVER:\n"
            "- TRANSLATE INTO CATEGORY VOCABULARY. Categories live "
            "in dimensions / candidates / calls. An aspect is a "
            "user-side axis, not a routing slot.\n"
            "- COLLAPSE distinct axes into one phrase. If "
            "target_population names two axes that vary "
            "independently, surface both.\n"
            "- INVENT axes not grounded in target_population or "
            "trait_role_analysis. Every entry traces back to "
            "something explicit upstream.\n"
            "- DUPLICATE the prose of target_population. The prose "
            "describes the population as a whole; this list "
            "decomposes the population into its independent axes."
        ),
    )
    dimensions: list[Dimension] = Field(
        ...,
        description=(
            "Translation of the aspects list above into concrete "
            "database-vocabulary checks. Each dimension addresses "
            "one or more aspects — the searchable counterpart of "
            "the user-vocabulary axis (or related group of axes) "
            "above. The trait_role_analysis above constrains what "
            "kind of check each translation produces; the aspects "
            "list above constrains what must be covered.\n"
            "\n"
            "How many: at most one per aspect — typically the same "
            "count as aspects, occasionally fewer when two aspects "
            "share one database-vocabulary check. The system "
            "prompt's DIMENSION INVENTORY section covers the "
            "category-aware decomposition rule and the coverage "
            "discipline.\n"
            "\n"
            "COVERAGE: every aspect addressed by at least one "
            "dimension; every dimension traces to at least one "
            "aspect. An aspect with no dimension is dropped "
            "coverage; a dimension with no aspect is invented.\n"
            "\n"
            "NEVER:\n"
            "- PAD. Don't add dimensions that don't trace to an "
            "aspect; don't split one aspect across multiple "
            "dimensions when one searchable check covers it.\n"
            "- ABSTRACT. Each dimension names one specific check "
            "the database can run.\n"
            "- VIOLATE trait_role_analysis. If the analysis "
            "constrains dimensions to describe a reference's "
            "identifiable attributes, do not slip in a "
            "population-describing dimension.\n"
            "- DROP AN ASPECT silently. If an aspect resists "
            "translation, the aspect itself was wrong (too "
            "abstract, not actually a separate axis) — go back and "
            "revise aspects rather than skipping it here."
        ),
    )
    combine_mode: TraitCombineMode = Field(
        ...,
        description=(
            "Closed-enum commit of how stage-4 will fold this trait's "
            "per-category scores into a single trait_score. "
            "PROCEDURALLY committed AFTER candidate analysis and "
            "BEFORE category_calls — the choice of mode shapes which "
            "categories make sense to commit.\n"
            "\n"
            "FRAMINGS — categories are alternative homes for the same "
            "underlying thing. Matching ANY ONE is sufficient evidence "
            "of the criterion. Stage-4 MAX-folds the per-category "
            "scores; redundant categories REINFORCE each other as "
            "alternative routes to the same signal. Use when the "
            "candidate analysis shows multiple categories converging "
            "on equivalent meanings (e.g. an identity that has clean "
            "homes in two adjacent categories).\n"
            "\n"
            "FACETS — categories cover DIFFERENT axes of a compound "
            "concept. ALL facets must fire to a degree for the "
            "criterion to be met. Stage-4 PRODUCT-folds the per-"
            "category scores; duplicating axis coverage AMPLIFIES the "
            "wrong signals. Use when the candidate analysis shows "
            "each dimension's clean-fit category covering a distinct "
            "identifiable axis the user wants compounded.\n"
            "\n"
            "OPERATIONAL TEST: read the candidate analysis. \"Do the "
            "dimensions cover ONE underlying thing through alternative "
            "homes (FRAMINGS), or do they cover SEVERAL axes the user "
            "wants compounded (FACETS)?\" If a fresh reader would "
            "expect a movie matching only one category to satisfy the "
            "trait → FRAMINGS. If a fresh reader would expect a movie "
            "to match across several categories together → FACETS.\n"
            "\n"
            "Single-dimensional traits commit FRAMINGS by default — "
            "with one category, MAX and PRODUCT collapse to passthrough.\n"
            "\n"
            "NEVER:\n"
            "- COMMIT FACETS WHEN CATEGORIES ARE FRAMINGS. Marvel = "
            "STUDIO_BRAND ∨ FRANCHISE_LINEAGE is FRAMINGS — both fire "
            "1.0 on an MCU film and matching either is sufficient. "
            "PRODUCT-folding would penalize the reference for failing "
            "to surface in BOTH categories simultaneously.\n"
            "- COMMIT FRAMINGS WHEN CATEGORIES ARE FACETS. Bro movie "
            "= STORY_THEMATIC_ARCHETYPE ∧ EMOTIONAL_EXPERIENTIAL ∧ "
            "NARRATIVE_DEVICES is FACETS — the user wants all three "
            "axes to compound. MAX-folding would let single-facet "
            "matches win at 1.0.\n"
            "- DEFAULT-FILL. The mode is a real commit driven by the "
            "candidate analysis, not a placeholder."
        ),
    )
    category_calls: list[CategoryCall] = Field(
        ...,
        description=(
            "The committed minimum-CALL-count routings. One call "
            "per category that ends up owning >=1 expression; when "
            "several dimensions share a best-fit category, they "
            "merge into ONE multi-expression call.\n"
            "\n"
            "MODE-AWARE ROUTING. Reads combine_mode (committed "
            "above):\n"
            "- FRAMINGS authorizes committing categories whose "
            "coverage OVERLAPS — the system MAX-folds them, so "
            "redundancy reinforces as alternative routes to the same "
            "signal. Multiple framings of one identity (e.g. "
            "STUDIO_BRAND + FRANCHISE_LINEAGE for a brand) are "
            "intentional, not padding.\n"
            "- FACETS DEMANDS choosing categories that COMPLEMENT "
            "rather than overlap — the system PRODUCT-folds them, so "
            "duplication of axis coverage amplifies the wrong "
            "signals. Each committed category should cover a "
            "distinct axis surfaced by the dimension list.\n"
            "\n"
            "OPERATIONAL TESTS:\n"
            "- COVERAGE. Every Dimension.expression in the "
            "inventory is owned by exactly one entry's expressions "
            "field. Zero owners → gap. Two → duplication.\n"
            "- MINIMUM-CALL. If I removed this call, would the "
            "remaining calls still cover its dimensions? If yes, "
            "merge or drop.\n"
            "- CANDIDATE-LINK. Every category here appeared as a "
            "candidate on at least one of its owned dimensions.\n"
            "- MODE-CONSISTENT. For FACETS, no two committed "
            "categories cover the SAME axis of the trait. For "
            "FRAMINGS, overlapping coverage across categories is "
            "permitted and often correct.\n"
            "\n"
            "NEVER:\n"
            "- EMIT A CALL TO A CATEGORY THAT WAS NOT A CANDIDATE "
            "on any of its owned dimensions.\n"
            "- LEAVE A DIMENSION UNCOVERED.\n"
            "- DUPLICATE A CATEGORY. Same-category dimensions merge "
            "into ONE multi-expression call.\n"
            "- PAD. Single-dimensional trait → ONE call with ONE "
            "expression."
        ),
    )
