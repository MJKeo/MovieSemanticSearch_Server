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
            "One concrete searchable piece in database-vocabulary. "
            "Phrase as one specific check the database could run — "
            "concrete enough that you could imagine writing the "
            "query from this single phrase.\n"
            "\n"
            "Database-vocabulary means: a numeric range or boundary, "
            "a structured-attribute match (credit, format, source), "
            "a tonal / experiential / thematic check the vector "
            "spaces can score against, or a structural plot / "
            "narrative attribute. Concrete and specific to this "
            "trait, not a generic category label.\n"
            "\n"
            "NEVER:\n"
            "- ABSTRACT UP. Vague gestures (\"has good vibes\", "
            "\"feels right\") aren't dimensions. Translate into "
            "something the database can actually check.\n"
            "- NAME A CATEGORY. Categories belong to "
            "category_candidates and category_calls.\n"
            "- EXPRESS ABSENCE. Presence-only — polarity is upstream "
            "and applied at merge time.\n"
            "- BUNDLE multiple unrelated checks. One dimension = one "
            "check. If two checks would have to combine, split into "
            "two."
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
            "1-2 sentences naming the dimensions this call owns and "
            "the presence-of-attribute semantics Step 4 should "
            "encode. When the source trait is a qualifier, encode \n"
            "how Step 4 should treat the retrieval — read the "
            "trait_role_analysis you committed and translate its "
            "operational meaning here (reference being positioned "
            "against, threshold candidates must clear or stay "
            "under, archetype to satisfy, setting to evaluate "
            "inside, etc., depending on what the role analysis "
            "said).\n"
            "\n"
            "NEVER:\n"
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

    target_population: str = Field(
        ...,
        description=(
            "1-2 sentences. Restate, in your own words, the "
            "population of movies this trait wants. What do those "
            "movies share? What can vary freely? Seed for the "
            "dimension inventory.\n"
            "\n"
            "NEVER:\n"
            "- NAME CATEGORIES OR ENDPOINTS.\n"
            "- COPY evaluative_intent VERBATIM. Restate in concrete "
            "population terms; if you're paraphrasing, you're not "
            "adding signal."
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
            "model. Translation into database-vocabulary "
            "(categories, vector spaces, structured fields) happens "
            "in the dimensions step that follows. Keeping these "
            "two steps separate is the point — the model loses "
            "axes when enumeration and translation collapse into "
            "one step.\n"
            "\n"
            "TEST: read this list back. Given only these aspects "
            "and the trait's evaluative_intent, would a fresh "
            "reader produce the same dimensions you're about to "
            "produce? Yes → keep. No → either an aspect is missing "
            "or the wording is too vague.\n"
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
    category_calls: list[CategoryCall] = Field(
        ...,
        description=(
            "The committed minimum-CALL-count routings. One call "
            "per category that ends up owning >=1 expression; when "
            "several dimensions share a best-fit category, they "
            "merge into ONE multi-expression call.\n"
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
