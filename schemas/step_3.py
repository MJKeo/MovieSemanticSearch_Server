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

from schemas.enums import CandidateFit, TraitCombineMode
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
    fit: CandidateFit = Field(
        ...,
        description=(
            "Commit, AFTER writing the covers/misses analysis above, "
            "how strongly this category fits the dimension. This label "
            "is what the consolidation step reads to separate real fits "
            "from floor-filler without lowering the candidate floor — so "
            "it must be an honest assessment, not a commitment "
            "(commitment happens later in category_calls).\n"
            "\n"
            "- CLEAN_OWNERSHIP: the category squarely owns the "
            "dimension — what_this_misses found no substantive gap, and "
            "the dimension can be expressed as the PRESENCE of an "
            "attribute to find.\n"
            "- COULD_CONSOLIDATE: a real neighbor that could host the "
            "dimension or be absorbed alongside a stronger category, but "
            "is not the obvious sole home.\n"
            "- LIKELY_DISREGARD: weak or indefensible coverage, surfaced "
            "mainly to meet the floor. ALSO commit this when the only "
            "way to apply the category here would be to describe an "
            "ABSENCE (what should NOT be present) rather than the "
            "presence of an attribute — such a category cannot cleanly "
            "own the dimension no matter how well it matches topically.\n"
            "\n"
            "Rank against the category's OWN taxonomy definition: if "
            "applying it to this dimension would violate that category's "
            "boundary, or another category's definition is honestly the "
            "better home for the aspect's intent, drop it below "
            "CLEAN_OWNERSHIP (COULD_CONSOLIDATE, or LIKELY_DISREGARD if "
            "indefensible). Reserve CLEAN_OWNERSHIP for the category whose "
            "definition squarely claims this aspect."
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
            "- EXPRESS ABSENCE. Name the attribute as present, never "
            "its absence — even when the trait is about steering away "
            "from something.\n"
            "- INVENT AN EXPRESSION not present in aspects. Every "
            "dimension traces to exactly one aspect string."
        ),
    )
    category_candidates: list[CategoryCandidate] = Field(
        ...,
        min_length=5,
        description=(
            "EXPLORATORY ROUTING LIST — at least 5 plausible "
            "categories per dimension, each with what-this-covers / "
            "what-this-misses prose. The floor of 5 is deliberate: "
            "it forces you to look past the obvious clean fit and "
            "surface adjacent categories that prior knowledge might "
            "otherwise skip. Expect that some of the 5 will be only "
            "weak fits; the routing-exploration step that follows is "
            "where filler gets pruned. This list is rough by design — "
            "honest exploration here, ruthless pruning later.\n"
            "\n"
            "For each candidate, write substantive what_this_covers / "
            "what_this_misses prose ANCHORED IN THE TAXONOMY. Even "
            "for a weak fit, name the specific aspect the candidate "
            "doesn't cover and which category the boundary redirects "
            "to. That prose is what the routing step reads to decide "
            "which candidates are real adjacencies vs filler — vague "
            "or fabricated coverage defeats the pruning step.\n"
            "\n"
            "NEVER:\n"
            "- DUPLICATE CATEGORIES per dimension.\n"
            "- FABRICATE COVERAGE. If a candidate truly has no "
            "overlap with the dimension, what_this_covers must say "
            "so plainly. Padding the prose to dress up filler robs "
            "the routing step of the signal it needs to drop the "
            "candidate.\n"
            "- COMMIT HERE. This layer surfaces options; the "
            "category_calls layer commits the minimum useful subset."
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
            "- DESCRIBE ABSENCE. Describe attributes the film HAS / "
            "what you want to see — never what it lacks, avoids, or is "
            "free of. Even when the trait is about steering away from "
            "something, name that thing as present. A category you "
            "could only apply here by describing an absence should not "
            "have been committed.\n"
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
            "Flat enumeration of the distinct PARTS the trait "
            "decomposes into. One short noun-phrase per entry, in "
            "user-vocabulary or natural English. This is the "
            "enumeration step; the dimensions step below routes each "
            "entry to candidate categories.\n"
            "\n"
            "Decompose into parts that are (1) conceptually DISTINCT "
            "and NON-OVERLAPPING — no two entries name the same axis "
            "from different angles, and no entry restates the whole "
            "trait while other entries cover its components; and "
            "(2) COLLECTIVELY COMPREHENSIVE — taken together the parts "
            "reconstruct the trait with nothing important left out. "
            "Enumerate the PARTS, not the whole: reassembling them into "
            "the minimum set of searches is the consolidation step's "
            "job downstream, not this step's. Breaking one concept into "
            "a few genuine, distinct pieces is safe — consolidation "
            "corrects over-enumeration later, but a part you never "
            "enumerate cannot be recovered. Cardinality follows what "
            "the trait actually says: a single-axis trait resolves to "
            "one part; a trait with several simultaneous conditions "
            "resolves to one part per condition.\n"
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
            "- ENUMERATE THE WHOLE ALONGSIDE ITS PARTS. If one entry "
            "would restate the entire trait and other entries name "
            "its components, keep only the components (or only the "
            "whole) — never both. Overlap between entries is the "
            "defect this step most needs to avoid.\n"
            "- TRANSLATE INTO CATEGORY VOCABULARY. Categories live "
            "in dimensions / candidates / calls. A part is a "
            "user-side axis, not a routing slot.\n"
            "- COLLAPSE genuinely distinct parts into one phrase, or "
            "manufacture parts to look richer than the trait is.\n"
            "- INVENT parts not grounded in target_population or "
            "trait_role_analysis. Every entry traces back to "
            "something explicit upstream.\n"
            "- DUPLICATE the prose of target_population. The prose "
            "describes the whole; this list decomposes it into "
            "distinct parts."
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
    consolidation_analysis: str = Field(
        ...,
        description=(
            "EXPLORATORY THINKING — produced AFTER dimensions and "
            "BEFORE combine_mode. EXPLORE the routing options across the "
            "whole candidate set FIRST, then CONCLUDE where this trait "
            "lands on the spectrum from a single combined shape to a "
            "breadth of distinct calls. Do NOT open with the verdict and "
            "rationalize backward — exploration leads the conclusion, or "
            "the call list ends up reverse-engineered to fit a path you "
            "locked in too early.\n"
            "\n"
            "Read the candidate fit labels as the starting signal: drop "
            "every LIKELY_DISREGARD outright; treat CLEAN_OWNERSHIP "
            "candidates as the anchors; and genuinely weigh each "
            "COULD_CONSOLIDATE candidate — it may be absorbed alongside a "
            "stronger category, dropped, or kept as its own call, "
            "whichever the evidence supports. This is real exploration, "
            "not a default-to-absorb with a token fallback.\n"
            "\n"
            "Then commit the MINIMUM set of calls that represents the "
            "trait. The trait is a single concept (cross-concept "
            "splitting already happened upstream), so it is often best "
            "served by one combined shape; reach for an additional call "
            "only when a genuinely important part is not already captured "
            "by the calls you have. A single category should own as many "
            "parts as it cleanly can (one multi-expression call), and a "
            "finer part that a broader call already implies folds into "
            "that call rather than spawning its own. Reassembling the "
            "parts into the whole is THIS step's job.\n"
            "\n"
            "Walk these, in order:\n"
            "\n"
            "(1) DEDUP. Parts whose clean-fit candidate is the same "
            "category merge into one multi-expression call.\n"
            "\n"
            "(2) GRANULARITY. A category that retrieves specific named "
            "entities is admissible only when the part itself names a "
            "specific entity; when the part names a category or kind "
            "(a genre, an archetype, a kind-of-movie defined by what its "
            "members share), such categories are out-of-scope no matter "
            "how cleanly prior knowledge associates named instances with "
            "it — routing a category to a hand-picked subset of named "
            "instances narrows the trait's scope rather than covering "
            "it. Applies to both polarities.\n"
            "\n"
            "(3) PLACE ON THE SPECTRUM. Given the surviving candidates, "
            "decide how many calls are genuinely warranted — from one "
            "combined shape up to a breadth of distinct calls — and "
            "state, for each call, the distinct part(s) it owns and what "
            "it adds that the others don't. A part with no clean home of "
            "its own that a broader call already covers folds in; it does "
            "not earn a brittle separate call.\n"
            "\n"
            "FORWARD REASONING, NOT BACK-RATIONALIZATION. Phrase as "
            "\"I see X across these parts, so I'm leaning toward Y,\" "
            "never \"the answer is Y because X.\"\n"
            "\n"
            "NEVER:\n"
            "- DECIDE BREADTH-VS-SHAPE BEFORE EXPLORING. Explore the "
            "options, then place the trait; the conclusion is earned "
            "here, not assumed.\n"
            "- COMMIT A LIKELY_DISREGARD CANDIDATE, or one whose only "
            "framing would describe an absence.\n"
            "- ADD A CALL THAT REPEATS SIGNAL another call already "
            "carries. Each committed call earns its place by adding "
            "distinct, important coverage.\n"
            "- INTRODUCE NEW NAMED ENTITIES the trait did not name.\n"
            "- LEAVE GENERIC. \"There are several candidates; I'll pick "
            "the best fits\" is a non-analysis."
        ),
    )
    combine_mode: TraitCombineMode = Field(
        ...,
        description=(
            "Closed-enum commit of how stage-4 will fold this trait's "
            "per-category scores into a single trait_score. Reads off "
            "where consolidation_analysis placed the trait on the "
            "breadth-vs-shape spectrum — this field is the mechanical "
            "translation of that conclusion into the closed enum: a "
            "single combined shape → SOLO; a breadth of distinct calls "
            "→ FRAMINGS or FACETS per the relationship below.\n"
            "\n"
            "SOLO — exactly one surviving category cleanly covers "
            "every dimension the trait calls for. Other candidates "
            "surfaced as adjacency context but do not add coverage "
            "the clean primary doesn't already provide. Stage-4 has "
            "nothing to fold; the single category's score IS the "
            "trait_score. When you commit SOLO, category_calls below "
            "contains exactly ONE entry — extra calls under SOLO are "
            "trimmed before retrieval and never reach the endpoints.\n"
            "\n"
            "FRAMINGS — multiple surviving categories are alternative "
            "homes for the same underlying thing, AND no single "
            "category cleanly covers the trait on its own. Matching "
            "ANY ONE is sufficient evidence of the criterion. Stage-4 "
            "MAX-folds the per-category scores; redundant categories "
            "reinforce each other as alternative routes to the same "
            "signal.\n"
            "\n"
            "FACETS — categories cover DIFFERENT axes of a compound "
            "concept. ALL facets must fire to a degree for the "
            "criterion to be met. Stage-4 PRODUCT-folds the per-"
            "category scores; duplicating axis coverage amplifies the "
            "wrong signals.\n"
            "\n"
            "DECISION ORDER. The choice is hierarchical, not a "
            "three-way pick, and it follows the spectrum placement "
            "consolidation_analysis already committed. If that "
            "concluded one combined shape covers the trait → SOLO. "
            "Only when it concluded a breadth of distinct calls is "
            "warranted does the FRAMINGS-vs-FACETS question apply "
            "(alternative homes for one signal → FRAMINGS; distinct "
            "compounding axes → FACETS).\n"
            "\n"
            "NEVER:\n"
            "- DIVERGE FROM consolidation_analysis. If the exploration "
            "concluded one mode, this field commits that mode. If you "
            "want a different answer, revise the exploration above.\n"
            "- DEFAULT-FILL. The mode is a real commit driven by the "
            "exploration's coverage and relationship analysis, not a "
            "placeholder.\n"
            "- HEDGE WITH EXTRA FRAMINGS when one category already "
            "covers the trait cleanly. Adding partial-coverage "
            "adjacents under FRAMINGS does not reinforce; it pads the "
            "call list without strengthening retrieval. The correct "
            "commit when one category suffices is SOLO."
        ),
    )
    category_calls: list[CategoryCall] = Field(
        ...,
        description=(
            "The committed minimum-CALL-count routings. One call "
            "per category that ends up owning >=1 expression; when "
            "several dimensions share a best-fit category, they "
            "merge into ONE multi-expression call. The category set, "
            "the dedup decisions, and the granularity pruning were "
            "all reasoned through in consolidation_analysis above — this "
            "field is the mechanical translation of that conclusion "
            "into structured calls.\n"
            "\n"
            "WHERE FILLER DIES. The category_candidates layer above "
            "carried a floor of 5 candidates per dimension by design, "
            "so the analysis intentionally surfaced some weak fits "
            "alongside the real adjacencies. The consolidation_analysis "
            "step should have already discarded the filler — this "
            "field commits only the categories that survived that "
            "pruning. If a candidate appeared upstream but did not "
            "make the consolidation_analysis's surviving set, it MUST "
            "NOT appear here. The commit list reflects what truly "
            "matters for retrieval, not the breadth of the analysis.\n"
            "\n"
            "MODE-AWARE ROUTING. Reads combine_mode (committed "
            "above):\n"
            "- SOLO means exactly one category cleanly covers the "
            "whole trait. The list contains exactly ONE entry. Any "
            "extras emitted under SOLO are trimmed before retrieval "
            "and never reach the endpoints — list ordering matters, "
            "so place the clean-fit primary first.\n"
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
            "- MODE-CONSISTENT. SOLO → exactly one entry. FACETS → "
            "no two committed categories cover the SAME axis of the "
            "trait. FRAMINGS → overlapping coverage across categories "
            "is permitted and often correct.\n"
            "\n"
            "NEVER:\n"
            "- EMIT A CALL TO A CATEGORY THAT WAS NOT A CANDIDATE "
            "on any of its owned dimensions.\n"
            "- LEAVE A DIMENSION UNCOVERED.\n"
            "- DUPLICATE A CATEGORY. Same-category dimensions merge "
            "into ONE multi-expression call.\n"
            "- PAD. When one category cleanly covers every dimension, "
            "the trait is SOLO with ONE call. Extra entries do not "
            "strengthen retrieval; they are trimmed."
        ),
    )
