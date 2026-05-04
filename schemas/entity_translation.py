# Stage 3 entity-endpoint structured-output schemas.
#
# Three category-scoped output schemas — PersonQuerySpec,
# CharacterQuerySpec, and TitlePatternQuerySpec — handed to the
# handler LLM by the per-category schema selector. The LLM never
# picks which family it sits in; routing is decided upstream by the
# step 3 category.
#
# Multi-target shape: each spec carries a list of targets. Two
# distinct entities in one call become two targets; credited aliases
# of ONE entity become multiple `forms` inside ONE target. Result
# scores merge by MAX per movie across targets AND across forms
# (union semantics — "any of these wins").
#
# Reasoning fields scaffold each commitment that follows them. They
# are read first by the LLM and ignored by the executor. Description
# vs interpretation: reasoning fields describe / explore; commitment
# fields commit.
#
# Role + polarity are NOT carried here. They are upstream commitments
# living on the parent Trait; the handler stamps them onto a wrapper
# at execution time. This module's specs are positive-presence
# search payloads only — every target describes what to find, never
# what to exclude.
#
# No class-level docstrings or external Field descriptions — all
# LLM-facing guidance lives inline. Developer notes live in comments
# above each class.

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, conlist, constr

from schemas.endpoint_parameters import EndpointParameters


# ---------------------------------------------------------------------
# Enums (local to this module — replaces the unified PersonCategory /
# ProminenceMode / TitlePatternMatchType from schemas/enums.py for
# entity-endpoint use). Each enum is genuinely exhaustive over its
# slot, so closed Literals are the right shape (principle 1).
# ---------------------------------------------------------------------


# Posting-table pick for a person target. UNKNOWN searches all five
# tables with even weight when the LLM does not recognize the person —
# strict fallback, not a default.
class PersonCategory(StrEnum):
    ACTOR = "actor"
    DIRECTOR = "director"
    WRITER = "writer"
    PRODUCER = "producer"
    COMPOSER = "composer"
    UNKNOWN = "unknown"


# Billing-band scoring for actor-table searches. Executor ignores this
# field for non-actor person_category values, so no special "pick X
# when not actor" guidance is needed in the prompt.
class PersonProminenceMode(StrEnum):
    DEFAULT = "default"
    LEAD = "lead"
    SUPPORTING = "supporting"
    MINOR = "minor"


# Centrality scoring for character searches.
class CharacterProminenceMode(StrEnum):
    DEFAULT = "default"
    CENTRAL = "central"


# Title-string match strategy. EXACT_MATCH is new (full-title equality);
# CONTAINS / STARTS_WITH map to the existing LIKE pattern paths.
class TitleMatchType(StrEnum):
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    EXACT_MATCH = "exact_match"


# ---------------------------------------------------------------------
# Person query
# ---------------------------------------------------------------------


# One named real person to look up. `forms` enumerates credited
# variants of THIS person only — different people are different
# targets, never aliases of each other.
class PersonTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person_exploration: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Fill in this template exactly. No commentary outside it.\n"
            "\n"
            "Films: <3-5 notable films this person worked on, with year>\n"
            "Credit per film:\n"
            "  - <film>: <every form this person is billed under in that "
            "film's cast/crew block, comma-separated, one identity per "
            "entry>\n"
            "  - ...\n"
            "Distinct forms: <deduped union of all credit-per-film "
            "entries, comma-separated, most common form first>\n"
            "Predominant role: <actor | director | writer | producer | "
            "composer | not sure>\n"
            "\n"
            "A 'form' is a single atomic identity — one name per form. "
            "If a literal credit bundles multiple identities (slash-"
            "combined names, or a stage name embedded inside a legal "
            "name), split it into its underlying single-identity "
            "components when listing the film's credited forms. "
            "Retrieval is exact string match against atomic credit "
            "entries; bundled strings match nothing.\n"
            "\n"
            "Skip a film if you can't recall the credit — absence "
            "beats fabrication. 'Not sure' on role triggers UNKNOWN "
            "downstream; use it instead of guessing.\n"
            "\n"
            "The queried surface form is often NOT the dominant "
            "credit — a stage or alternate name may not appear as "
            "the primary credit on most cast blocks. Walk widely "
            "enough to surface every distinct credited form."
        ),
    )

    forms: conlist(
        constr(strip_whitespace=True, min_length=1), min_length=1
    ) = Field(
        ...,
        description=(
            "Every atomic name from person_exploration's 'Distinct "
            "forms' line for THIS one person. Most common form "
            "first; rest are aliases of the SAME person. Retrieval "
            "is exact-string MAX across forms — extras cost ~0, "
            "omissions silently drop films. Bias toward inclusion "
            "of any atomic form grounded in a specific film credit.\n"
            "\n"
            "Different people → separate targets, not aliases. Skip "
            "descriptive phrases, scene quotes, and fan nicknames — "
            "only credited names. Skip diacritic / casing / "
            "punctuation / hyphenation variants — normalization "
            "handles those."
        ),
    )

    person_category: PersonCategory = Field(
        ...,
        description=(
            "Posting table to query. Pick a specific role when "
            "retrieval_intent or expression names/implies it, or when "
            "person_exploration committed to one. UNKNOWN only when "
            "exploration ended in 'not sure' — it unions all five "
            "tables evenly and is a fallback, not a default.\n"
            "\n"
            "Local test: would I want all five tables hit evenly? If "
            "no, UNKNOWN is wrong."
        ),
    )

    prominence_exploration: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Spartan: quote the prominence language in "
            "retrieval_intent or expression signaling LEAD / "
            "SUPPORTING / MINOR, or state 'no prominence signal'. "
            "Quote first; do not justify a mode already picked."
        ),
    )

    prominence_mode: PersonProminenceMode = Field(
        ...,
        description=(
            "Billing-band score. DEFAULT = no prominence language "
            "(typical). LEAD / SUPPORTING / MINOR = explicit "
            "leading / supporting / cameo language quoted in "
            "prominence_exploration. Fame is not a LEAD signal.\n"
            "\n"
            "Local test: did prominence_exploration quote a phrase "
            "pinning this band? If no, DEFAULT."
        ),
    )


# Per-call container. Multi-target = union of distinct people; per-
# movie score is MAX across targets. Inherits from EndpointParameters
# so the orchestrator's isinstance-routing on `preference_specs` can
# dispatch this directly to the entity executor — there is no
# wrapping `EntityEndpointParameters` layer.
class PersonQuerySpec(EndpointParameters):
    query_exploration: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Spartan: state whether retrieval_intent and expressions "
            "name ONE person, MULTIPLE distinct people, or one "
            "person under several variants. N expressions ≠ N "
            "targets — variants collapse into one target's `forms`; "
            "different people become separate targets.\n"
            "\n"
            "Local test: same individual or different? Same → one "
            "target. Different → multiple."
        ),
    )

    targets: conlist(PersonTarget, min_length=1) = Field(
        ...,
        description=(
            "One PersonTarget per distinct person identified in "
            "query_exploration. Scores merge by MAX per movie across "
            "targets — union semantics ('any of these people wins'), "
            "never intersection."
        ),
    )


# ---------------------------------------------------------------------
# Character query
# ---------------------------------------------------------------------


# One fictional character to look up. Same shape as PersonTarget but
# scoped to character credit strings — civilian / secret-identity
# pairings, alternate-incarnation names, longer-vs-shorter variants.
class CharacterTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character_exploration: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Fill in this template exactly. No commentary outside it.\n"
            "\n"
            "Films: <the most popular / relevant films featuring this "
            "character, with year — walk enough to surface every "
            "distinct credited form>\n"
            "Credit per film:\n"
            "  - <film>: <every form this character is credited under "
            "in that film's cast block, comma-separated, one identity "
            "per entry>\n"
            "  - ...\n"
            "Distinct forms: <deduped union of all credit-per-film "
            "entries, comma-separated, most common form first>\n"
            "\n"
            "A 'form' is a single atomic identity — one name per form. "
            "If a literal cast-list credit bundles multiple identities "
            "(slash-combined names, or alternate identities listed "
            "together), split it into its underlying single-identity "
            "components when listing the film's credited forms. "
            "Retrieval is exact string match against atomic credit "
            "entries; bundled strings match nothing.\n"
            "\n"
            "Skip a film if you can't recall the credit — absence "
            "beats fabrication. Skip scene quotes, fan nicknames, "
            "and descriptive phrases; only credited names.\n"
            "\n"
            "A character may be credited under several distinct forms "
            "within a single film, and under different forms across "
            "reboots or incarnations. Walk widely enough to surface "
            "every distinct form; emitting only the queried form "
            "silently drops every film credited differently."
        ),
    )

    forms: conlist(
        constr(strip_whitespace=True, min_length=1), min_length=1
    ) = Field(
        ...,
        description=(
            "Every atomic name from character_exploration's 'Distinct "
            "forms' line for THIS one character. Most common form "
            "first; rest are aliases of the SAME character. "
            "Retrieval is exact-string MAX across forms — extras "
            "cost ~0, omissions silently drop films. Bias toward "
            "inclusion of any atomic form grounded in a specific "
            "film credit.\n"
            "\n"
            "Different characters → separate targets, not aliases. "
            "Skip generic role labels and descriptive phrases — only "
            "identifiable names. Skip diacritic / casing / "
            "punctuation / hyphenation variants — normalization "
            "handles those."
        ),
    )

    prominence_exploration: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Spartan: quote the centrality language in "
            "retrieval_intent or expression signaling subject-of-"
            "film framing (possessive title subject, story-of, "
            "centers-on), or state 'no centrality signal'. Quote "
            "first; do not justify a mode."
        ),
    )

    prominence_mode: CharacterProminenceMode = Field(
        ...,
        description=(
            "Centrality score. DEFAULT = named without centrality "
            "language (typical). CENTRAL = explicit subject-of-film "
            "language quoted in prominence_exploration. Fame is not "
            "a CENTRAL signal.\n"
            "\n"
            "Local test: did prominence_exploration quote a phrase "
            "framing this character as the film's subject? If no, "
            "DEFAULT."
        ),
    )


# Per-call container. See PersonQuerySpec for the rationale behind
# inheriting from EndpointParameters.
class CharacterQuerySpec(EndpointParameters):
    query_exploration: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Spartan: state whether retrieval_intent and expressions "
            "name ONE character, MULTIPLE distinct characters, or "
            "one character under several variants. N expressions ≠ "
            "N targets — variants collapse into one target's "
            "`forms`; different characters become separate targets.\n"
            "\n"
            "Local test: same character or different? Same → one "
            "target. Different → multiple."
        ),
    )

    targets: conlist(CharacterTarget, min_length=1) = Field(
        ...,
        description=(
            "One CharacterTarget per distinct character identified "
            "in query_exploration. Scores merge by MAX per movie "
            "across targets — union semantics, never intersection."
        ),
    )


# ---------------------------------------------------------------------
# Title-pattern query
# ---------------------------------------------------------------------


# One literal title-text fragment with a match strategy.
class TitlePatternTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern_exploration: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Spartan: quote the language in retrieval_intent or "
            "expression signaling the match kind — substring-"
            "anywhere, title-prefix, or exact full-title equality. "
            "Quote first; do not name match_type yet.\n"
            "\n"
            "Do not default to CONTAINS without checking. "
            "STARTS_WITH and EXACT_MATCH have specific signals; "
            "missing them is the failure mode."
        ),
    )

    pattern: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Literal text fragment to match. Draw exact characters "
            "from the language pattern_exploration quoted; strip "
            "framing words the user wrapped the fragment in.\n"
            "\n"
            "No '%' or '_' wildcards — executor handles LIKE "
            "escaping. No surrounding quotes — the user's quotes "
            "are not part of the search text."
        ),
    )

    match_type: TitleMatchType = Field(
        ...,
        description=(
            "How `pattern` is compared. CONTAINS = appears anywhere "
            "(typical). STARTS_WITH = title begins with pattern. "
            "EXACT_MATCH = entire title equals pattern. Pick the "
            "value matching the language pattern_exploration "
            "quoted.\n"
            "\n"
            "Local test: did pattern_exploration quote language "
            "specific to this match_type? If no, CONTAINS."
        ),
    )


# Per-call container. See PersonQuerySpec for the rationale behind
# inheriting from EndpointParameters.
class TitlePatternQuerySpec(EndpointParameters):
    query_exploration: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Spartan: state whether retrieval_intent and expressions "
            "name ONE pattern or MULTIPLE distinct ones (e.g. a list "
            "of words any of which would qualify). N expressions ≠ "
            "N targets.\n"
            "\n"
            "Local test: does each expression name a separate match "
            "the user would accept on its own? Yes → multiple. No → "
            "one."
        ),
    )

    targets: conlist(TitlePatternTarget, min_length=1) = Field(
        ...,
        description=(
            "One TitlePatternTarget per distinct pattern identified "
            "in query_exploration. Scores merge by MAX per movie "
            "across targets — union semantics."
        ),
    )


# No EntityEndpointParameters wrapper exists for this endpoint. The
# three category-scoped specs above (PersonQuerySpec / CharacterQuerySpec
# / TitlePatternQuerySpec) are themselves EndpointParameters subclasses
# and are selected per category by endpoint_registry._ENTITY_DISPATCH.
# This is the analog of how SEMANTIC dispatches per role — entity
# dispatches per CategoryName.
