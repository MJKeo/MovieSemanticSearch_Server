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
            "Two sentences. (1) Who this person is and which credit-"
            "string forms might appear on real cast/crew listings — "
            "primary common form, legal names rarely used publicly, "
            "stage names or mononyms, longer-vs-shorter credit "
            "variants. (2) Which role this person is predominantly "
            "known for (actor, director, writer, producer, composer); "
            "'not sure' is a valid answer. Describe; do not commit.\n"
            "\n"
            "NEVER:\n"
            "- INVENT forms from feel. If you have no real reason to "
            "believe a variant appears in credits, leave it out.\n"
            "- ROLE-GUESS to fill sentence (2). 'Not sure' is the "
            "signal that triggers UNKNOWN downstream — use it."
        ),
    )

    forms: conlist(
        constr(strip_whitespace=True, min_length=1), min_length=1
    ) = Field(
        ...,
        description=(
            "Every credited string variant for THIS one person, "
            "drawn from the name inventory in person_exploration's "
            "first sentence. Most common credited form first; "
            "remaining entries are aliases of the SAME person. "
            "Retrieval takes MAX score across forms — extra forms "
            "that match nothing cost ~0; omitting a real one "
            "silently drops every film that uses it. Bias toward "
            "inclusion.\n"
            "\n"
            "NEVER:\n"
            "- LIST A DIFFERENT PERSON. Different people are "
            "different targets, not aliases.\n"
            "- LIST DESCRIPTIVE PHRASES, scene quotes, or fan "
            "nicknames that never appear on a credit block.\n"
            "- LIST DIACRITIC / CASING / PUNCTUATION / HYPHENATION "
            "variants — shared normalization handles those.\n"
            "- INVENT middle names or suffixes the person does not "
            "credit under."
        ),
    )

    person_category: PersonCategory = Field(
        ...,
        description=(
            "Posting table to query. Pick a specific role when "
            "retrieval_intent or the expression names or strongly "
            "implies it, OR when person_exploration committed to a "
            "predominant role. Pick UNKNOWN only when "
            "person_exploration ended in 'not sure' — UNKNOWN unions "
            "all five tables with even weight and is a fallback, not "
            "a default.\n"
            "\n"
            "Local test: 'if I removed this commitment, which table "
            "would the executor hit?' If the answer is 'all five "
            "evenly' and that wasn't your intent, you picked UNKNOWN "
            "by mistake."
        ),
    )

    prominence_exploration: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "One sentence. Quote the prominence language in "
            "retrieval_intent or the expression that signals a "
            "LEAD / SUPPORTING / MINOR band, or state 'no prominence "
            "signal'. Explore first; do not justify a mode you have "
            "already picked."
        ),
    )

    prominence_mode: PersonProminenceMode = Field(
        ...,
        description=(
            "Billing-band score. DEFAULT = no prominence language "
            "present (typical case). LEAD = explicit leading-role "
            "language quoted in prominence_exploration. SUPPORTING / "
            "MINOR = explicit supporting / cameo language quoted in "
            "prominence_exploration. Fame of the person is not a "
            "LEAD signal.\n"
            "\n"
            "Local test: 'did prominence_exploration quote a phrase "
            "that pins this band?' If no, the answer is DEFAULT."
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
            "One sentence. Read retrieval_intent and every expression "
            "and state whether they describe ONE person, MULTIPLE "
            "distinct people, or one person under several name "
            "variants. N expressions does NOT imply N targets — "
            "variants of one person collapse into one target's "
            "`forms`; only genuinely different people become "
            "separate targets.\n"
            "\n"
            "Local test: 'are these names the same individual under "
            "different credits, or different individuals?' Same → "
            "one target. Different → multiple targets."
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
            "One sentence. Who this character is and which "
            "credit-string forms film cast lists might use across "
            "their appearances — most common cast-list form, "
            "secret-identity / civilian / legal-name pairings, "
            "alternate-incarnation names from spin-offs or reboots, "
            "longer-vs-shorter variants. Describe the form "
            "inventory; do not commit.\n"
            "\n"
            "NEVER:\n"
            "- INVENT cast-list strings from feel. Stick to forms "
            "you have real reason to believe appear in credits.\n"
            "- LIST SCENE QUOTES, fan nicknames, or descriptive "
            "phrases — they do not appear on a credit block."
        ),
    )

    forms: conlist(
        constr(strip_whitespace=True, min_length=1), min_length=1
    ) = Field(
        ...,
        description=(
            "Every credited string variant for THIS one character, "
            "drawn from the form inventory in character_exploration. "
            "Most common cast-list form first; remaining entries are "
            "aliases of the SAME character. Retrieval takes MAX score "
            "across forms — extras cost ~0, omissions silently drop "
            "films. Bias toward inclusion, but only for IDENTIFIABLE "
            "strings.\n"
            "\n"
            "NEVER:\n"
            "- LIST A DIFFERENT CHARACTER. Different characters are "
            "different targets.\n"
            "- LIST GENERIC ROLE LABELS ('the cop', 'the wizard') "
            "or descriptive phrases — only identifiable name strings "
            "that would appear on a real credit block.\n"
            "- LIST DIACRITIC / CASING / PUNCTUATION / HYPHENATION "
            "variants — shared normalization handles those."
        ),
    )

    prominence_exploration: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "One sentence. Quote the centrality language in "
            "retrieval_intent or the expression that signals "
            "subject-of-film framing (possessive title subject, "
            "story-of, centers-on), or state 'no centrality signal'. "
            "Explore first; do not justify a mode."
        ),
    )

    prominence_mode: CharacterProminenceMode = Field(
        ...,
        description=(
            "Centrality score. DEFAULT = the character is named "
            "without centrality language (typical case). CENTRAL = "
            "explicit subject-of-film language quoted in "
            "prominence_exploration. Fame of the character is not a "
            "CENTRAL signal.\n"
            "\n"
            "Local test: 'did prominence_exploration quote a phrase "
            "framing this character as the film's subject?' If no, "
            "the answer is DEFAULT."
        ),
    )


# Per-call container. See PersonQuerySpec for the rationale behind
# inheriting from EndpointParameters.
class CharacterQuerySpec(EndpointParameters):
    query_exploration: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "One sentence. Read retrieval_intent and every "
            "expression and state whether they describe ONE "
            "character, MULTIPLE distinct characters, or one "
            "character under several name variants. N expressions "
            "does NOT imply N targets — variants of one character "
            "collapse into one target's `forms`; only genuinely "
            "different characters become separate targets.\n"
            "\n"
            "Local test: 'is this the same character under different "
            "credit strings, or a different character?' Same → one "
            "target. Different → multiple targets."
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
            "One sentence. Quote the language in retrieval_intent or "
            "the expression that signals which kind of title match "
            "the user wants — substring-anywhere, title-prefix, or "
            "exact full-title equality. Explore first; do not name "
            "match_type yet.\n"
            "\n"
            "NEVER:\n"
            "- DEFAULT TO CONTAINS without checking. STARTS_WITH and "
            "EXACT_MATCH have specific signals; missing them is the "
            "failure mode."
        ),
    )

    pattern: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Literal text fragment to match. Draw the exact "
            "characters from the language pattern_exploration "
            "quoted; strip any framing words the user wrapped the "
            "fragment in.\n"
            "\n"
            "NEVER:\n"
            "- ADD '%' OR '_' wildcards — the executor handles LIKE "
            "escaping.\n"
            "- WRAP IN QUOTES — the user's quotes are not part of "
            "the search text."
        ),
    )

    match_type: TitleMatchType = Field(
        ...,
        description=(
            "How `pattern` is compared. CONTAINS = appears anywhere "
            "in the title (typical case). STARTS_WITH = title begins "
            "with the pattern. EXACT_MATCH = entire title equals "
            "the pattern. Pick the value matching the language "
            "quoted in pattern_exploration.\n"
            "\n"
            "Local test: 'did pattern_exploration quote language "
            "specific to this match_type?' If no, CONTAINS is the "
            "safe choice."
        ),
    )


# Per-call container. See PersonQuerySpec for the rationale behind
# inheriting from EndpointParameters.
class TitlePatternQuerySpec(EndpointParameters):
    query_exploration: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "One sentence. Read retrieval_intent and every "
            "expression and state whether they describe ONE pattern "
            "or MULTIPLE distinct ones (e.g. a list of words, any of "
            "which would qualify). N expressions does NOT imply N "
            "targets.\n"
            "\n"
            "Local test: 'does each expression name a separate match "
            "the user would accept on its own?' Yes → multiple "
            "targets. No → one target."
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
