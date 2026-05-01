# Step 4 studio endpoint structured output model.
#
# Translates one Step 3 CategoryCall (expressions: list[str] +
# retrieval_intent: str) into one or more concrete studio searches
# the executor can run against ProductionBrand postings and the
# freeform-token resolver.
#
# Two-layer reasoning:
#   - Top-level exploration commits the studio inventory + combine
#     semantics (reads expressions and retrieval_intent).
#   - Per-StudioRef studio_exploration commits brand-vs-freeform for
#     one named studio (reads top-level + parametric knowledge).
#
# All commit fields read off a prose layer above them — they do not
# re-derive from the upstream inputs. No class-level docstrings or
# Field descriptions on reasoning fields beyond what's needed to
# anchor the read; procedural framing lives in the system prompt.

from pydantic import BaseModel, ConfigDict, Field, conlist, constr

from schemas.endpoint_parameters import EndpointParameters
from schemas.enums import ScoringMethod
from schemas.production_brands import ProductionBrand


# One named studio. Aliases of the same studio collapse into ONE
# StudioRef (variants live in this ref's freeform_names). A different
# studio is a different StudioRef.
class StudioRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "The studio this ref targets. Reads from the studio "
            "inventory committed in the top-level `exploration`, "
            "named exactly as it appeared there. Anchors "
            "`studio_exploration` below to one entity."
        ),
    )

    studio_exploration: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "1-2 sentences. Evidence gathering for THIS studio. "
            "Reads from `name` plus parametric knowledge.\n"
            "\n"
            "Cover: (a) is the user naming a registry parent brand, "
            "or a sub-label / subsidiary / long-tail studio? "
            "(b) what surface forms does this studio appear under "
            "in production credits (acronym / expanded / alternate)?\n"
            "\n"
            "Specificity beats catalog breadth: a sub-label of a "
            "covered brand is NOT the brand. Touchstone is not "
            "Disney; Fox Searchlight is not 20th Century. If the "
            "user named the smaller entity, route freeform.\n"
            "\n"
            "NEVER:\n"
            "- STRETCH TO A BRAND when the user named a sub-label "
            "or subsidiary, even if that sub-label falls under a "
            "covered umbrella.\n"
            "- INVENT ALIASES. Only forms that genuinely appear in "
            "credits.\n"
            "- COMMIT HERE. Evidence only — `brand` and "
            "`freeform_names` carry the commitment."
        ),
    )

    brand: ProductionBrand | None = Field(
        default=None,
        description=(
            "Closed-registry parent brand. Reads from the umbrella-"
            "vs-subsidiary determination in `studio_exploration`.\n"
            "\n"
            "Set ONLY when the user clearly meant the ENTIRE brand "
            "and not a specific subsidiary or sub-label contained "
            "within it. Null when the user named a sub-label, "
            "subsidiary, or long-tail studio — even if that entity "
            "falls under a covered umbrella. The brand path auto-"
            "handles time-bounded ownership and renames — do not "
            "reason about those.\n"
            "\n"
            "Test: would the user accept the WHOLE catalog of this "
            "brand as a hit? Yes → brand. No → null + "
            "`freeform_names`."
        ),
    )

    freeform_names: conlist(
        constr(strip_whitespace=True, min_length=1),
        min_length=0,
        max_length=3,
    ) | None = Field(
        default=None,
        description=(
            "Up to 3 IMDB surface forms for THIS studio — variants "
            "of one entity (acronym / expanded / alternate well-"
            "known form), not different studios. Reads from the "
            "surface-form list in `studio_exploration`.\n"
            "\n"
            "Set when the named entity doesn't match any known brand, "
            "or when it better matches a specific subsidiary / sub-"
            "label of a covered brand. Ignored entirely when `brand` "
            "is also set on the same ref — no fall-through to "
            "freeform when the brand posting is empty. Null/empty "
            "when brand fully covers.\n"
            "\n"
            "Emit forms IMDB credits actually use, not the user's "
            "phrasing. Normalization (case / diacritic / punctuation "
            "/ ordinals) and tokenization (whitespace + hyphens) "
            "run at execution — do NOT pre-normalize, do NOT pad "
            "with spelling variants.\n"
            "\n"
            "Fewer than 3 when fewer distinct forms exist.\n"
            "\n"
            "NEVER:\n"
            "- MIX STUDIOS. One ref = one entity. A second studio "
            "is a second StudioRef.\n"
            "- EXPAND TO LEGAL SUFFIXES the studio doesn't credit "
            "under (\"A24 Films LLC\" when IMDB credits as \"A24\").\n"
            "- TRANSLATE semantically (\"Japan Broadcasting "
            "Corporation\" for NHK) unless IMDB uses the translation."
        ),
    )


# Step 4 studio endpoint output. Reasoning layer commits the studio
# inventory and scoring semantics; commit layer (`studios`,
# `scoring_method`) reads off that prose.
class StudioQuerySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exploration: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "One sentence. Evidence gathering across the call. "
            "Reads PRIMARILY from retrieval_intent (scoring-method "
            "semantics) and expressions (studio inventory).\n"
            "\n"
            "Cover: (a) which distinct studios are named — two "
            "expressions naming the same studio under different "
            "aliases collapse to ONE; (b) does retrieval_intent "
            "frame the studios as alternatives the user is open "
            "to (ANY / or-case) or as joint involvement the user "
            "requires (ALL / all-of)?\n"
            "\n"
            "NEVER:\n"
            "- COMMIT HERE. Evidence only — `studios` and "
            "`scoring_method` carry the commitment.\n"
            "- DOUBLE-COUNT aliases as separate studios.\n"
            "- COLLAPSE genuinely distinct studios into one entry."
        ),
    )

    studios: list[StudioRef] = Field(
        default_factory=list,
        description=(
            "Reads from the studio inventory committed in "
            "`exploration` above. Emit one StudioRef per distinct "
            "studio that prose identified. Aliases of one studio "
            "collapse into ONE ref (variants live in that ref's "
            "`freeform_names`). Empty when no studio resolved — "
            "executor returns an empty result map."
        ),
    )

    scoring_method: ScoringMethod = Field(
        ...,
        description=(
            "How per-ref scores combine into the call's score. "
            "Reads from the scoring-method assessment in `exploration` "
            "above.\n"
            "\n"
            "ANY: we only care if the movie has at least one of the "
            "studio refs. Movies score equally high for matching 1+ "
            "values. Use when retrieval_intent treats the studios as "
            "alternatives (\"indie distributors like A24, Neon, "
            "Mubi\").\n"
            "\n"
            "ALL: we care how many studio refs the movie matches. "
            "Movies score higher depending on how many values they "
            "match. Use when retrieval_intent describes joint "
            "involvement (co-production, overlap).\n"
            "\n"
            "Test: would the user be satisfied if a movie matched "
            "ONLY ONE of the named studios? Yes → ANY. No → ALL.\n"
            "\n"
            "Single-studio calls: ANY (the test collapses)."
        ),
    )


# Category-handler wrapper. role + polarity are stamped post-LLM-
# call from the parent Trait, not declared on this schema (see
# endpoint_parameters.py).
class StudioEndpointParameters(EndpointParameters):
    parameters: StudioQuerySpec = Field(
        ...,
        description=(
            "Studio endpoint payload. One or more named studios "
            "with scoring-method semantics. Per-ref: `brand` for umbrella "
            "registry brands, `freeform_names` for sub-labels and "
            "long-tail. Describe target studios directly regardless "
            "of polarity — negation is on the wrapper's polarity "
            "field, never inside parameters."
        ),
    )
