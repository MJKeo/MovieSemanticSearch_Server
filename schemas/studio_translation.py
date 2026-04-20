# Step 3 studio endpoint structured output model.
#
# Translates a studio dealbreaker or preference description from step 2
# into a concrete query specification that step 4 can execute against
# the brand-posting and freeform-token paths.
#
# Receives: intent_rewrite (step 1) + one item's description and
# routing_rationale (step 2).
#
# See search_improvement_planning/v2_search_data_improvements.md
# (Query-Time Resolution) for the full design rationale, and
# .claude/plans/ok-this-all-sounds-elegant-melody.md for the
# closed-enum-plus-freeform split this schema encodes.
#
# No class-level docstrings or Field descriptions — all LLM-facing
# guidance lives in the system prompt. Developer notes live in
# comments above the class.

from pydantic import BaseModel, ConfigDict, Field, conlist, constr

from schemas.production_brands import ProductionBrand


# Step 3 studio endpoint output.
#
# Two matching paths:
#   brand_id     — closed enum (31 curated ProductionBrand values).
#                  Set for umbrella queries that name a registry
#                  brand. Executor reads lex.inv_production_brand_postings.
#   freeform_names — up to 3 surface forms likely to appear in IMDB's
#                  production_companies strings. Set for sub-labels,
#                  long-tail studios, or anything not in the registry.
#                  Executor normalizes + tokenizes + does DF-filtered
#                  intersection against lex.studio_token, then joins
#                  the resulting production_company_ids against
#                  movie_card.production_company_ids.
#
# The schema permits either / both / neither. Executor semantics:
#   - brand_id set AND returns non-empty → use brand, ignore freeform.
#   - brand_id unset → use freeform_names.
#   - brand_id set but empty result → fall through to freeform as backup.
#   - neither set → empty result.
#
# Schema ordering is load-bearing. `thinking` is first so the LLM
# reasons about umbrella-vs-specific scope and the correct canonical
# form(s) before committing to brand_id / freeform_names.
class StudioQuerySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Scoped reasoning field. One or two sentences stating whether the
    # query is umbrella / parent-brand (→ brand_id) or specific sub-
    # label / long-tail (→ freeform_names), then naming the registry
    # brand or listing the surface forms the phrasing maps to. LLM-
    # facing guidance lives in the system prompt.
    thinking: constr(strip_whitespace=True, min_length=1) = Field(...)

    # Closed-registry umbrella brand. Use for queries at the parent-
    # brand level (Disney, Warner Bros., A24, MGM, etc.). Null when
    # the query names a sub-label or long-tail studio.
    brand_id: ProductionBrand | None = Field(default=None)

    # Up to 3 IMDB-surface-form strings the LLM believes appear in
    # production_companies credits for the target studio. Empty or
    # null when brand_id covers the query umbrella-style. Up to 3
    # covers condensed/acronym + expanded + alternate well-known form.
    # Executor normalizes + tokenizes each name; a match is
    # intersection-within-name (all discriminative tokens must match
    # one company string) and union-across-names (any name matching
    # counts as a hit).
    freeform_names: conlist(
        constr(strip_whitespace=True, min_length=1),
        min_length=0,
        max_length=3,
    ) | None = Field(default=None)
