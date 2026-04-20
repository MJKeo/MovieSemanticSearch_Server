"""Prompt-rendering helper for the stage-3 studio translator.

The studio translator's LLM needs to map a user's phrasing to one of
the 31 `ProductionBrand` enum values when the query is umbrella-level,
and to emit freeform surface forms otherwise. To pick the right brand
the LLM needs to see which IMDB `production_companies` strings each
brand covers — otherwise it can't tell that "Marvel Studios" maps to
MARVEL_STUDIOS (or DISNEY when umbrella), or that "Touchstone" is
already inside DISNEY.

This module renders a compact registry listing suitable for direct
inclusion in the system prompt. The rendering is deliberately terse:
one line per brand, the enum slug, the display name, and a bounded
sample of surface forms. The goal is coverage-with-anchors, not
completeness — the freeform path catches anything that falls through.

Structural mirror of schemas/award_surface_forms.py — same "render
function returns a string to embed in the prompt" pattern.
"""

from __future__ import annotations

from schemas.production_brands import ProductionBrand


# Cap on surface forms shown per brand in the prompt. Large enough to
# cover the major identity variants (Walt Disney Pictures, Pixar,
# Marvel Studios, etc.) without ballooning the prompt for umbrella
# brands that carry 30+ sub-label strings.
_MAX_SURFACE_FORMS_PER_BRAND = 8


def _unique_surface_forms(brand: ProductionBrand) -> list[str]:
    """Return the distinct IMDB surface strings for a brand, preserving
    the registry's declaration order.

    A brand's `companies` tuple may list the same surface string twice
    when the same name applies across non-overlapping eras (e.g., UA's
    `United Artists` has both a 1919-1981 and a 2024- row). The LLM
    doesn't need the window; it just needs the identity, so we
    deduplicate.
    """
    seen: set[str] = set()
    out: list[str] = []
    for company in brand.companies:
        if company.string in seen:
            continue
        seen.add(company.string)
        out.append(company.string)
    return out


def render_brand_registry_for_prompt() -> str:
    """Render every ProductionBrand as one prompt line.

    Format:
        <slug> — <display_name>
          forms: "<string1>", "<string2>", ... [+N more]

    The `+N more` suffix flags umbrella brands where the truncated
    sample doesn't exhaust the membership — the LLM should understand
    these are catch-all umbrellas covering more strings than listed.
    """
    lines: list[str] = []
    for brand in ProductionBrand:
        forms = _unique_surface_forms(brand)
        shown = forms[:_MAX_SURFACE_FORMS_PER_BRAND]
        form_list = ", ".join(f'"{s}"' for s in shown)
        remainder = len(forms) - len(shown)
        suffix = f" [+{remainder} more]" if remainder > 0 else ""
        lines.append(f"{brand.value} — {brand.display_name}")
        lines.append(f"  forms: {form_list}{suffix}")
    return "\n".join(lines)
