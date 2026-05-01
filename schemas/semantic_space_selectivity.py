# Role-keyed vector-space selectivity guidance for the semantic
# endpoint prompt. Rendered into the prompt at import time via
# prompt_builder._ENDPOINT_PLACEHOLDER_RENDERERS so the carver vs.
# qualifier bar can be tuned without rewriting markdown prose.
#
# Carver scoring averages elbow-decayed scores across active spaces
# evenly — every marginal space dilutes the gate. Qualifier scoring
# is a normalized weighted sum — additional spaces round out the
# match without diluting load-bearing ones. The selectivity bar
# follows from those two scoring shapes; if either shape changes,
# this file is the single edit point.

from dataclasses import dataclass

from schemas.enums import Role


@dataclass(frozen=True)
class SelectivityProfile:
    role: Role
    typical_count: str
    bar: str
    weight_rule: str | None = None  # qualifier-only


# Carver first, qualifier second — render order follows tuple order
# so the strict-bar baseline appears before the looser qualifier
# framing. If you reorder, the prompt's pedagogical progression
# inverts.
_PROFILES: tuple[SelectivityProfile, ...] = (
    SelectivityProfile(
        role=Role.CARVER,
        typical_count="1–2 spaces, occasionally 3",
        bar=(
            "Carver execution sums elbow-decayed scores evenly across "
            "active spaces and divides by the count. Every marginal "
            "space directly dilutes the gate, so commit only to "
            "spaces whose signal is genuinely load-bearing for the "
            "trait. A third space is justified only when each of the "
            "three would clearly let a wrong movie pass the trait if "
            "dropped. Prefer one well-targeted space over two thin "
            "ones."
        ),
    ),
    SelectivityProfile(
        role=Role.QUALIFIER,
        typical_count="2–4 spaces, sometimes more",
        bar=(
            "Qualifier scoring is a normalized weighted sum, so "
            "additional spaces round out the match without diluting "
            "the load-bearing ones the way a carver would. Fanning "
            "out is fine when each added space genuinely captures "
            "part of the trait — still drop spaces whose signal is "
            "below SUPPORTING (barely-there)."
        ),
        weight_rule=(
            "CENTRAL for spaces whose signal `retrieval_intent` names "
            "as defining the match. SUPPORTING for spaces that round "
            "out the experience. All-supporting is acceptable for "
            "broad-and-balanced traits — a truthful signal, not a "
            "cop-out."
        ),
    ),
)


def render_semantic_selectivity_for_prompt() -> str:
    lines: list[str] = []
    for profile in _PROFILES:
        lines.append(f"### Role: {profile.role.value.title()}")
        lines.append(f"Typical commit: {profile.typical_count}.")
        lines.append("")
        lines.append(profile.bar)
        if profile.weight_rule:
            lines.append("")
            lines.append(f"Weight rule: {profile.weight_rule}")
        lines.append("")
    return "\n".join(lines).rstrip()
