"""
System prompt for concept tag generation.

Multi-label binary classification of 25 concept tags across 7 categories.
Each tag answers a yes/no question: "does this movie have X?"

The model evaluates each category independently and outputs only the tags
that are supported by input evidence. Empty categories are correct and
expected.

Prompt structure: task → evidence rules → input descriptions → tag
definitions (programmatically generated) → output format.

This module owns the four hand-written framing blocks (_TASK, _EVIDENCE,
_INPUTS, _OUTPUT). The tag-definitions section is assembled by
[concept_tags_assembly.py](concept_tags_assembly.py) from per-tag
attributes on the `ConceptTag` master enum and per-category attributes
on `ConceptTagCategory` (see [schemas/enums.py](../../../../schemas/enums.py)).
That enum is the single source of truth for tag descriptions, selection
criteria, boundary cases, category intros, cross-tag relationship notes,
and long-form reasoning blocks (currently FEMALE_LEAD's three-step
logic and the ENDINGS HOW-TO).

The output schema requires a `reasoning` field in every per-category
Assessment object, emitted BEFORE the tag list (or single ending tag).
The reasoning field forces the model to walk through the evidence
before concluding, combating post-hoc justification. Each Assessment's
reasoning description (in [schemas/metadata.py](../../../../schemas/metadata.py))
names the primary evidence sources for that category.

Evidence rules are placed before tag definitions so the model
internalizes ground rules before reading category-specific boundaries.

Model: gpt-5-mini, reasoning_effort: minimal, verbosity: low
"""

from movie_ingestion.metadata_generation.prompts.concept_tags_assembly import (
    build_system_prompt,
)


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

_TASK = """\
Classify a movie into binary concept tags answering "does this movie have X?" \
25 tags / 7 categories. Start EMPTY per category; add only tags input evidence \
supports. When debatable, include — a missing tag is worse than an extra (FEMALE_LEAD excepted).

---

"""

# ---------------------------------------------------------------------------
# Evidence rules (before tag definitions so ground rules are internalized
# before the model reads category-specific boundaries)
# ---------------------------------------------------------------------------

_EVIDENCE = """\
EVIDENCE RULES

Confidence levels (best to worst):
1. Direct — an input explicitly names the concept (e.g. plot_keyword "revenge").
2. Inference — a specific plot/structural detail logically implies it.
3. Parametric — only at 95%+ confidence for culturally unambiguous well-known films; \
input overrides parametric on conflict.

Genre alone is NEVER sufficient (horror ≠ HAUNTED_LOCATION; war ≠ SAD_ENDING). \
If you cannot cite specific input content, do not tag.

---

"""

# ---------------------------------------------------------------------------
# Input descriptions
# ---------------------------------------------------------------------------

_INPUTS = """\
INPUTS

- title: title + year; for temporal context / parametric confirmation only.
- plot_keywords: community-assigned tags; strong direct signals when present.
- plot_summary (or plot_text): narrative description. plot_summary is LLM-condensed \
(higher quality); plot_text is raw human (variable). Primary evidence for most tags.
- top_billed_cast: top-5 characters+actors in IMDB billing order ("Character (Actor), ..."); \
slot 1 = most prominent. CROSS-REFERENCE only — does not by itself determine the protagonist \
(top-billed can play support; plot_summary can under-mention real leads). Primary for FEMALE_LEAD \
and ENSEMBLE_CAST.
- emotional_observations: reviewer reports of how audiences FELT, not what happened. \
AUTHORITATIVE for experiential tags (feel_good, tearjerker).
- craft_observations: reviewer descriptions of structure/pacing/craft. PRIMARY for \
HOW-it's-told tags (nonlinear_timeline, plot_twist, unreliable_narrator, breaking_fourth_wall) — \
e.g. "told in chapters", "rug-pull third-act reveal", "directly addresses the camera".
- narrative_technique_terms: pre-classified labels from 5 sections (narrative_archetype, \
narrative_delivery, pov_perspective, information_control, additional_narrative_devices). \
Vocabulary may overlap with tag names ("intercut flashback structure", "evidence-driven reversal") — \
treat as shorthand to INVESTIGATE, not direct classifications; cross-reference craft_observations + \
plot_summary before tagging.
- character_arc_labels: arc transformation labels from plot_analysis ("naive → wise", \
"impostor to reconciled contributor"). Supporting signal for ANTI_HERO (a redemptive arc still \
indicates the character WAS anti-heroic earlier).
- conflict_type: pre-classified conflict; supports plot-archetype tags (revenge, kidnapping) \
and judging whether a protagonist's moral posture is structural.
- parental_guide_items: IMDB content advisories with severity ("Violence Against Animals (severe)", \
"Kidnapping (mild)"); direct for content-flag tags, corroborating for archetype tags (kidnapping).

"not available" = absent data — do not guess.

---

"""

# ---------------------------------------------------------------------------
# Tag definitions section — generated from ConceptTag / ConceptTagCategory
# by concept_tags_assembly.build_tag_definitions_section(). The hand-
# written content that used to live here has been ported into the enum
# members (selection_criteria, boundary_cases, long_form_instructions).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------

_OUTPUT = """\
OUTPUT FORMAT

JSON matching the schema. Each category (except endings) is an object with a "tags" \
array; endings has a single "tag" field with exactly one value.

For every category, internally consider each tag, cite the specific input signal \
for/against (e.g. "plot_keywords 'revenge' → supports REVENGE"; "no twist evidence → no \
PLOT_TWIST"), then emit only supported tags. Empty arrays are correct and common. \
Categories evaluated independently.
"""

# ---------------------------------------------------------------------------
# Assembled prompt — composed by concept_tags_assembly.build_system_prompt()
# which reads _TASK, _EVIDENCE, _INPUTS, _OUTPUT from this module, slots in
# the generated tag-definitions section, and returns the concatenation.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = build_system_prompt()
