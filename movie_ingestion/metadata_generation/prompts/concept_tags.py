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

Model: gpt-5-mini, reasoning_effort: medium, verbosity: low
"""

from movie_ingestion.metadata_generation.prompts.concept_tags_assembly import (
    build_system_prompt,
)


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

_TASK = """\
You classify movies into binary concept tags. Each tag answers: \
"does this movie have X?" There are 25 tags across 7 categories. \
Your starting point for every category is EMPTY — add tags that \
input evidence supports.

Every tag still requires input evidence — but when a tag is \
debatable, include it. A missing tag is worse than an extra tag.

---

"""

# ---------------------------------------------------------------------------
# Evidence rules (before tag definitions so ground rules are internalized
# before the model reads category-specific boundaries)
# ---------------------------------------------------------------------------

_EVIDENCE = """\
EVIDENCE RULES

Three confidence levels:

1. Direct evidence — an input explicitly names the concept \
(plot_keyword "revenge", information_control term "plot twist / \
reversal").
2. Concrete inference — a specific plot event or structural detail \
in the input logically implies the concept.
3. Parametric knowledge (fallback only) — for well-known films where \
the concept is culturally unambiguous. Use ONLY at 95%+ confidence. \
If input evidence contradicts parametric knowledge, trust the input.

Genre conventions alone are NEVER sufficient. A horror movie is not \
automatically HAUNTED_LOCATION. A war movie does not automatically \
have a SAD_ENDING.

If you cannot cite specific input content that justifies a tag, \
do not include it.

---

"""

# ---------------------------------------------------------------------------
# Input descriptions
# ---------------------------------------------------------------------------

_INPUTS = """\
INPUTS

- title: Movie title with release year. Use for temporal context \
and parametric knowledge confirmation only.
- plot_keywords: Community-assigned tags. Strong direct signals for \
many tags when present.
- plot_summary (or plot_text): Narrative description. plot_summary \
is LLM-condensed (higher quality); plot_text is raw human-written \
(variable quality). Primary evidence for most tags.
- top_billed_cast: The 5 most important characters and the actors \
who played them, in IMDB billing order, formatted as \
"Character (Actor), ...". This is a prominence ranking — slot 1 \
is the most prominent role. Use it as a CROSS-REFERENCE signal against the plot \
narrative to judge how prominent each named character really is. \
It does NOT by itself determine who the protagonist is (a top-billed \
actor can still play a supporting role, and plot summaries can \
under-mention real leads), but it meaningfully nudges classification. \
Primary signal for FEMALE_LEAD and ENSEMBLE_CAST; useful supporting \
signal for any tag that depends on identifying the protagonist.
- emotional_observations: Audience emotional response from reviewers. \
Reports how audiences actually felt, not what happened in the plot. \
AUTHORITATIVE source for experiential tags (feel_good, tearjerker).
- craft_observations: Reviewer descriptions of narrative structure, \
pacing, and storytelling craft. PRIMARY signal for tags about HOW \
the story is told — nonlinear_timeline, plot_twist, \
unreliable_narrator, breaking_fourth_wall. Reviewers describe \
structural choices here in plain language (e.g. "told in chapters", \
"rug-pull third-act reveal", "directly addresses the camera").
- narrative_technique_terms: Pre-classified structural labels from 5 \
sections (narrative_archetype, narrative_delivery, pov_perspective, \
information_control, additional_narrative_devices). Each section maps \
to specific tags — check the relevant section for each tag. NOTE: \
these labels may use vocabulary that overlaps with concept-tag names \
(e.g. "intercut flashback structure", "evidence-driven reversal"). \
Treat them as descriptive shorthand to investigate, not as direct \
classifications. Cross-reference with craft_observations and \
plot_summary before tagging.
- character_arc_labels: Arc transformation labels from plot_analysis \
(thematic transformations like "naive → wise", "impostor to reconciled \
contributor"). PRIMARY signal for ANTI_HERO disambiguation: an arc \
that lands on a redemptive/moral end-state ("X to moral arrival", \
"criminal to cooperator") is a hard disqualifier for anti_hero even \
if mid-runtime behavior is morally compromised.
- conflict_type: Pre-classified conflict type. Supporting signal for \
plot-archetype tags (revenge, kidnapping) and for judging whether a \
protagonist's moral posture is structural to the conflict.
- parental_guide_items: IMDB content-advisory categories with \
severity ratings (e.g. "Violence Against Animals (severe)", \
"Kidnapping (mild)"). Direct evidence for content-flag tags and a \
supporting signal for plot-archetype tags whose content is \
typically flagged (kidnapping in particular).

When an input is marked "not available", treat it as absent data — \
do not guess what it might contain.

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

JSON matching the provided schema. Each category (except endings) \
is an object with a single "tags" array. The endings category has \
a single "tag" field with exactly one value.

Before emitting the tags for a category, work through each tag in \
that category internally:
  - For each tag you consider, identify the specific input signal \
for or against it (e.g., "plot_keywords include 'revenge' — \
supports REVENGE" or "no input evidence of a twist — no \
PLOT_TWIST").
  - Decide which tags (if any) are supported.

Then populate the tags array with only the supported tags. An \
empty tags array is correct and common. Each category is \
evaluated independently.
"""

# ---------------------------------------------------------------------------
# Assembled prompt — composed by concept_tags_assembly.build_system_prompt()
# which reads _TASK, _EVIDENCE, _INPUTS, _OUTPUT from this module, slots in
# the generated tag-definitions section, and returns the concatenation.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = build_system_prompt()
