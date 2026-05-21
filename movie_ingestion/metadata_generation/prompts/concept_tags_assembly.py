"""
Programmatic assembly of the concept_tags system prompt.

Source-of-truth tag definitions live as attributes on the `ConceptTag`
master enum in [schemas/enums.py](schemas/enums.py); category-level
intros, cardinality, cross-tag notes, and section-instruction blocks
live on `ConceptTagCategory`. This module reads from both and emits the
tag-definitions section of the system prompt as plain text.

The non-generated parts of the prompt (`_TASK`, `_EVIDENCE`, `_INPUTS`,
`_OUTPUT`) still live in [concept_tags.py](concept_tags.py); this
module only generates the per-tag definitions sandwiched between them.

Layout per category:
    <DISPLAY_LABEL> — <intro_text>     (intro_text omitted when None)

    <cross_tag_note>                   (when present — top-of-section)

    - <TAG_NAME>: <description>
      <long_form_instructions>         (when present — FEMALE_LEAD only)
      Check: <selection_criteria>
      NOT <tag_name>: <boundary_cases>

    <section_instructions>             (when present — endings HOW-TO)

The category iteration order follows `ConceptTagCategory` member
declaration order, which mirrors the original prompt's section order.
"""

from __future__ import annotations

from schemas.enums import ConceptTag, ConceptTagCategory


def _format_tag_block(tag: ConceptTag) -> str:
    """Render a single tag definition as a multi-line block.

    Format mirrors the pre-refactor prompt structure: one line for the
    definition, an optional long-form reasoning block when present
    (used by FEMALE_LEAD), then Check: and NOT: lines. The slug
    (`tag.value`) is used in the NOT label so the model sees the exact
    string token it should not emit.
    """
    lines = [f"- {tag.name}: {tag.description}"]
    if tag.long_form_instructions:
        # The long-form block is multi-paragraph; render it as its own
        # body so STEP 1/2/3 etc. read cleanly. A leading blank line
        # separates it visually from the one-line description.
        lines.append("")
        lines.append(tag.long_form_instructions)
        lines.append("")
    lines.append(f"  Check: {tag.selection_criteria}")
    lines.append(f"  NOT {tag.value}: {tag.boundary_cases}")
    return "\n".join(lines)


def _format_category_section(category: ConceptTagCategory) -> str:
    """Render one category's full section: header, cross-tag note,
    per-tag blocks, then section instructions.

    Cross-tag notes go at the TOP of the section (so the LLM reads
    inter-tag relationships before encountering the tags). Section
    instructions (only ENDINGS uses this) go at the BOTTOM so they
    serve as a "how to choose between these tags" wrap-up after the
    tag definitions have been read.
    """
    parts: list[str] = []

    # Section header. Intro_text is optional (CHARACTERS section has none).
    if category.intro_text:
        parts.append(f"{category.display_label} — {category.intro_text}")
    else:
        parts.append(category.display_label)
    parts.append("")

    if category.cross_tag_note:
        parts.append(category.cross_tag_note)
        parts.append("")

    # Tags belonging to this category, in master-enum declaration order.
    # ConceptTag's declaration order matches numeric concept_tag_id order
    # within each category, so iterating ConceptTag directly preserves
    # the intuitive section ordering for free.
    tags_in_category = [t for t in ConceptTag if t.category is category]
    for tag in tags_in_category:
        parts.append(_format_tag_block(tag))
        parts.append("")

    if category.section_instructions:
        parts.append(category.section_instructions)
        parts.append("")

    return "\n".join(parts)


def build_tag_definitions_section() -> str:
    """Assemble the `_TAG_DEFINITIONS` portion of the system prompt by
    iterating every `ConceptTagCategory` member in declaration order.

    Wraps the body with the same lead-in and trailing separator the
    pre-refactor hand-written block used so the surrounding `_TASK`,
    `_EVIDENCE`, `_INPUTS`, and `_OUTPUT` blocks compose into a
    well-formed prompt without modification.
    """
    lead_in = (
        "TAG DEFINITIONS\n\n"
        "Consider each category below. For each, check whether any tags "
        "apply based on the evidence rules and signals listed. Empty "
        "lists are correct when no tags apply.\n\n"
        "\n"
    )
    body = "\n\n".join(
        _format_category_section(cat) for cat in ConceptTagCategory
    )
    # Trailing separator + blank line matches the original prompt so the
    # `_OUTPUT` block concatenated downstream reads cleanly.
    return lead_in + body + "\n---\n\n"


def build_system_prompt() -> str:
    """Compose the full system prompt: hand-written framing + generated
    tag definitions + hand-written output spec.

    Imports the four hand-written constants from concept_tags.py at call
    time to avoid a circular import (concept_tags.py imports
    build_system_prompt from this module).
    """
    from movie_ingestion.metadata_generation.prompts.concept_tags import (
        _TASK, _EVIDENCE, _INPUTS, _OUTPUT,
    )
    return _TASK + _EVIDENCE + _INPUTS + build_tag_definitions_section() + _OUTPUT
