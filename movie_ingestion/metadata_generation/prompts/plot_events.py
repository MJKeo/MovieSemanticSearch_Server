"""
System prompt for Plot Events generation.

Two branch-specific prompts, selected by build_plot_events_prompts()
based on whether the movie has a quality synopsis (>= 2500 chars):

- SYSTEM_PROMPT_SYNOPSIS: Condensation task. The synopsis is already a
  comprehensive plot recount -- the LLM abbreviates it into a shorter
  summary. Inputs: title, overview, plot_synopsis.

- SYSTEM_PROMPT_SYNTHESIS: Consolidation task. Multiple partial sources
  (summaries, overview) are consolidated into a single organized
  account. Framed as text consolidation (not narrative creation) to
  prevent hallucination when input is sparse. No parametric knowledge
  permitted. Inputs: title, overview, plot_summaries.

Both branch prompts include strict no-hallucination rules. The synopsis
branch constrains the LLM to the provided synopsis; the synthesis branch
frames the task as text consolidation so the LLM has no reason to
generate content beyond what the input provides.

Output is a single plot_summary field. Setting and character fields were
removed after evaluation showed setting is redundant (already captured
in the summary) and structured character extraction adds analytical
burden better handled by the downstream plot_analysis generator.
Character names appear naturally in the plot_summary text.

Plot keywords are deliberately excluded from input -- evaluation showed
they act as hallucination springboards and get incorrectly treated as
plot events.

The plot_summary output is the most critical field in the entire pipeline --
it feeds 4 of 5 Wave 2 generations as plot_synopsis.

"""

# ---------------------------------------------------------------------------
# Branch-specific prompts (ADR-033)
# ---------------------------------------------------------------------------

# Synopsis branch: the movie has a detailed synopsis (>= 2500 chars).
# The LLM's job is to condense it into a shorter but still rich summary,
# not to synthesize from multiple sparse sources.
SYSTEM_PROMPT_SYNOPSIS = """\
Condense a detailed plot synopsis into a shorter, high-signal, spoiler-containing summary of WHAT HAPPENS in a movie.

TASK: The synopsis below is already a comprehensive plot recount. Abbreviate it into a condensed plot_summary that preserves major plot beats, character arc transitions, thematic turning points, and the resolution. Drop scene-level description, dialogue specifics, minor subplots, and transitional events.

GOAL: Preserve character NAMES, locations, concrete actions. Focus on EVENTS/FACTS, not themes. Only essential characters, 1-3 core conflicts.

INPUTS (some may be empty)
- title: "Title (Year)"
- overview: marketing premise (use for context, not as primary source)
- plot_synopsis: full plot recount (PRIMARY source -- condense this)

RULES
- The plot_synopsis is your primary source. Use overview only for supplementary context.
- Only describe what is evident from the provided data. Do not supplement with your own knowledge of this film. If data is limited, produce a shorter summary rather than inventing details.
- Concrete wording only. No moralizing or theme talk.
- Keep plot_summary under 4,000 tokens. Length should be proportional to story complexity -- shorter input warrants a shorter summary.

OUTPUT
JSON with a single field: plot_summary.
plot_summary: Condensed chronological start-to-end summary. Preserve major plot beats, character names, locations, key events. Drop scene-level detail and dialogue. Use compact wording over flowery prose."""


# Synthesis branch: the movie has no quality synopsis. The LLM
# consolidates multiple partial sources (user-written summaries,
# overview) into a single organized plot account. Framed as text
# consolidation (not narrative creation) to prevent hallucination
# when input is sparse.
SYSTEM_PROMPT_SYNTHESIS = """\
Extract a high-signal, spoiler-containing account of WHAT HAPPENS in a movie, working EXCLUSIVELY from the provided input data.

CONSTRAINT: You are a text consolidation tool operating on the data below. You have no knowledge of any film. Before including any name, event, location, or detail, internally verify it appears in the input. If it does not, omit it. Do not cite sources or explain where details came from in your output -- just write the consolidated account directly.

TASK: The input contains multiple partial sources describing the same movie. Consolidate them into a single organized account, removing redundancy and resolving contradictions by preferring the most detailed, internally consistent version.

GOAL: Preserve character NAMES, locations, concrete actions. Focus on EVENTS/FACTS, not themes. Only essential characters, 1-3 core conflicts.

INPUTS (some may be empty)
- title: "Title (Year)"
- overview: marketing premise (often vague -- do not treat as plot detail)
- plot_summaries: user-written event summaries

RULES
- If the input names a character, use that name. If it describes a character without naming them, use the descriptive reference (e.g. "his fiancee"). Never supply a name the input does not provide.
- If the input describes an event, include it. If it does not, do not infer what "probably" happens.
- On conflict between sources: prefer most detailed, internally consistent version.
- Concrete wording only. No moralizing or theme talk.
- Output length is naturally proportional to input volume. A one-sentence overview produces a short paragraph. Three detailed summaries produce a rich account. Do not pad.

OUTPUT
JSON with a single field: plot_summary.
plot_summary: Chronological account of what happens, built entirely from the input text. When the input is thin, this will be short -- that is correct behavior."""
