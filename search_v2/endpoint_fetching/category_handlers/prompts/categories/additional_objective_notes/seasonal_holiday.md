# Seasonal / holiday - additional objective notes

## Target

Handle named season or holiday framing: a movie for watching at that time, a movie set during that time, or a compact seasonal package such as a holiday movie or summer blockbuster.

## Semantic Decision

This category fires a single semantic call. Pick the vector spaces whose ingest-side text actually carries the seasonal evidence the user named.

Ask:
- Is the user naming a viewing occasion? Use `watch_context` for seasonal viewing, holiday movie night, summer crowd-pleasing viewing, or family holiday watching.
- Is the user naming story setting? Use `plot_events` for stories set during the holiday or season.
- Are both present or normally fused by the phrase? Use both, but keep each surface specific to what that space carries.

Semantic owns the seasonal meaning. Phrase the body so the season or holiday is named explicitly rather than abstracted into genre.

## Boundary Checks

- Pure date/place setting with no holiday packaging can belong to Narrative setting.
- A documentary or story about a holiday as a subject belongs to Central topic.
- "Family movie", "romance", "horror", or "action" without seasonal framing belongs elsewhere.
- "Snowed in", "at summer camp", or similar plot content belongs to Plot events unless framed seasonally.

## No-Fire

No-fire when the target is not seasonal/holiday, is too vague to name a season or holiday, or would require inventing a viewing scenario from outside the query.
