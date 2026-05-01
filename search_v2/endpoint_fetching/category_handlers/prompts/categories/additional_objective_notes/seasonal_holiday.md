# Seasonal / holiday - additional objective notes

## Target

Handle named season or holiday framing: a movie for watching at that time, a movie set during that time, or a compact seasonal package such as a holiday movie or summer blockbuster.

## Semantic Decision

Always author the semantic read first.

Ask:
- Is the user naming a viewing occasion? Use `watch_context` for seasonal viewing, holiday movie night, summer crowd-pleasing viewing, or family holiday watching.
- Is the user naming story setting? Use `plot_events` for stories set during the holiday or season.
- Are both present or normally fused by the phrase? Use both, but keep each surface specific.

Do not shrink semantic prose to match a keyword proxy. Semantic owns the seasonal meaning.

## Keyword Augmentation

Keyword is a proxy, not the authority. Fire it only when the registry has a clean shadow of the seasonal package.

Ask:
- Does a holiday, horror, family, romance, comedy, or spectacle member definition directly cover the seasonal package?
- Is the proxy the strongest single deterministic signal, not just a plausible neighbor?
- Would the proxy add crisp binary signal while semantic keeps the actual season or holiday?

Skip keyword when the season has no clean registry shadow, or when the proxy would turn the ask into a generic genre.

## Boundary Checks

- Pure date/place setting with no holiday packaging can belong to Narrative setting.
- A documentary or story about a holiday as a subject belongs to Central topic.
- "Family movie", "romance", "horror", or "action" without seasonal framing belongs elsewhere.
- "Snowed in", "at summer camp", or similar plot content belongs to Plot events unless framed seasonally.

## No-Fire

No-fire when the target is not seasonal/holiday, is too vague to name a season or holiday, or would require inventing a viewing scenario or proxy from outside the query.
