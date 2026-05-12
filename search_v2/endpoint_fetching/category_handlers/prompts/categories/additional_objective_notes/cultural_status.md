# Cultural status / canonical stature - additional objective notes

## Target

Handle whole-work cultural position or reputation shape: classic, cult classic, underrated, overhyped, divisive, still holds up, influential, iconic, landmark, era-defining, culturally significant, ahead of its time.

## Semantic Decision

Always author the semantic read first. Use `reception` for the movie's broad reputation, canon position, durability, influence, devoted following, critic/audience split, or quality-versus-recognition gap.

Ask:
- What reputation shape is the user actually asking for?
- Is it whole-work status, not one praised/criticized component?
- Does the phrase describe canon/influence/reception shape rather than a simple quality prior?

## Metadata Augmentation

Metadata is a supporting prior only. Fire it when scalar reception or popularity genuinely reinforces the status term. Two distinct shapes:

**Single-concept stature** — `classic`, `iconic`, `landmark`, `culturally significant`, `widely canonical`. One underlying idea (canonical stature) with multiple evidencing axes. Fire `reception=well_received` and/or `popularity=popular` as **substitutable** evidence → `scoring_method=ANY`. Either scalar alone is some evidence of canonical stature.

**Compound stature** — `cult classic`, `cult favorite`, `beloved hidden gem`. Two reinforcing axes that BOTH must be present: niche reach + positive reception. Decompose:
- `cult classic` → `popularity=niche` AND `reception=well_received` → `scoring_method=ALL`. Both are required — niche alone isn't a cult classic (could be an obscure flop), and well-received alone isn't a cult classic (could be a mainstream prestige film). The "cult" half maps to popularity, the "classic" half maps to reception; they reinforce rather than substitute.

**Gap / polarization shapes** — `underrated`, `overhyped`, `divisive`, `ahead of its time`. The intent is a *relationship* between axes (quality > recognition, hype > quality, critic ≠ audience), not the axes themselves. Scalar priors cannot express that relationship even composed with ALL, so metadata abstains and the semantic reception body carries the slice.

Ask:
- Does the status imply broad quality, broad popularity, both, or a gap between them?
- If both, are they substitutable evidence of one concept (ANY) or reinforcing facets of a compound (ALL)?
- Would a scalar prior help without changing the semantic status shape?

## Boundary Checks

- Well-received, popular, great, or highly rated without status shape belongs to General appeal.
- Praised for tension, criticized for pacing, or loved for performances belongs to Specific praise / criticism.
- Formal wins and nominations belong to Awards.
- "Classic" alone is not an era. Explicit old/modern wording is a separate Release date trait.

## No-Fire

No-fire when the target is aspect-level praise, simple numeric/general quality, awards, era only, or a named curated list with no resolved status trait.
