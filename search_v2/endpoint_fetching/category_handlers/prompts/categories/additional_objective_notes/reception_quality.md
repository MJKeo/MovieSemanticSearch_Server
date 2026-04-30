# Reception quality + superlative — additional notes

This category covers **how the movie was received and its stature as a piece of work**: cult / acclaimed / underrated / divisive / overhyped, cultural influence ("era-defining"), still-holds-up, thematic weight on the acclaim side ("has something to say"), cast popularity as acclaim ("stacked A-list cast"), "classic" as general canonical stature, and quality superlatives ("best horror of the 80s", "scariest ever", "funniest"). The category is about reception-quality judgment, not award records and not list membership.

## How the two endpoints split the work

The candidate set has two roles that usually compose rather than compete. Both firing is the common outcome; firing only one is valid when only one carries signal; the empty combination is valid when the atom routed here is really an award or named-list ask.

- **SEMANTIC (reception space)** carries the **qualitative-reception prose**. `reception_summary` holds a short capsule in the register reviews use ("cult reception", "underrated arthouse horror", "widely considered the high-water mark of the genre"). `praised_qualities` and `criticized_qualities` hold short labels for the specific axes people named (e.g. "atmospheric dread", "ensemble performances", "cultural influence"). This is the space that distinguishes "underrated" from "acclaimed" from "divisive" — framings a scalar cannot express.
- **METADATA (reception column)** anchors a **numeric prior** via the `reception` attribute (`well_received` or `poorly_received`). This is an **additive lift**, not a hard gate — an underrated arthouse film can sit below a popular crowd-pleaser on the numeric scale while the Semantic prose still carries the match. Use it when the framing implies the movie should be broadly well-regarded or broadly poorly-regarded; skip it when the framing is about reception *shape* (cult, divisive) rather than reception *direction*.

## Underrated is a case where prose matters more than score

"Underrated" specifically says the reception score is **lower than the work deserves**. The METADATA column cannot express that gap — a naive `well_received` pick contradicts what "underrated" means, and `poorly_received` is wrong too. The right read is: **Semantic carries this primarily** (reception_summary names the underrated shape and praised_qualities name the axes praised by the minority who noticed), and METADATA either does not fire or fires with careful framing (never as a hard-good floor). The category-shape axis on Cat 10 (popularity: niche) captures the "lesser-known" slice; this category handles the "quality is higher than the score would suggest" slice through the reception prose.

## Axis-of-superlative decomposes upstream

"Best horror of the 80s", "scariest movie ever made", "funniest comedy" are multi-atom queries. Step 2 splits them: the **axis** (horror, scary, funny) routes to whichever category carries it (Cat 11 genre, Cat 22 viewer-experience, etc.), the **era** (80s) routes to Cat 10 release_date, and the **superlative / acclaim framing** ("best", "scariest ever", "highly regarded") is what reaches you. Handle only your slice. Do not emit the axis or the era from this handler — those are other categories' work. Your output is the reception-quality half: Semantic reception prose about the film being widely considered top-tier in its slice, plus the METADATA reception lift.

## Boundaries with nearby categories

- **Awards (Cat 8).** Any framing anchored to formal recognition — "Oscar-winning", "BAFTA-nominated", "Palme d'Or winner", "won the Golden Globe" — is Cat 8's territory and does not belong here. The upstream split places those atoms on Cat 8 separately; if one routed here, the input is a misroute and **no-fire is the right response**. Do not translate "Oscar-winning" into praised_qualities ("awards") or into a `well_received` lift — the award endpoint handles structured recognition with per-ceremony precision the reception channels cannot match.
- **Curated canon / named list (Cat 28).** Membership in a specific named list — "Criterion Collection", "AFI Top 100", "Sight & Sound greatest films", "IMDb Top 250", "National Film Registry", "1001 Movies to See Before You Die" — is Cat 28. That category uses a different Semantic strategy (list-citation decoding) and its own reception framing. **"Classic" used as generic canonical stature** — "classic films", "a true classic", "timeless classics" — lives here: the user is asking about general stature, not a named list. The discriminator is whether the user named a specific list.
- **Popularity / niche (Cat 10).** "Obscure", "lesser-known", "hidden gems" as pure well-known-ness routes to METADATA popularity on Cat 10, not here. This category handles the **quality** judgment; popularity is a separate axis. "Underrated" sits on the boundary — it compounds popularity (niche) and reception (quality-above-score). Upstream splits it; handle only the reception-quality half when it reaches you.

## When to no-fire (both endpoints silent)

Return the empty combination when the atom routed here is not actually a reception-quality judgment:

- The framing is purely award-anchored (routes to Cat 8).
- The framing names a specific curated list (routes to Cat 28).
- The atom is just a popularity axis with no quality claim (routes to Cat 10 popularity).
- The phrasing is too vague to pin any reception-quality axis ("good movies", "quality films" with no further signal). Inventing praised_qualities or a numeric lift from nothing is worse than no-fire.

Record the misroute in `overall_endpoint_fits` and leave both endpoints at `should_run_endpoint: false`.

## The one principle

Fire exactly the subset the user's framing calls for. SEMANTIC when the requirement carries a reception **shape** (cult, acclaimed, underrated, divisive, classic, era-defining, stacked cast) the prose channel can express. METADATA when the framing implies a consistent direction on the numeric scale (broadly well-received, broadly panned) — and skip it when the shape is "lower than deserved" or "polarizing" that the scalar cannot honestly encode. Most reception-quality asks benefit from both; firing one alone is fine when only one applies; firing neither is correct when the atom belongs to Cat 8 or Cat 28.
