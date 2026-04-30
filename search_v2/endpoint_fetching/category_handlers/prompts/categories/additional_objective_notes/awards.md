# Award records — additional notes

This category is about **formal recognition records** — wins and nominations at tracked ceremonies, ceremony-specific filters, and multi-win superlatives. Your job is to translate the requirement into the five filter axes (ceremonies, award names, category tags, outcome, years) plus a scoring shape (mode + mark). The endpoint-context chunk above covers the axis mechanics and the five canonical scoring patterns; the notes here focus on the decisions small models most often get wrong.

## Outcome — winner vs. nominee

Pick based on the user's framing, not on surface vocabulary.

- "Oscar-**winning**", "won the Palme d'Or", "Best Picture **winner**" → `winner`.
- "Oscar-**nominated**", "Cannes **nomination**", "up for a BAFTA" → `nominee`.
- Bare ceremony references ("Oscars", "Cannes films", "recognized at Sundance") signal recognition reach rather than the win/nom split — leave `outcome` null so both count, unless the surrounding phrasing makes one side explicit.

A generic "award-winning" with no ceremony signals the winner side specifically — populate `outcome: winner`. A generic "award-nominated" flips to `nominee`.

## Scoring mode — FLOOR vs. THRESHOLD

Classify the requirement's intensity shape first, then commit:

- **FLOOR** — binary pass at `has_count >= scoring_mark`. Use for any specific-filter request ("won Best Picture at the Oscars", "BAFTA-nominated") and for any explicit count ("at least 3 wins", "multi-Oscar winner" = floor at 2, "won 11 Oscars" = floor at 11). When the user has locked down the filter axes or named a specific count, more wins past the mark do not make the match any "more correct" — the answer is yes or no.
- **THRESHOLD** — gradient `min(has_count, scoring_mark) / scoring_mark`. Use when the user wants "the more recognized, the better": generic "award-winning" (no ceremony / prize / category named), superlative framings ("most decorated", "the most Oscars"), qualitative-plenty framings ("heavily decorated", "loaded with awards", "swept the ceremony").

A useful check: if adding a tenth win should score the film higher than a film with two wins, you are in THRESHOLD territory. If the first qualifying win is all that matters, you are in FLOOR territory.

## Boundaries with nearby categories

- **Reception quality / superlative (Cat 25).** "Critically acclaimed", "cult classic", "underrated", "best of the 80s", "era-defining", "still holds up" are reception and stature framings — Cat 25's slice, not yours. No-fire when the only signal is a general quality judgment with no award vocabulary. Step 2 splits compound phrasings like "Oscar-winning cult classic" into a Cat 8 atom (Oscar-winning) and a Cat 25 atom (cult classic) — handle only the award slice routed to you; never try to encode "acclaimed" into scoring_mark or category_tags.
- **Curated canon / named list (Cat 28).** Criterion, AFI Top 100, Sight & Sound, National Film Registry are curated-list memberships, not award records. No-fire.
- **Metadata reception (Cat 10).** "Worst movies", "critically panned", "poorly received" route to metadata reception at the low pole — not to the Razzie side of this endpoint. Razzie data is only pulled when the user explicitly names Razzies or Golden Raspberries; never infer Razzie intent from general negative-quality language.

## Out-of-scope ceremonies

Only ceremonies in the tracked registry (listed above) can be emitted. If the user names a ceremony that is not in the registry — Emmy (TV), Tony (theatre), MTV Movie & TV Awards, Saturn Awards, Hugo, Goya, César, Cesar, People's Choice, Golden Horse, etc. — there is no posting record to hit. No-fire, and record the out-of-scope ceremony in `coverage_gaps`. Do NOT substitute a different tracked ceremony as a "close enough" proxy.

## Negated award phrasings

A negated award phrase is still a positive-direction filter; wrapper polarity handles the inversion. "Movies that won no major awards" describes the award concept directly (any major-ceremony win, FLOOR at 1) and pairs with `polarity: negative` on the wrapper — the executor subtracts the matches from the candidate set. Do NOT flip `outcome` to `nominee`, drop `scoring_mark` to zero, or otherwise invert axes inside `parameters` to simulate negation. The parameter payload always describes the target concept in positive form.

## When to no-fire

- **No award vocabulary at all.** The atom names only reception quality, stature, or canon membership. Upstream dispatch was wrong — record the mismatch in `coverage_gaps` and return `should_run_endpoint: false`.
- **Ceremony outside the tracked registry.** The specific ceremony named cannot be resolved. No-fire rather than substituting a different ceremony.
- **Ambiguous "recognition" with no concrete axis.** Phrases like "gets respect" or "people take seriously" without any ceremony / prize / outcome / count signal do not map onto this endpoint's structured axes. No-fire.

The tightest failure mode: firing on a reception or stature phrase that happens to sit near award language in the query. If you cannot point to explicit award vocabulary in the atomic rewrite or the parent fragment, no-fire.
