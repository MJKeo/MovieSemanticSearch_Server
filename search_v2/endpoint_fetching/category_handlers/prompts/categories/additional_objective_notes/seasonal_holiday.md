# Seasonal / holiday — additional notes

This category owns seasonal and holiday framings — Christmas, Halloween, Thanksgiving, Fourth of July, Valentine's Day, summer-blockbuster. Covers both the "movie for watching AT this season" angle and the "movie SET at this season" angle. The vocabulary carries **no dedicated seasonal tags**, which is why two endpoints are needed and why KEYWORD works through proxy rewriting rather than a direct label match.

## How the two endpoints split the work

- **KEYWORD** has no seasonal registry member. To fire, it must rewrite the seasonal intent into the **single strongest proxy tag** from the existing vocabulary — a genre or audience member whose definition covers the kind of movie the season is shorthand for. One member per call: KEYWORD emits the best proxy, SEMANTIC carries the rest. If no proxy is a clean fit (the season has no canonical genre shadow in the vocabulary), KEYWORD does not fire.
- **SEMANTIC** contributes two sub-spaces. `watch_context.watch_scenarios` for the viewing-occasion angle ("Christmas viewing", "Halloween movie night", "date-night Valentine's pick"). `plot_events.plot_summary` for the narrative-setting angle ("set on Christmas Eve", "takes place during Thanksgiving dinner", "summer camp slasher"). Populate both when the ask supports both angles — one when it only supports one.

## The proxy-chain mechanic for KEYWORD

The proxy rewrite lets KEYWORD contribute signal despite the missing vocabulary, but every proxy is approximate. Treat the proxy as "the registry member whose definition most directly covers the kind of movie the user means when they invoke this season" — not the literal season label.

Canonical anchors:

- **Halloween** → `HORROR` is the broad default. Narrow only if the query cites the sub-form's defining premise: `SLASHER_HORROR` for a stalker/killer framing, `SUPERNATURAL_HORROR` for ghosts/demons/possession, `MONSTER_HORROR` for monster-centric, and so on. Bare "Halloween movies" takes the broad `HORROR`.
- **Christmas** → `HOLIDAY_FAMILY` when the registry's seasonal-family member cleanly covers it; `FAMILY` as the broader fallback when the framing leans general-family rather than specifically festive.
- **Summer blockbuster** → `ACTION_EPIC` / `ACTION` / `ADVENTURE_EPIC` / the closest large-spectacle member whose definition names sweeping scope and scale.
- **Valentine's Day** → no clean proxy. The romance registry members (romantic comedy, feel-good romance, romantic epic) are closer to genre asks than to "the Valentine's Day ritual". When no registry member's definition genuinely names the seasonal packaging, KEYWORD does not fire and SEMANTIC carries the signal alone.

The proxy is a ranked pick, not a formula. Compare the definitions of the candidate members against the seasonal framing and pick the one that best covers it on its own merits. If the strongest candidate's definition only tangentially touches the seasonal ask, prefer no-fire over forcing a proxy whose definition does not really support it.

## Semantic routing when SEMANTIC fires

- **`watch_context.watch_scenarios`** holds occasion terms directly — "Christmas viewing", "Halloween movie night", "Fourth of July watch party", "Valentine's Day date". Populate `self_experience_motivations` and `external_motivations` when the ask carries a viewer-pull or social framing ("cozy holiday feeling", "watching with the family over the holidays").
- **`plot_events.plot_summary`** holds compact prose naming the seasonal setting of the story itself — "a story set on Christmas Eve in a small town"; "events unfold over a Halloween night at an isolated camp". Use it only when the user frames the season as narrative setting, not when they only frame it as viewing occasion.
- Pick the **primary_vector** between the two populated spaces by which angle the user's phrasing leans into more. A pure viewing-occasion phrase ("Halloween viewing") leans watch_context; a pure setting phrase ("set on Christmas Eve") leans plot_events; a compound phrase that carries both ("a good Christmas movie to watch at Christmas") usually takes watch_context as primary because the viewing-occasion framing is the higher-signal dimension for seasonal asks.

## Boundaries with nearby categories

- **Top-level genre (Cat 11).** The proxy chain routes through genre registry members, but the user's intent is seasonal, not a genre ask. Step 2 decides the category. If the atom arrived here, treat it as seasonal — pick the proxy as a rewrite of the seasonal framing and let SEMANTIC carry the seasonal-specific signal the genre tag misses. Do NOT emit a semantic plot_analysis body about the genre; the watch_context + plot_events split is what this category owns.
- **Plot events / narrative setting (Cat 20).** "Movies set on Christmas Eve" phrased as pure plot-setting with no holiday framing may have been routed to Cat 20, not here. When this category owns the atom, `plot_events` fires as the seasonal-narrative angle — but the request is still framed seasonally ("a Christmas movie that takes place on Christmas Eve"), not as bare plot content. If the atom routed here really is pure plot-setting with no seasonal viewing or packaging framing, no-fire on both endpoints and record the misroute.
- **Occasion / comfort-watch (Cat 23).** Generic viewing-occasion framings without a named season ("a Sunday comfort watch", "rainy-day viewing") belong to Cat 23. Seasonal asks are a specific sub-form — named holiday or season plus the viewing or setting angle. A bare seasonal word with no accompanying framing ("summer movies" with no blockbuster / vacation / setting cue) may be ambiguous; prefer no-fire over forcing a broad proxy.

## When to no-fire

Emit the empty combination (both endpoints `should_run_endpoint: false`) when:

- The atom routed here is not actually seasonal — "family movies" without a holiday framing, "romantic movies" without a Valentine's cue, "action movies" without a summer cue. Cat 21 / Cat 11 / other categories own those.
- The seasonal word is present but the user's actual ask is a different category's territory (e.g. "a Halloween-themed party playlist" is not a movie ask).
- The phrasing is too thin to author either a proxy pick or a watch_scenarios / plot_events body without fabrication ("seasonal stuff", "movies about holidays" with no specific holiday).

Forcing a proxy the definition does not really support, or padding a watch_scenarios body from a bare seasonal noun with no viewing or setting framing, is worse than no-fire.
