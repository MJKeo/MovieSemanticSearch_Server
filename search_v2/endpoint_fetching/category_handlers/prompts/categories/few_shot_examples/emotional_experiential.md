# Emotional / experiential - few-shot examples

<example>
## Input
```xml
<raw_query>a slow-burn thriller</raw_query>
<target_entry>
  <captured_meaning>Slow-burn pacing as the viewing experience.</captured_meaning>
  <category_name>Emotional / experiential</category_name>
  <atomic_rewrite>A thriller whose tension builds slowly.</atomic_rewrite>
</target_entry>
```

## Expected Decision
- `semantic_walk`: aspect "slow-building tension / restrained pacing feel"; candidate `viewer_experience` (central) — pacing/tension live there, not in keyword.
- `keyword_walk`: walk the registry honestly; broad thriller / suspense members surface but cover genre, not the experiential feel — gap noted.
- `coverage_assignments`: semantic owns the slice; keyword skipped because no registry member directly names the experiential effect. `intentionally_uncovered` empty (semantic covers what matters; the registry simply has no slow-burn-pacing tag).
- `semantic_parameters`: `viewer_experience` only.
- `keyword_parameters`: null; do not add generic thriller or suspense tags.
</example>

<example>
## Input
```xml
<raw_query>make me cry</raw_query>
<target_entry>
  <captured_meaning>A movie designed to make the viewer cry.</captured_meaning>
  <category_name>Emotional / experiential</category_name>
  <atomic_rewrite>Tearjerker emotional effect.</atomic_rewrite>
</target_entry>
```

## Expected Decision
- `semantic_walk`: aspects "viewer goal of crying" (`watch_context`), "cathartic emotional palette" (`viewer_experience`), and "audience-label tearjerker reception" (`reception` — only if framed as audience-label).
- `keyword_walk`: surface a direct tearjerker registry member; coverage prose names it as a clean fit for the effect.
- `coverage_assignments`: semantic for the goal + felt response; keyword for the direct tearjerker tag. Both slices are distinct facets of one ask. `intentionally_uncovered` empty.
- `semantic_parameters`: `watch_context` for the viewer goal; `viewer_experience` for emotional palette; `reception` if framed as audience label.
- `keyword_parameters`: tearjerker-style tag if the registry definition matches the effect.
</example>

<example>
## Input
```xml
<raw_query>movies with a happy ending</raw_query>
<target_entry>
  <captured_meaning>Movies whose ending resolves happily.</captured_meaning>
  <category_name>Emotional / experiential</category_name>
  <atomic_rewrite>Happy ending.</atomic_rewrite>
</target_entry>
```

## Expected Decision
- `semantic_walk`: aspect "positive resolution / satisfying ending aftertaste"; candidate `viewer_experience` targeting `ending_aftertaste`.
- `keyword_walk`: surface a direct happy-ending registry member if one fits cleanly.
- `coverage_assignments`: semantic for the experiential aftertaste; keyword for the direct happy-ending tag. `intentionally_uncovered` empty.
- `semantic_parameters`: `viewer_experience.ending_aftertaste`.
- `keyword_parameters`: happy-ending tag.
</example>

<example>
## Input
```xml
<raw_query>date night movie</raw_query>
<target_entry>
  <captured_meaning>A movie for a date-night situation.</captured_meaning>
  <category_name>Emotional / experiential</category_name>
  <atomic_rewrite>Date-night viewing occasion.</atomic_rewrite>
</target_entry>
```

## Expected Decision
- No-fire for this category — `coverage_assignments` empty; `intentionally_uncovered` names "date-night viewing occasion, owned by Viewing occasion".
- Reason: this is a concrete viewing situation, not an emotional or experiential property. Viewing occasion owns it.
</example>
