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
- `semantic_walk`: aspect "slow-building tension / restrained pacing feel"; `viewer_experience` (strengths: pacing/tension live there directly; weaknesses: none for this slice).
- `keyword_walk`: broad thriller / suspense members carry strengths: genre proxy; weaknesses: under-coverage (no member names the slow-burn pacing FEEL) AND over-coverage (year-round thrillers pulled regardless of pacing). Drop them per the local test — dominated by semantic on the same content with no distinct strength to add.
- `coverage_exploration`: semantic alone owns the slice. Keyword candidates fail the fire test (under-coverage on the pacing aspect, over-coverage on the genre prior; no distinct strength semantic doesn't already carry).
- `coverage_assignments`: semantic only.
- `semantic_parameters`: `viewer_experience` only.
- `keyword_parameters`: null.
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
- `semantic_walk`: aspects "viewer goal of crying", "cathartic emotional palette", "audience-label tearjerker reception"; `watch_context` (strengths: viewer-goal framing), `viewer_experience` (strengths: cathartic emotional palette), `reception` (strengths: audience-label framing if framed that way).
- `keyword_walk`: tearjerker registry member with strengths: direct effect tag; weaknesses: none — the registry has a member that names exactly this effect.
- `coverage_exploration`: semantic carries the graded experiential signal across multiple aspects; keyword adds gate-style sharpness via the tearjerker tag. Both fire — distinct contributions on the same slice.
- `coverage_assignments`: semantic + keyword.
- `semantic_parameters`: `watch_context` for the viewer goal; `viewer_experience` for emotional palette; `reception` if framed as audience label.
- `keyword_parameters`: tearjerker tag.
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
- `semantic_walk`: aspect "positive resolution / satisfying ending aftertaste"; `viewer_experience` targeting `ending_aftertaste` (strengths: ending-aftertaste vocabulary directly; weaknesses: none for this slice).
- `keyword_walk`: happy-ending registry member with strengths: direct ending-shape tag; weaknesses: none — the registry has a member that names exactly this shape.
- `coverage_exploration`: semantic carries the experiential aftertaste; keyword adds gate-style sharpness via the happy-ending tag. Both fire.
- `coverage_assignments`: semantic + keyword.
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
- `semantic_walk`: no candidates with substantive strengths for THIS category — date-night is a viewing occasion, owned by the Viewing occasion category.
- `keyword_walk`: no candidates with substantive strengths.
- `coverage_exploration`: every walk surfaced no useful candidate; this category abstains. Routing handles the actual retrieval via the Viewing occasion call.
- `coverage_assignments`: empty.
- All `*_parameters`: null.
</example>
