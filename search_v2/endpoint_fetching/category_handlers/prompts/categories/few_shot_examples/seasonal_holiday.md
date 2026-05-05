# Seasonal / holiday - few-shot examples

<example>
## Input
```xml
<raw_query>Christmas movies</raw_query>
<target_entry>
  <captured_meaning>Movies framed as Christmas movies.</captured_meaning>
  <category_name>Seasonal / holiday</category_name>
  <atomic_rewrite>Christmas viewing and Christmas-set stories.</atomic_rewrite>
</target_entry>
```

## Expected Decision
- `semantic_walk`: aspects "Christmas viewing occasion" and "Christmas story setting"; `watch_context` (strengths: holiday viewing occasion; weaknesses: under-coverage of the on-screen Christmas setting itself, which lives in plot_events) and `plot_events` (strengths: Christmas-set narrative; weaknesses: under-coverage of the viewing-occasion framing).
- `keyword_walk`: holiday-packaging registry member if one fits with clean strengths; generic family/romance candidates carry over-coverage weaknesses (year-round romance / family pulled alongside the holiday slice).
- `coverage_exploration`: semantic owns the experiential / setting slice; keyword fires only when its walk surfaces a holiday-packaging member with clean strengths — its sharpness layers on top of semantic. Skip keyword when only over-broad romance/family candidates surface.
- `coverage_assignments`: semantic always; keyword when the registry has a clean holiday-packaging member.
- `semantic_parameters`: `watch_context` always; `plot_events` if the phrase reads as Christmas-set as well as Christmas-viewing.
- `keyword_parameters`: one strongest holiday-packaging proxy, or null when only generic-romance/family adjacencies exist.
</example>

<example>
## Input
```xml
<raw_query>something for Halloween viewing</raw_query>
<target_entry>
  <captured_meaning>A movie for Halloween viewing.</captured_meaning>
  <category_name>Seasonal / holiday</category_name>
  <atomic_rewrite>Halloween movie-night fit.</atomic_rewrite>
</target_entry>
```

## Expected Decision
- `semantic_walk`: aspect "Halloween viewing occasion / seasonal spooky fit"; `watch_context` (strengths: viewing-occasion fit; weaknesses: none for this slice). Optional `plot_events` only if Halloween-night setting is part of the meaning.
- `keyword_walk`: broad horror-family member with strengths: spooky proxy fit; weaknesses: over-coverage (pulls non-Halloween horror year-round). Narrower horror only if the call names a sub-form.
- `coverage_exploration`: semantic owns the viewing-occasion slice; keyword's broad horror layers on as a sharpness signal — its over-coverage is acceptable because semantic isolates the Halloween framing.
- `coverage_assignments`: semantic + keyword.
- `semantic_parameters`: central `watch_context`; optional `plot_events` only when Halloween-night story setting is part of the meaning.
- `keyword_parameters`: crisp horror-family proxy.
</example>

<example>
## Input
```xml
<raw_query>Valentine's Day movies</raw_query>
<target_entry>
  <captured_meaning>Movies for Valentine's Day viewing.</captured_meaning>
  <category_name>Seasonal / holiday</category_name>
  <atomic_rewrite>Valentine's Day viewing package.</atomic_rewrite>
</target_entry>
```

## Expected Decision
- `semantic_walk`: aspect "Valentine's Day viewing / date-night holiday package"; `watch_context` (strengths: date-night package fit; weaknesses: none for this slice).
- `keyword_walk`: generic-romance members carry strengths: romance proxy; weaknesses: over-coverage (year-round romance pulled alongside the Valentine's slice) AND under-coverage (no Valentine's-specific signal). Drop the candidate per the local test — its strengths are dominated by what semantic already carries, and its over-coverage isn't refined by anything.
- `coverage_exploration`: semantic alone covers the slice. Keyword's only candidate fails the fire test (no distinct strength, dominated by semantic on the same content).
- `coverage_assignments`: semantic only.
- `semantic_parameters`: `watch_context`.
- `keyword_parameters`: null.
</example>

<example>
## Input
```xml
<raw_query>documentaries about Christmas traditions</raw_query>
<target_entry>
  <captured_meaning>Documentaries about Christmas traditions as a subject.</captured_meaning>
  <category_name>Seasonal / holiday</category_name>
  <atomic_rewrite>Christmas traditions as documentary subject matter.</atomic_rewrite>
</target_entry>
```

## Expected Decision
- `semantic_walk`: no candidates with substantive strengths — Christmas-as-subject is owned by Central topic, not this category's spaces.
- `keyword_walk`: no candidates with substantive strengths — same reason.
- `coverage_exploration`: every walk surfaced no useful candidate; this category abstains. Routing handles the actual retrieval via the Central topic call.
- `coverage_assignments`: empty.
- All `*_parameters`: null.
</example>
