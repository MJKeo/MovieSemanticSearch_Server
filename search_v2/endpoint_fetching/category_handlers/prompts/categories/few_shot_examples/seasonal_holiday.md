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
- `semantic_walk`: aspects "Christmas viewing occasion" and "Christmas story setting"; candidates `watch_context` (holiday viewing) and `plot_events` (Christmas-set narrative).
- `keyword_walk`: surface a holiday-packaging registry member if one fits cleanly; flag generic family/romance candidates as too loose.
- `coverage_assignments`: semantic owns the holiday-viewing + Christmas-setting slice; add keyword only if a registry member directly covers holiday packaging. `intentionally_uncovered` empty.
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
- `semantic_walk`: aspect "Halloween viewing occasion / seasonal spooky fit"; candidate `watch_context` (central). Optional `plot_events` candidate only if Halloween-night setting is part of the captured meaning.
- `keyword_walk`: surface broad horror-family registry member as a proxy; narrower horror only if the call names that sub-form.
- `coverage_assignments`: semantic for the viewing-occasion slice; keyword for the broad horror-proxy slice. `intentionally_uncovered` empty.
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
- `semantic_walk`: aspect "Valentine's Day viewing / date-night holiday package"; candidate `watch_context` only.
- `keyword_walk`: surface generic-romance candidates honestly and mark them as too loose — they drop the holiday packaging.
- `coverage_assignments`: semantic owns the slice. Skip keyword unless a registry definition directly covers the seasonal package; otherwise leave it out (the walk's "too loose" finding is the audit). `intentionally_uncovered` empty.
- `semantic_parameters`: `watch_context`.
- `keyword_parameters`: null — a generic-romance commit would mix holiday-day signal with year-round romance.
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
- No-fire for this category — `coverage_assignments` empty; `intentionally_uncovered` names "Christmas as subject matter, not seasonal viewing" so reviewers see what was walked away from.
- Reason: the holiday is the subject matter, not seasonal viewing or holiday movie packaging. Central topic owns the retrieval.
</example>
