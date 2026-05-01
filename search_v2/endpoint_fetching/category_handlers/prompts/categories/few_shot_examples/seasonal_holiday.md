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
- `semantic_intent`: Christmas seasonal package; watch-context holiday viewing plus plot-events Christmas setting.
- `augmentation_opportunities`: keyword worth running if the chosen registry member directly covers holiday movie packaging.
- `semantic_parameters`: include `watch_context`; include `plot_events` if the phrase is read as Christmas-set as well as Christmas-viewing.
- `keyword_parameters`: one strongest holiday proxy, not a generic family/romance guess.
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
- `semantic_intent`: Halloween viewing occasion and seasonal spooky fit.
- `augmentation_opportunities`: keyword worth running for the broad horror proxy; narrower horror only if the input names that sub-form.
- `semantic_parameters`: central `watch_context`; optional `plot_events` only if Halloween-night story setting is part of the captured meaning.
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
- `semantic_intent`: Valentine's Day viewing or date-night holiday package.
- `augmentation_opportunities`: keyword not worth running if the best member only says generic romance and drops the holiday package.
- `semantic_parameters`: `watch_context`.
- `keyword_parameters`: null unless a registry definition directly covers the seasonal package.
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
- No-fire for this category.
- Reason: the holiday is the subject matter, not seasonal viewing or holiday movie packaging. Central topic owns the retrieval.
</example>
