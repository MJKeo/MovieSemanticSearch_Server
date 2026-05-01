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
- `semantic_intent`: slow-building tension and restrained pacing feel.
- `augmentation_opportunities`: keyword not worth running unless a registry member directly names the experiential effect.
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
- `semantic_intent`: cathartic emotional impact; viewer goal plus felt response.
- `augmentation_opportunities`: keyword worth running for a direct tearjerker signal.
- `semantic_parameters`: `watch_context` for the viewer goal; `viewer_experience` for emotional palette; `reception` if framed as an audience label.
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
- `semantic_intent`: positive resolution and satisfying ending aftertaste.
- `augmentation_opportunities`: keyword worth running for a direct happy-ending tag.
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
- No-fire for this category.
- Reason: this is a concrete viewing situation, not an emotional or experiential property. Viewing occasion owns it.
</example>
