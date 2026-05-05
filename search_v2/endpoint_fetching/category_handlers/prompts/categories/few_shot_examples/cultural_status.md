# Cultural status / canonical stature - few-shot examples

<example>
## Input
```xml
<raw_query>classic movies</raw_query>
<target_entry>
  <captured_meaning>Movies with classic canonical stature.</captured_meaning>
  <category_name>Cultural status / canonical stature</category_name>
  <atomic_rewrite>Classic status.</atomic_rewrite>
</target_entry>
```

## Expected Decision
- `semantic_walk`: aspect "whole-work canonical stature / lasting reputation"; candidate `reception` (central) — canonical stature lives in reception prose / praised qualities, not in scalar columns alone.
- `metadata_walk`: surface `reception` and `popularity` columns as priors that reinforce stature; flag the gap that scalars alone don't capture canonical-status framing.
- `coverage_assignments`: semantic for the stature framing; metadata for reception/popularity priors. Overlap is the design — both pull on "classic". `intentionally_uncovered` empty.
- `semantic_parameters`: `reception` body capturing canonical-stature language.
- `metadata_parameters`: reception and/or popularity prior; no release-date logic here (era is a sibling trait's concern).
</example>

<example>
## Input
```xml
<raw_query>underrated thrillers</raw_query>
<target_entry>
  <captured_meaning>Thrillers considered better than their recognition suggests.</captured_meaning>
  <category_name>Cultural status / canonical stature</category_name>
  <atomic_rewrite>Underrated reception-status shape.</atomic_rewrite>
</target_entry>
```

## Expected Decision
- `semantic_walk`: aspect "quality-versus-recognition gap"; candidate `reception` (central) — the gap is reception-shape content, only semantic can name it.
- `metadata_walk`: walk the scalar reception/popularity columns honestly; coverage prose names them as unable to express the gap (high-quality + low-recognition is a relationship, not a level).
- `coverage_assignments`: semantic owns the slice; metadata skipped because the walk surfaced no clean fit. `intentionally_uncovered` may name "scalar quality-vs-recognition gap" if the slice can't be carried any other way.
- `semantic_parameters`: `reception` body framed around the underrated relationship.
- `metadata_parameters`: null unless the input separately gives a clean scalar prior.
</example>

<example>
## Input
```xml
<raw_query>modern classics</raw_query>
<target_entry>
  <captured_meaning>Movies with modern-classic status.</captured_meaning>
  <category_name>Cultural status / canonical stature</category_name>
  <atomic_rewrite>Classic status, with modern handled by a sibling era trait.</atomic_rewrite>
</target_entry>
```

## Expected Decision
- `semantic_walk`: aspect "canonical stature among newer films"; candidate `reception` (central).
- `metadata_walk`: surface `reception` and `popularity` columns as broad priors; do NOT surface `release_date` here — the modern date window belongs to the sibling era trait, not this category.
- `coverage_assignments`: semantic for the stature; metadata for reception/popularity priors. `intentionally_uncovered` empty (release-date is owned by a sibling trait, not this category).
- `semantic_parameters`: `reception`.
- `metadata_parameters`: reception/popularity only; do not encode the modern date window here.
</example>

<example>
## Input
```xml
<raw_query>praised for its cinematography</raw_query>
<target_entry>
  <captured_meaning>Aspect-level praise for cinematography.</captured_meaning>
  <category_name>Cultural status / canonical stature</category_name>
  <atomic_rewrite>Visual craft praise.</atomic_rewrite>
</target_entry>
```

## Expected Decision
- No-fire for this category — `coverage_assignments` empty; `intentionally_uncovered` names "specific aspect praise for cinematography, owned by Specific praise / criticism or Visual craft acclaim".
- Reason: this is specific aspect praise, not whole-work cultural status. Specific praise / criticism or Visual craft acclaim owns it.
</example>
