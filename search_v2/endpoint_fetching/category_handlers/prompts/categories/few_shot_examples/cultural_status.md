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
- `semantic_intent`: whole-work canonical stature and lasting reputation.
- `augmentation_opportunities`: metadata worth running if reception/popularity priors reinforce the status.
- `semantic_parameters`: `reception`.
- `metadata_parameters`: reception and/or popularity prior; no release-date logic here.
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
- `semantic_intent`: quality-versus-recognition gap.
- `augmentation_opportunities`: metadata usually not worth running; scalar reception/popularity cannot express the gap.
- `semantic_parameters`: `reception`.
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
- `semantic_intent`: canonical stature among newer films.
- `augmentation_opportunities`: metadata worth running for broad reception/popularity if clean.
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
- No-fire for this category.
- Reason: this is specific aspect praise, not whole-work cultural status. Specific praise / criticism or Visual craft acclaim owns it.
</example>
