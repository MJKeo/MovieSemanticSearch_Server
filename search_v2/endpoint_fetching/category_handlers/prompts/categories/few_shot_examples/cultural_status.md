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
- `semantic_walk`: aspect "whole-work canonical stature / lasting reputation"; `reception` (strengths: canonical-stature framing lives in reception prose; weaknesses: under-coverage of scalar reception priors that semantic alone can't anchor numerically).
- `metadata_walk`: `reception` and `popularity` columns with strengths: scalar priors that anchor stature numerically; weaknesses: under-coverage of the canonical-status framing (scalars alone don't capture "classic"); `release_date` skipped — era is a sibling trait's concern.
- `coverage_exploration`: semantic carries the framing; metadata's scalars layer on as priors that reinforce. Both fire — semantic's experiential framing fills metadata's framing weakness; metadata's scalar anchor fills semantic's numerical weakness.
- `coverage_commitments`: `semantic.verdict=commit` (slice: canonical-stature framing) + `metadata.verdict=commit` (slice: scalar anchor).
- `semantic_parameters`: `reception` body capturing canonical-stature language.
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
- `semantic_walk`: aspect "quality-versus-recognition gap"; `reception` (strengths: the gap is reception-shape content the embedding can name; weaknesses: none for this slice — only semantic can frame a relationship between two scalars).
- `metadata_walk`: scalar reception/popularity columns with strengths: nominal anchors on either side of the gap; weaknesses: under-coverage (scalars alone can't express the high-quality + low-recognition relationship; either one in isolation is the wrong signal).
- `coverage_exploration`: semantic alone owns the slice. Metadata fails the fire test — its strengths are nominal anchors that don't compose into the gap relationship.
- `coverage_commitments`: `semantic.verdict=commit` (slice: underrated relationship) + `metadata.verdict=abstain` (reason: dominated-by-sibling — semantic owns the relationship, scalars alone don't compose).
- `semantic_parameters`: `reception` body framed around the underrated relationship.
- `metadata_parameters`: null.
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
- `semantic_walk`: aspect "canonical stature among newer films"; `reception` (strengths: stature framing).
- `metadata_walk`: `reception` and `popularity` columns with strengths: broad scalar priors; weaknesses: none for this slice. `release_date` NOT surfaced — the modern date window belongs to the sibling era trait.
- `coverage_exploration`: semantic carries the stature framing; metadata's scalars layer on as priors. Both fire.
- `coverage_commitments`: `semantic.verdict=commit` (slice: stature framing) + `metadata.verdict=commit` (slice: scalar anchor).
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
- `semantic_walk`: no candidates with substantive strengths for THIS category — aspect-level praise is owned by Specific praise / criticism or Visual craft acclaim.
- `metadata_walk`: no candidates with substantive strengths.
- `coverage_exploration`: every walk surfaced no useful candidate; this category abstains. Routing handles the retrieval via a sibling category.
- `coverage_commitments`: every endpoint `verdict=abstain` (reason: no-walk-candidate — the call's intent doesn't land in this category at all).
- All `*_parameters`: null.
</example>
