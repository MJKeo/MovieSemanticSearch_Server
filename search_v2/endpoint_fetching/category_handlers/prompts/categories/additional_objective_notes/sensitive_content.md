# Additional objective notes - Sensitive content

## Target

Fire for concrete content axes or content-intensity ceilings: gore, blood,
graphic violence, nudity, sexual content, strong language, drug use, animal
death, or rating-level intensity.

## Decision Questions

- What content axis or rating ceiling is explicit?
- Is the user requiring presence, excluding presence, or dialing intensity?
- Is the ask binary/hard, or gradient/soft?
- Which candidate endpoints add real signal? Address each one before
  committing.

## Endpoint Fit

- Keyword: binary registry content flags. Use only when a definition directly
  covers the axis. Do not use genre tags as fake content flags.
- Metadata: global rating ceiling only. Use for "family-friendly intensity" or
  "nothing above PG-13." Do not use it for a specific axis like gore.
- Semantic: viewer_experience.disturbance_profile for gradient intensity:
  "not too bloody", "violent but not graphic", "disturbing but not brutal."

## Hard vs Soft

- "No / without / zero" + clean binary flag -> hard exclusion through Keyword.
- "With / where it has" + clean binary flag -> hard inclusion through Keyword.
- "Not too / not overly / less graphic" -> semantic intensity gradient.
- "PG-13 or lower / family-friendly intensity" -> metadata ceiling.

Parameters describe the content or rating surface itself. Parent polarity
carries whether that surface helps or hurts.

## Boundaries

- Pure audience packaging ("for kids", "family movie") belongs to Target
  audience.
- Vague mood or weight ("nothing heavy", "something light") belongs to
  experiential tone unless a concrete content axis is named.
- Do not infer content axes from outside film knowledge or genre assumptions.

## No-Fire

Return no endpoint payloads when the target names no concrete content axis, no
rating ceiling, and no disturbance gradient grounded in the words provided.
