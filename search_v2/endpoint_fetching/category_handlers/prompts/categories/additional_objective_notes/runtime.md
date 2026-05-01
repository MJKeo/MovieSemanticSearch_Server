# Additional objective notes - Runtime

## Target

Fire only for explicit movie length: duration, runtime ceiling/floor, or
short/long length preference.

## Decision Questions

- What concrete duration surface is targeted?
- Is the phrase about the movie's length, not franchise size or viewing
  occasion?
- Can the endpoint express the implied minutes/range without inventing one?

## Boundaries

- The default feature-film floor is system behavior, not a Runtime firing.
- "Short films" may carry a runtime slice; media type is handled elsewhere.
- "Long-running franchise" is franchise lineage, not runtime.
- No-fire when no duration language is present.

