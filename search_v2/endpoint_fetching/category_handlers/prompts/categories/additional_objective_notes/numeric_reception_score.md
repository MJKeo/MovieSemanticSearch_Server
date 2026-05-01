# Additional objective notes - Numeric reception score

## Target

Fire only for explicit numeric reception thresholds or rating-scale values.

## Decision Questions

- Does the phrase contain a concrete number, percent, star value, or score
  threshold?
- Is the user asking for a hard numeric cutoff rather than a broad quality
  prior?
- Can the endpoint represent the number without guessing the source scale?

## Boundaries

- "Rated above 8" and "70%+" fire here.
- "Highly rated", "well-reviewed", "best", and "great" are General appeal.
- Awards, canon, cult, underrated, and divisive status are not numeric
  thresholds.
- No-fire rather than converting qualitative praise into a made-up number.

