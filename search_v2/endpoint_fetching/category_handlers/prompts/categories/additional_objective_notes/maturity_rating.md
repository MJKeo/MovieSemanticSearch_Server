# Additional objective notes - Maturity rating

## Target

Fire for explicit MPAA-style rating predicates: exact rating, ceiling, floor,
or exclusion of a named rating.

## Decision Questions

- Did the user name a rating or rating bound?
- Is the rating the retrieval surface, rather than an audience or content
  sensitivity proxy?
- Is negation carried by the wrapper, not by inventing inverse parameters?

## Boundaries

- "PG-13 family movie" splits: rating here; audience fit elsewhere.
- "No R-rated" still targets the R rating; polarity handles direction.
- "Family-friendly", "for kids", "no gore", and "not graphic" do not fire
  here unless a literal rating slice is present.

