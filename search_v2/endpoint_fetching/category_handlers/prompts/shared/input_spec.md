# Input spec

On each call, you receive a payload describing one committed retrieval call to translate into endpoint parameters. The payload is serialized as XML in the user message. The committed call is the source of truth — the upstream stage has already classified the category, decided polarity and match-mode, and folded in any modifier or qualifier context. Your job is to obey the committed brief and produce faithful endpoint parameters; do not re-interpret intent or re-derive routing.

## `<retrieval_intent>`

A 1–3 sentence brief authored by the upstream stage specifically for you. Names the dimension(s) being searched, the shape of the search (named-entity match, continuous experiential axis, archetype/iconography, structural attribute, setting/temporal window, etc.), and what the call is trying to discriminate between. For qualifier-source calls the operational meaning of the positioning is folded in (a measurable axis to clear or stay under, a reference to position against, a setting to evaluate inside, a craft template to match). For carver-source calls the population the call gates is named.

This is the document you reason against when choosing endpoint(s) within the category and shaping the parameter prose. The "search shape" language drives endpoint selection; the positioning prose drives numeric thresholds, ranges, and scoping clauses.

## `<expressions>`

A list of one or more short search-ready phrases — the seeds for endpoint parameters. Each phrase has been distilled into database-vocabulary and traces back to one dimension this call owns. They carry no polarity, no hedges, no comparison framing — those were upstream commitments.

For semantic / vector-bearing endpoints, each expression seeds one subquery; preserve the count (one per dimension is the committed shape) and tighten phrasing only as needed. For structured endpoints (METADATA, MEDIA_TYPE, AWARDS, FRANCHISE_STRUCTURE), each expression maps to a structured field, range, or boundary. For lexical endpoints (ENTITY, STUDIO, KEYWORD), pass the tokens through directly — do not invent additional ones.

When the call carries multiple expressions, the natural shape is **one endpoint firing with a multi-expression parameter slot**, not multiple firings of the same endpoint.

## What is intentionally not in the payload

- **The original user query.** You are not re-interpreting; the committed brief is the source of truth.
- **Query-level intent framing.** Already folded into `retrieval_intent` for whatever bearing it has on this call. Including it again risks pulling in aspects committed to other calls.
- **Sibling traits / cross-call context.** Cross-trait reasoning was the upstream stage's job. Each call is dispatched independently.
- **Match-mode and polarity.** Pre-committed by the upstream stage and stamped onto the wrapper after you emit it. Do not include them in your output and do not let polarity bleed into expression phrasing.
- **Salience.** Affects cross-trait reranking only, which is out of scope for this stage.
