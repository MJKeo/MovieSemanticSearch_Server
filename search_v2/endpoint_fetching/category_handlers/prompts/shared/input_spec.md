# Input spec

Each call delivers one CategoryCall (see vocabulary) serialized as XML in the user message. The payload carries only what's needed to translate the call: the retrieval intent and the expressions. Everything else — original query, sibling traits, polarity, upstream confidence — is intentionally absent. See "what is intentionally not in the payload" below.

## `<retrieval_intent>`

A 1–3 sentence brief authored by the upstream stage specifically for you. Names the dimension(s) being searched and what the call is trying to discriminate between, in plain user/database vocabulary. Carries every nuance the short `expressions` phrases can't carry on their own — modifier context, scoping clauses, and whatever makes this call distinct from adjacent ones the upstream stage may have spawned.

This is the source of truth for what the call retrieves. The bucket and endpoint sections below specify how to translate it into parameters.

## `<expressions>`

A list of one or more short database-vocabulary phrases. Each one has been distilled from the user's surface form — no hedges, no polarity words, no comparison framing — and traces back to one dimension this call owns.

The cardinality of the list is meaningful: each entry corresponds to one dimension this call owns. It does not, on its own, dictate the shape of the parameter payload — whether N expressions become N parameter slots, one merged slot, N targets, or N aspects depends on the endpoint and bucket. The endpoint section below specifies how expressions should be read for that endpoint.

## What is intentionally not in the payload

- **The original user query.** You are not re-interpreting; the committed brief is the source of truth.
- **Query-level intent framing.** Already folded into `retrieval_intent` for whatever bearing it has on this call. Including it again risks pulling in aspects committed to other calls.
- **Sibling traits / cross-call context.** Cross-trait reasoning was the upstream stage's job. Each call is dispatched independently.
- **Polarity.** Pre-committed by the upstream stage and stamped onto the wrapper after you emit. See the `polarity` entry in the vocabulary block.
- **Salience and upstream routing confidence.** Cross-trait reranking signals; not your concern. Translate the call as committed; do not condition on confidence.
