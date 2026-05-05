# Failure-mode guardrails

Before emitting, check for these failure modes. In each case, **leaving the affected endpoint's `{route}_parameters` null with the rationale captured in `intentionally_uncovered` is the correct response** — not fabricated parameters.

- **Ambiguous requirement.** The call's intent is too vague to point at concrete candidates in any walk. Do not invent specificity that is not in the call.
- **Out of scope for every endpoint.** No walk surfaces a clean fit. Leave `coverage_assignments` empty and name the unaddressable aspects in `intentionally_uncovered`.
- **Self-contradictory requirement.** Modifiers on the call flip the requirement against itself in a way the endpoints cannot express.

Beyond those, watch for these augmentation-specific traps:

- **Walking abstractly.** A walk that describes the endpoint's general fitness ("keyword can cover registry tags") instead of concrete candidates ("HORROR covers the broad scary signal; nothing covers the specific Christmas-eve framing") defeats the grounding the walk exists to enforce. Walks must name actual vector spaces / registry members / columns and explicit covers/misses prose.
- **Committing past the walk.** An assignment whose endpoint's walk surfaced no clean fit is a fabrication. The walks are the audit; commitments must be readable off them. If a deterministic walk shows no clean signal, do not assign that endpoint — leave its parameters null.
- **Leading with deterministic tags.** Semantic typically owns the bulk for this bucket because the calls are graded / experiential. Starting commitment from a deterministic-first read flattens evaluative meaning into a categorical label. Walk every endpoint, then commit — the structure of the schema enforces this if you populate it top-down.
- **Suppressing a clean deterministic signal because semantic already covers it.** Overlap is the point. When the deterministic walk shows a clean binary or canonical match, assign that endpoint even when the semantic slice already contains the meaning — the deterministic surface catches signals that the embeddings blur across.
- **Letting deterministic augmentation reshape the semantic slice.** Deterministic assignments reinforce semantic in parallel. They do not narrow, contradict, or filter it. If a deterministic assignment would change what the semantic call retrieves, drop the deterministic assignment rather than degrade the semantic slice.
- **Splitting one slice across endpoints.** If semantic cleanly owns an aspect, let it own that aspect alone. Fragmenting it across the deterministic endpoints is padding.

Make the coverage call **explicit** in `coverage_assignments` and `intentionally_uncovered` — name which parts each fired endpoint handles, and which (if any) intentionally go uncovered.
