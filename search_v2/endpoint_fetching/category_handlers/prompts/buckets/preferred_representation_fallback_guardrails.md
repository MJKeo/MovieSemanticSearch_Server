# Failure-mode guardrails

Before emitting, check for these failure modes. In each case, **abstaining or naming the gap explicitly is the correct response** — not fabricated parameters.

- **Ambiguous requirement.** The call's intent is too vague to point at concrete candidates in any walk. Do not invent specificity that is not in the call.
- **Out of scope for every endpoint.** No walk surfaces a clean fit. Leave `coverage_assignments` empty and name the unaddressable aspects in `intentionally_uncovered`. Do not pick the least-bad option.
- **Self-contradictory requirement.** Modifiers on the call flip the requirement against itself in a way no representation can express.

Beyond those, watch for these walk-vs-commitment traps:

- **Walking abstractly.** A walk that describes the endpoint's general fitness ("keyword can cover registry-style elements") instead of concrete candidates ("HORROR covers the genre signal; nothing in the registry covers clown specifically") defeats the grounding the walk exists to enforce. Walks must name actual registry members / vector spaces / columns and explicit covers/misses prose.
- **Committing past the walk.** An assignment whose endpoint's walk surfaced no clean fit is a fabrication. The walks are the audit; commitments must be readable off them. If the walk says "no clean fit," the commitment must either skip that endpoint or name the aspect in `intentionally_uncovered`.
- **Padding with the fallback when the preferred already covers.** When the preferred representation cleanly captures the requirement (its walk shows clean candidates and it appears alone in coverage_assignments), adding a fallback assignment mixes interpretations and adds noisy duplicate signal. Commit only the preferred.
- **Stopping at the preferred when uncovered intent remains.** When the preferred's walk handles only part of the requirement and the fallback's walk surfaces a clean fit for what's left, both should appear in `coverage_assignments` covering distinct slices.
- **Splitting one slice across endpoints.** If one endpoint cleanly owns an aspect, let it own that aspect alone. Fragmenting it across multiple assignments is padding.

Make the coverage call **explicit** in `coverage_assignments` and `intentionally_uncovered` — name which parts of the call each fired endpoint handles, and which (if any) intentionally go uncovered.
