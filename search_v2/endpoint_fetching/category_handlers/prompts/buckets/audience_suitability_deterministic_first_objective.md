# Objective

This category maps to **multiple candidate endpoints that should each fire in parallel whenever they carry a real signal**, introduced above. Audience-suitability requirements are inherently multi-faceted: a single concept like "suitable for kids" or "no gore" is genuinely better served by a deterministic gate, an inclusion or exclusion scoring signal, and a semantic intensity / watch-context query all running in parallel than by any one of them alone. Overlap across endpoints is welcome — every endpoint that has a real signal to contribute should fire, even when its angle is already partially captured by another.

Your task: scope the suitability concept across every angle the expression exposes, then for each candidate endpoint decide whether it carries a real complementary signal toward this requirement and emit parameters for the endpoints that should fire.

Work through the decision in order:

1. **Scope the concept holistically.** Read the target requirement and enumerate every angle it exposes — hard ceilings, content categories the user wants more of, content categories the user wants to avoid, tone, watch-context. This is the opportunity inventory that the per-endpoint decisions will draw from.
2. **Per-endpoint fire-or-abstain.** For each candidate endpoint, ask whether it can carry a distinct slice of the inventory. Hard ceilings and clean content exclusions belong on deterministic gates; tags for desired or avoided content belong on inclusion/exclusion scoring; tone, intensity, and watch-context belong on the semantic channel. Fire every endpoint that clears the bar — overlap with another endpoint is not a reason to skip.
3. **Address every candidate explicitly.** Silent skipping is the failure mode this bucket exists to prevent. If an endpoint should not fire, say so deliberately rather than passing it over.

**Polarity rule: emit presence of an attribute, not direction.** Endpoint parameters describe what the content has. Whether that presence helps or hurts the user is decided when the signals are combined later — do not encode that decision into the parameter itself.

**Declining to fire — per-endpoint or for the whole combination — is a valid outcome** when an endpoint has nothing distinct to contribute. But the default posture is to fire every endpoint that carries real complementary signal, not to collapse to a single one out of conservatism.
