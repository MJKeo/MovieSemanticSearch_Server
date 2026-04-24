# Objective

This category maps to **multiple candidate endpoints**, each introduced above. Each can individually answer the kind of question this category covers, but they answer *different versions* of it — firing more than one would mix answers to different questions rather than reinforce a single one.

Your task: pick which single endpoint (if any) best fits the target requirement, and fill in its parameters.

Work through the decision in order:

1. Break the target requirement into its discrete aspects. For each aspect, describe how every candidate endpoint could cover it, then name the best-covering endpoint and any gaps it leaves.
2. Use that per-aspect analysis to commit to the single endpoint that best fits the requirement overall — or decide that none of them does.
3. If one is picked, populate its parameter payload.

**No-fire is a valid and preferred outcome.** Upstream dispatch is not infallible — if no candidate is a genuine fit, returning `endpoint_to_run: "None"` is always better than picking the least bad option and inventing parameters.
