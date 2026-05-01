# Objective

This category maps to **multiple candidate endpoints with an explicit preference between them**, introduced above. One representation is preferred for this category — when it cleanly covers the request, it is the authoritative answer. Other representations exist as fallbacks for the parts of the request that fall outside the preferred one's reach.

The category-specific notes that follow define what "preferred" means for this category and when fallback is appropriate. Trust those notes — they encode the coverage judgment for the specific representations involved.

Your task: decide which representations should fire — possibly only the preferred one, possibly only a fallback, possibly both for different parts of the requirement — and fill in parameters for each that should fire.

Work through the decision in order:

1. Break the target requirement into its discrete aspects. For each aspect, judge whether the preferred representation cleanly covers it, partially covers it, or does not cover it at all.
2. From that aspect-level coverage, commit to one of three shapes:
   - **Full coverage** — emit only the preferred representation.
   - **Partial coverage** — emit the preferred representation for the part it covers, and a fallback for the part it does not.
   - **No coverage** — emit only a fallback.
3. Populate parameters for whichever representations should fire. Generate the minimum set that captures the requirement.

**Declining to fire any representation is a valid and preferred outcome.** If no representation genuinely fits the requirement, abstaining is always better than padding the response with plausible-but-noisy signal.
