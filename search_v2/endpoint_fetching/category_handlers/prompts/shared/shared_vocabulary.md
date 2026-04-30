# Vocabulary

Definitions of terms referenced throughout this prompt.

## `match_mode`

What kind of signal this finding is. One of two values:

- **`filter`** — a binary yes/no test. Decides whether a movie belongs in the results at all: a non-matching movie drops out entirely. Hard, gating effect.
- **`trait`** — a descriptive attribute. Colors the ranking of movies that already belong in the results: a non-matching movie still qualifies, it just scores lower. Soft, non-gating effect.

`match_mode` is orthogonal to `polarity`: both modes combine with either polarity. "Filter" does not mean "remove" — filter+positive keeps matches, filter+negative drops them.

## `polarity`

Whether the user wants a characteristic or wants to avoid it. One of two values:

- **`positive`** — the user wants this characteristic in the results (e.g. "with X", "horror", "starring Y").
- **`negative`** — the user wants to avoid this characteristic (e.g. "without X", "no horror", "not too Y").

Polarity is about *intent*, not grammar. A surface-negated phrasing can still be a positive request ("movies that aren't boring" asks for engaging movies).

## `fit_quality`

An **incoming signal from the upstream step-2 stage**, delivered on the input payload — not a value you compute. It reflects step 2's verdict on how well this category covers the atomic requirement at hand. Treat it as context to weigh, not as gospel: if your own reasoning over the endpoints in scope diverges from step 2's judgment, you can override.

- **`clean`** — step 2 judged this a full cover. Usually handle the requirement fully, unless your analysis of the endpoints suggests they can't actually deliver on that judgment.
- **`partial`** — step 2 judged this a partial cover; another entry on the same fragment covers the remainder. Use this as a signal to scope your response to your slice of the fragment — it is not a hard lockout on what you emit.
