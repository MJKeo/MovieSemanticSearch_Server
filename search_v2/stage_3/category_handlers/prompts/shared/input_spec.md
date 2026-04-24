# Input spec

On each call, you receive a payload describing one atomic user requirement to decide on, along with surrounding context. The payload is serialized as XML in the user message. This section describes **what each field contains**; decision rules for what to emit in response live elsewhere in this prompt.

## `<raw_query>`

The user's original query, verbatim — word order, phrasing, and typos preserved. All other input fields are downstream restatements derived from this.

## `<overall_query_intention_exploration>`

A 2–4 sentence gloss from the upstream stage describing what the full query is asking for. Carries framing that colors individual requirements — occasion ("date night"), audience ("for the kids"), overall mood — even when those framings don't appear in the requirement atom itself.

## `<target_entry>`

The single atomic requirement this call is responsible for. Exactly one per call. Sub-fields:

- **`<captured_meaning>`** — a neutral, one-sentence observation of the requirement as it appears in the original query, before committing to a category.
- **`<category_name>`** — the category this requirement has been classified into. Already determined upstream; not something to re-classify.
- **`<fit_quality>`** — step 2's verdict on how well the category covers this requirement. See the vocabulary section above for interpretation.
- **`<atomic_rewrite>`** — the captured meaning re-expressed *as a category-grounded request*. Same content as `captured_meaning`, rephrased to foreground the category's lens. Specifics from the original query are preserved rather than generalized: "brother" stays "brother" (not broadened to "sibling"); "1990s" stays "1990s" (not broadened to "older").

## `<parent_fragment>`

The fragment of the user's query that produced `target_entry`. Sub-fields:

- **`<query_text>`** — the verbatim span of the user's query this fragment covers.
- **`<description>`** — a short restatement of what the fragment is asking for.
- **`<modifiers>`** — a list of polarity/role cues bound to the fragment. **These modifiers apply to your `target_entry` atom**: the parent fragment is the linguistic frame around the atom, so a modifier on the fragment shapes how the atom should be interpreted. Each modifier has:
    - `original_text` — the verbatim span of the modifier in the query.
    - `effect` — a terse note describing how this modifier shifts interpretation (e.g. "negates the following clause", "marks the subject as the director rather than actor").
    - `type` — one of:
        - **`polarity_modifier`** — flips or modulates sign/strength. Examples: *not*, *not too*, *without*, *preferably*, *ideally*.
        - **`role_marker`** — binds the attribute to a role or dimension. Examples: *starring*, *directed by*, *about*, *set in*, *based on*.

The parent fragment does **not** include its full atom list — only the one you are handling, which is surfaced separately above as `<target_entry>`. Any other atoms on this fragment belong to other handler calls.

## `<sibling_fragments>`

Every other fragment of the user's query besides the parent fragment (possibly empty if the query is a single fragment). Each has the same sub-fields as `<parent_fragment>` but with no atom list — just the framing, phrasing, and modifier cues. Provides cross-fragment context on what else the user is asking for and which framings have been committed to elsewhere in the query.
