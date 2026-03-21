Read docs/conventions_draft.md, docs/conventions.md, and
docs/personal_preferences.md.

Process each entry in conventions_draft.md ONE AT A TIME. For each
entry, present it to the user with:

1. Whether it's already covered by an existing convention in
   conventions.md (may be worded differently but same intent).
   If so, recommend removing it and explain which existing
   convention covers it.
2. If not already covered, propose a **generalized** version of
   the convention — abstract away the specific incident that
   triggered it and frame it as a broad guideline. The goal is
   to avoid accumulating many narrow, overlapping conventions.
   Check if an existing convention can be widened to cover the
   new case instead of adding a new entry.
3. Wait for the user's decision before moving to the next entry.
   The user can:
   - **Promote** — add to conventions.md
   - **Reject** — discard entirely
   - **Redirect to module** — add to a specific module doc in
     docs/modules/ instead of conventions.md. This is appropriate
     for conventions that are valid but scoped to a single module
     or subsystem. Ask which module doc if not obvious from context.
   - **Revise** — request rewording before promoting or redirecting

After all entries are processed, clear conventions_draft.md back
to its header:
```
# Conventions Draft

Observed patterns staged for review. Remove entries you disagree
with, then run /solidify-draft-conventions to merge the rest into
docs/conventions.md.

Entries are added automatically during /safe-clear based on
patterns observed in the session.
```

Finally, review the decisions the user made during this process
and the conventions themselves. Identify any broader patterns in
how the user thinks about system design or organization that
would be useful to remember. Propose these as additions to
docs/personal_preferences.md — ask for approval before writing.

Report what was promoted, what was redirected (and where), what
was removed as duplicates, and what preferences were extracted.
