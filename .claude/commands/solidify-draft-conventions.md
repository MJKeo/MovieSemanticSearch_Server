Read docs/conventions_draft.md and docs/conventions.md.

For each entry in conventions_draft.md:
1. Check if it's already covered by an existing convention in
   conventions.md (may be worded differently but same intent)
2. If already covered, remove it from the draft
3. If not covered, add it to the appropriate section in
   conventions.md, using the same formatting style as existing
   entries

After processing all entries, clear conventions_draft.md back to
its header:
```
# Conventions Draft

Observed patterns staged for review. Remove entries you disagree
with, then run /solidify-draft-conventions to merge the rest into
docs/conventions.md.

Entries are added automatically during /safe-clear based on
patterns observed in the session.
```

Report what was promoted and what was removed as duplicates.

Important: Before running this command, the user should have
already reviewed conventions_draft.md and removed any entries
they disagree with. Do NOT ask for confirmation on individual
entries — assume everything still in the file is approved.
