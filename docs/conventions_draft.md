# Conventions Draft

Observed patterns staged for review. Remove entries you disagree
with, then run /solidify-draft-conventions to merge the rest into
docs/conventions.md.

Entries are added automatically during /safe-clear based on
patterns observed in the session.

## No re-export shims when moving modules
**Observed:** User explicitly rejected keeping old files as re-export shims after moving schemas to a new package. Said "No hacking by re-exporting moved files in their original file. Update at the source of each import directly."
**Proposed convention:** When moving a module to a new location, update all import sites directly. Never leave the old file as a re-export shim — delete it and fix every consumer.
**Sessions observed:** 1
