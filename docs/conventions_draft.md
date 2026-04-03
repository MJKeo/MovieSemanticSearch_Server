# Conventions Draft

Observed patterns staged for review. Remove entries you disagree
with, then run /solidify-draft-conventions to merge the rest into
docs/conventions.md.

Entries are added automatically during /safe-clear based on
patterns observed in the session.

## Labeled embedding text for short categorical fields
**Observed:** User approved adding semantic labels ("genres: ", "conflict: ", "themes: ") to short categorical fields in embedding text while leaving prose fields unlabeled. Backed by research (Anthropic contextual retrieval, Google Gemini docs). Applied to PlotAnalysisOutput.embedding_text() and ReceptionOutput.embedding_text().
**Proposed convention:** When building text for vector embedding, prefix short/ambiguous categorical fields with a semantic label ("field_name: value1, value2"). Leave prose fields (summaries, overviews) unlabeled — they are self-contextualizing. This helps the embedding model disambiguate the role of short values.
**Sessions observed:** 2

## Per-term normalization in embedding text, not whole-string
**Observed:** User directed normalizing each individual term before joining with ", " rather than running normalize_string() on the entire assembled embedding text. Applied to PlotAnalysisOutput.embedding_text() and ReceptionOutput.embedding_text() label fields.
**Proposed convention:** In embedding_text() methods, normalize each label/term individually before joining. Do not run normalize_string() on the final assembled string — this preserves label prefixes and structural formatting (colons, commas) while still normalizing the actual content.
**Sessions observed:** 2

## Metadata classes own their embedding representation
**Observed:** User explicitly directed moving embedding formatting logic from vector_text.py into embedding_text() methods. Applied to PlotAnalysisOutput ("move that to within the embedding_text method... that's a much cleaner approach") and ReceptionOutput. The vector_text functions became thin wrappers.
**Proposed convention:** Each *Output metadata class should own its embedding text assembly in its embedding_text() method. Vector text functions in vector_text.py should be thin wrappers that may enrich (e.g., merge genres, add tier) but delegate core formatting to the metadata class.
**Sessions observed:** 2

## No numeric scores in vector embedding text
**Observed:** User explicitly said "Do not add numbers into the vector" when Claude proposed including IMDB rating and Metacritic score alongside the reception tier label. Numeric scores are metadata filter territory, not embedding territory.
**Proposed convention:** Never include raw numeric scores (ratings, vote counts, years, budgets) in vector embedding text. Use qualitative labels derived from scores (e.g., "universally acclaimed" from a score >= 81) instead. Numeric filtering belongs in metadata scoring, not vector similarity.
**Sessions observed:** 1

## No re-export shims when moving modules
**Observed:** User explicitly rejected keeping old files as re-export shims after moving schemas to a new package. Said "No hacking by re-exporting moved files in their original file. Update at the source of each import directly."
**Proposed convention:** When moving a module to a new location, update all import sites directly. Never leave the old file as a re-export shim — delete it and fix every consumer.
**Sessions observed:** 1
