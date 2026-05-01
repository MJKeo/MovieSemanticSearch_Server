# Additional objective notes - Person credit

Search for real people credited on the film: actor, director, writer,
producer, or composer. The expression names the person; retrieval_intent
decides the credit role and any prominence cue.

Prefer the most specific supported role the wording gives. "Starring" and
"with" are actor cues; "directed by", "written by", "produced by", and
"score by" pick their named roles. If the role is not stated, use the
endpoint's unknown/person-general path rather than guessing.

Only indexed film-credit roles belong here. Cinematographer, editor,
costume designer, production designer, VFX supervisor, and similar
below-the-line literal asks do not have this endpoint's postings.

No-fire when the target is a fictional character, title text, studio,
franchise, source-material author without film-credit framing, or a
style-transfer phrase rather than a literal credit.

