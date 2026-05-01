# Additional objective notes - Title text lookup

Search for literal text in movie titles. This category is about the title
string itself: exact title, title contains a word/phrase, or title starts
with a word/phrase.

Keep the target text narrow. Strip framing words such as "called",
"titled", "with the word", and keep only the literal title fragment.
Do not interpret the fragment as a theme, franchise, person, genre, or
subject.

Use exact-match only when the intent asks for that exact title. Use
contains / starts-with when the user asks for words or phrases appearing
inside titles.

No-fire when the phrase is not title text: "star-studded", "Christmas
movies", "Bond films", "movies about Paris", or a full exact-title flow
that should already have been handled upstream.

