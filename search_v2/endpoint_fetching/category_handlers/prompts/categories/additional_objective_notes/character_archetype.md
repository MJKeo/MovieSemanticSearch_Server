# Additional objective notes

## Category Target

Static character type or role pattern: anti-hero, femme fatale, lovable
rogue, reluctant hero, love-to-hate villain. Ask: "What type of
character must be present?"

## Endpoint policy

**Exactly one endpoint fires for this category — never both.** The
archetype itself *is* the target; there is no within-class refinement
for semantic to layer on top of a keyword commit (contrast
CENTRAL_TOPIC, where BIOGRAPHY tags the class and semantic carries the
specific subject within it). If a registry member cleanly names the
archetype, keyword carries the call alone and adding semantic only
dilutes by surfacing films whose prose loosely echoes the archetype
language without actually fitting. If no registry member cleanly names
it, semantic carries the call alone on narrative_techniques.

- **Keyword fires alone** when a single registry member directly
  defines the archetype (e.g., a dedicated `ANTI_HERO` tag for
  "anti-hero protagonist"). Semantic abstains.
- **Semantic fires alone** on narrative_techniques when the archetype
  is real but no registry member directly names it (e.g., "manic pixie
  dream girl", "lovable rogue" when no such tag exists). Keyword
  abstains.

Do not pick an adjacent or broader character tag to satisfy the
keyword path. A tag that misses the specific flavor is not coverage —
abstain on keyword and let semantic carry it.

Qualifiers that *seem* to justify firing both endpoints almost always
belong to a sibling category and should route there instead:
- "anti-hero with a tragic past" → archetype here, "tragic past" to
  STORY_THEMATIC_ARCHETYPE.
- "modern femme fatale" → archetype here, "modern" to NARRATIVE_SETTING.

Once siblings carry the qualifier, the residual archetype request is
clean-binary: tag or no tag.

## Body authoring (when semantic fires)

The archetype lands primarily in
`narrative_techniques.audience_character_perception`. Archetype-
adjacent traits may also pull `characterization_methods` or
`character_arcs` only when the trait *explicitly* grounds them
(otherwise leave those sub-fields empty).

**One trait → one term per active sub-field.** Restate the exact
archetype the user named in canonical craft-name register. Do NOT
emit a constellation of adjacent archetypes from memory or from any
example list. Each archetype names a *distinct* audience-character
relationship:

- "love-to-hate antagonist" — externally reviled yet enjoyed.
- "morally gray lead" — internally conflicted; mixed sympathy.
- "misunderstood outsider" — wrongly judged; audience sees true self.
- "sympathetic monster" — does monstrous things; audience pities.
- "lovable rogue" — breaks rules charmingly; audience roots for them.

These are NOT synonyms — they describe different relationships
between audience and character. A user asking for one is not asking
for the others. Emit one precise term that mirrors the user's
phrasing.

**No synonym padding.** Unlike `viewer_experience` /
`watch_context` (where ingest-side text deliberately repeats near-
duplicate terms), `narrative_techniques` term lists are precise
craft labels. Adding a second term means the user grounded a
second distinct archetype, not that you're rephrasing the first.

**Substitution test in the narrowest form.** Before adding any
archetype term, ask: "Would the user accept this term as the *same
thing* they said?" If not, drop it — it is a different archetype,
not a paraphrase.

## Boundaries

Sibling categories that own adjacent slices — upstream routing
sends each ask to its own home, so these boundaries are
expectations about what this handler will and will not receive,
not a fallback path. They affect how you read overlapping or
ambiguous siblings in the `<sibling_categories>` block.

- Specific named persona (James Bond, Hannibal Lecter) → NAMED_CHARACTER
  or CHARACTER_FRANCHISE.
- Character trajectory or theme (redemption arc, fall from grace) →
  STORY_THEMATIC_ARCHETYPE.
- Ensemble / single-lead structural configuration → NARRATIVE_DEVICES.
- Mere on-screen presence of a character type as a story element →
  ELEMENT_PRESENCE.
