# Cultural tradition / national cinema — additional notes

This category covers **named cinema traditions treated as aesthetic movements** — Bollywood, Korean cinema, Hong Kong action, Italian neorealism, French New Wave, Nordic noir, J-horror, Dogme 95. The target is the tradition itself, not the legal country of production and not where shooting happened.

## The two channels answer different questions

- **Keyword** tags the tradition as an aesthetic. The classification registry carries members for the well-indexed traditions (HINDI for Bollywood, CANTONESE for Hong Kong cinema, ITALIAN for the neorealist / giallo / Fellini tradition, JAPANESE for jidaigeki / kaiju / anime traditions, KOREAN for K-cinema, ANIME as a stylistic form, GIALLO and SPAGHETTI_WESTERN as specific Italian movements, and so on). When a registry member names the tradition, the tag is authoritative — it was populated at ingest to mark films that belong to that tradition even when legal paperwork says otherwise.
- **Metadata** anchors to `country_of_origin` (and, rarely, `audio_language`) as a **legal/financial proxy** for the tradition. It is a lossy proxy. A Hollywood-funded Hong Kong action movie carries US country_of_origin and is invisible to a country-only query; a French art film shot in English still sits under France. Use metadata only when the keyword registry has no tag for the tradition.

## How to pick between them

Compare the candidate tag's definition (not its label) against the requirement:

- **If a registry member's definition names the tradition** — fire Keyword. The tag is authoritative and captures aesthetically-aligned films the legal-origin proxy would miss. Falling back to country when a tag exists is a silent failure mode — the country column loses co-productions and diaspora films that the tag keeps.
- **If no registry member covers the tradition** — fire Metadata on `country_of_origin`, populated with the concrete country or countries that best approximate the tradition's geographic center. This is the best remaining signal for less-indexed cinemas (Senegalese cinema, Iranian New Wave, Czech New Wave, etc.) even though it will miss diaspora contributions.

Mutex means exactly one. When Keyword applies, Metadata adds nothing useful — the tag already subsumes the country set. When Metadata is the only available channel, Keyword has no tag to offer.

## Boundaries with nearby categories

- **Structured metadata (Cat 10).** A bare legal-origin attribute — "American production", "Japanese-produced film" as a paperwork fact — belongs to Cat 10. A tradition framing — "Japanese cinema", "French New Wave", "Hong Kong action" — belongs here. The discriminator is whether the phrasing names an aesthetic / movement or a production-paperwork attribute.
- **Filming location (Cat 13).** Where a crew physically shot is not a tradition. "Filmed in Korea" is Cat 13; "Korean cinema" is Cat 12. If the parent fragment centers on shoot location rather than aesthetic heritage, no-fire here.

## When to no-fire

- **Vague framing with no named tradition.** "Movies with a foreign feel" names no specific cinema and is not a country ask — neither endpoint can target it.
- **Misrouted location or legal-origin phrasing.** If the atom is really "filmed in Korea" or "American production" with no tradition cue, the dispatch was wrong. Return `endpoint_to_run: "None"` and capture the mismatch in `endpoint_coverage`.
- **Self-contradictory modifier no channel can express.** Polarity flips live on the wrapper — do not mutate the classification or country list to simulate negation.

The tightest failure mode: firing `country_of_origin` when the registry actually carries the tradition as a tag. Check the registry first; only fall back to metadata when no tag's definition names the tradition the user asked for.
