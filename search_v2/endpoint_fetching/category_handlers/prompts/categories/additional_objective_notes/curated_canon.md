# Curated canon / named list — additional notes

This category covers **membership in a specific named curated list**: Criterion Collection, AFI Top 100, Sight & Sound's greatest-films polls, IMDb Top 250, BFI, National Film Registry, "1001 Movies to See Before You Die", film-school canon, and similar publication- or institution-maintained canons. The system does not store list-membership as a structured attribute — matches come from Semantic prose that cites the list and from a soft numeric reception prior.

## Decoding what a named list implies is the primary task

A named list is shorthand for a reception shape the LLM must unpack. The prose you author for Semantic should reflect the list's actual aesthetic, not just repeat its name. Examples of what the decoding looks like:

- **Criterion Collection** → curated arthouse / international / restored-classics aesthetic; auteur-driven cinema valued for craft and cultural preservation.
- **AFI Top 100** → American mainstream canon; high-reception studio-era and prestige films broadly familiar to US audiences.
- **Sight & Sound greatest films** → critic-canonical; leans arthouse and international, with a slower, more formalist sensibility than popular canons.
- **IMDb Top 250** → broadly popular AND highly rated; audience-weighted mainstream canon.
- **National Film Registry** → US cultural / historical significance, not only aesthetic quality.
- **"1001 Movies to See Before You Die"** → broadly canonical across eras and movements, educational-breadth sensibility.

The decoding lets the Semantic prose reach list-cited movies even when the review text does not name the list verbatim. Use the list's actual aesthetic vocabulary in `reception.reception_summary` and `praised_qualities`; use canonical-stature framing in `plot_analysis.elevator_pitch` when the list carries identity-level prestige.

## How the two endpoints split the work

- **SEMANTIC — reception (central) and often plot_analysis (supporting).** `reception.reception_summary` and `praised_qualities` carry the list-citation language reviews actually use ("a Criterion pick", "AFI-honored", "canonized in Sight & Sound's top ten", "included in the National Film Registry"). `plot_analysis.elevator_pitch` picks up canonical-stature framings at the identity level when the list signals auteur / masterpiece standing. Pick `reception` as `primary_vector` — list citations land there more reliably than in plot_analysis.
- **METADATA — reception as a soft additive lift, never a gate.** Fire `target_attribute: reception` with `well_received` when the list broadly skews to high-reception films (AFI, IMDb Top 250, "1001 Movies"). Skip or weight it lightly for lists where the aesthetic frequently includes lower-scored arthouse picks (Sight & Sound, Criterion) — an arthouse critic-canon film can sit below a mainstream blockbuster on the generic reception scalar, and treating the scalar as a hard floor would reject legitimate list members. Polarity and match_mode stay `trait` / `positive` either way; the lift is additive and the prose carries the actual list semantics.

## When to no-fire

Return the empty combination when the interpretation step cannot honestly commit:

- **Unknown or fabricated list.** The user names a "list" that maps to no real canon ("my favorite critic's top 10", "the Reddit film club's list", a made-up-sounding title). Inventing list semantics is worse than silence.
- **Vague list-adjacent phrasing without a named list.** "Famous curated picks", "films everyone says to watch" with no named source — that is generic reception quality, which routes to Cat 25, not here. If it reached you anyway, no-fire.
- **The atom is really a different category.** "Oscar winners" is Cat 8 (awards), not a curated list. "Classic films" as generic canonical stature is Cat 25. If the atom is a misroute, silence.

Record the interpretation failure in `overall_endpoint_fits` and leave both endpoints at `should_run_endpoint: false`.

## Boundaries with nearby categories

- **Reception quality (Cat 25).** "Classic", "acclaimed", "highly regarded" as general stature with no named source → Cat 25. "Criterion Collection", "AFI Top 100", "Sight & Sound" as specific named lists → here. The discriminator is whether the user named a specific canon-maintaining publication or institution.
- **Awards (Cat 8).** Awards are formal ceremony records with per-award precision — "Oscar winners", "Palme d'Or", "Golden Globes" → Cat 8. Curated lists are publication or institutional canonization — Criterion, AFI, Sight & Sound, BFI, National Film Registry → here. Formal ceremony with a prize is Cat 8; editorial canon is Cat 28.

## The one principle

Decode what the named list means, then author Semantic prose in that list's actual aesthetic register and add the METADATA reception lift only when the list broadly skews high-reception. If the list is unknown or the atom is really a different category's ask, no-fire rather than inventing list semantics.
