# Occasion / self-experience goal / comfort-watch — additional notes

This category covers the **reason or situation for watching**, not what the movie is or how it feels moment-to-moment. The atom names a viewing occasion (date night, rainy Sunday, background watching, watch with my brother), a self-experience goal the user wants the movie to produce in them ("make me cry", "cheer me up", "challenge me", "something mindless"), a comfort-watch archetype ("go-to", "feel-better"), or a gateway / entry-level pull ("good first anime", "accessible arthouse").

## How the two endpoints split the work

- **SEMANTIC** is the load-bearing channel. It fires at the **sub-space level** — pick the sub-spaces the ask actually touches, not all three reflexively.
  - `watch_context` is the default sub-space for this category. `watch_scenarios` carries occasion and watch-partner framings; `self_experience_motivations` carries goal phrasings the user frames as what they want the movie to do to them ("make me cry", "cheer me up", "something mindless"); `external_motivations` carries social / companion pulls; `key_movie_feature_draws` carries gateway-style draws ("short runtime", "accessible entry point").
  - `viewer_experience` fires only when the ask implies a **specific emotional target** the during-viewing feel needs to deliver — "make me cry" implies cathartic / melancholic palette; "cheer me up" implies uplifting / warm palette. A bare "rainy Sunday movie" implies no emotional target and should leave `viewer_experience` silent.
  - `reception` fires only when the ask invokes **critic or audience labels** that reviewers use for this framing — "tearjerker", "comfort-rewatch", "crowd-pleaser", "accessible arthouse". Use `praised_qualities` for the label. A bare occasion ask with no reviewer-label framing leaves `reception` silent.
- **KEYWORD** contributes a narrow, precise signal: the registry's `ExperientialTag` members `TEARJERKER` and `FEEL_GOOD`. Fire additively when the self-experience goal maps cleanly to one of these definitions. If no registry member's definition names the goal, KEYWORD stays silent. Do not reach for nearby members (e.g. HAPPY_ENDING for "cheer me up", DRAMA for "make me cry") — those are different concepts and would silently broaden the result.

## Multi-endpoint discipline

"Make me cry" is the canonical all-channels-fire case: the watch_context motivation is named, the viewer_experience emotional target is implied, reviewer labels ("tearjerker") exist, and a KEYWORD tag matches. All four signal surfaces carry distinct, complementary evidence and fire together.

Narrow asks fire narrowly. "Rainy Sunday background movie" names the occasion only — fire `watch_context` alone, leave every other surface silent. Do not pad the fire just because the bucket permits multiple endpoints; each firing channel must carry signal the others cannot.

Gateway asks ("good first anime", "accessible arthouse") typically fire `watch_context` (the entry-point motivation and key draws) plus `reception` (accessibility is a reviewer-applied label). KEYWORD stays silent — the registry has no "gateway" member.

## Boundaries with nearby categories

- **Viewer experience / feel / tone (Cat 22).** Cat 22 is the **during-viewing aesthetic** — the movie IS dark, cerebral, whimsical, cozy-feeling. Cat 23 is the **reason or situation for watching** — the movie is FOR a cozy Sunday, FOR cheering me up. "Cozy movie" sits on the boundary: if the framing is aesthetic ("cozy-feeling movie"), it is Cat 22; if the framing is occasion or goal ("movie for a cozy night in"), it is Cat 23. The discriminator is whether the phrase describes the movie's interior feel or the viewer's external situation / intent. Upstream routed the atom here; trust the routing unless the captured meaning genuinely describes interior feel with no situational framing — in which case no-fire and let Cat 22 handle it.
- **Post-viewing resonance (Cat 26).** Cat 26 is **aftertaste** — "haunting", "stays with you", "can't stop thinking about it". Cat 23 is the **goal or occasion** for sitting down to watch — "something to make me think about life afterward" frames a goal, "a movie that haunts you after" frames aftertaste. If the phrase names lasting effect as the defining property of the movie, Cat 26 owns it.
- **Target audience (Cat 17).** Cat 17 is the **packaged audience tier** — family, for teens, for adults. Cat 23's "watch with my brother" is a **companion-specific occasion**, not a packaging label. The Semantic sub-fields differ in emphasis: Cat 17 fires `watch_context.watch_scenarios` from a packaging framing; Cat 23 fires `watch_context.watch_scenarios` + `self_experience_motivations` from an occasion / goal framing. If the atom is clearly an audience packaging, the upstream category label routed it wrong — return empty combination rather than fabricate.

## When to no-fire (every endpoint)

Return empty combination when the atom routed here is not actually an occasion, goal, comfort, or gateway framing. Common misroutes:

- An atom that is really a genre or plot request ("a heist movie", "something with a twist ending"). KEYWORD might technically carry a member for these, but that is a different category's work — firing KEYWORD from a misrouted genre atom silently converts Cat 23 into a Cat 11 / Cat 16 handler.
- An atom describing the movie's interior feel with no situational framing — that is Cat 22's territory.
- An atom that is too vague to anchor any sub-space ("something good", "a nice movie") with no occasion, no goal, no comfort archetype, no gateway pull. Inventing a watch_scenario the input does not support dilutes the query.

Record the reasoning in `overall_endpoint_fits` and leave each endpoint's breakdown at `should_run_endpoint: false`.

## The one principle

Fire exactly the sub-spaces the atom's framing calls for, and fire KEYWORD only when a registry member's definition genuinely names the self-experience goal. Narrow asks fire narrowly; multi-dimensional asks fire across multiple surfaces. Padding firing surfaces to match the bucket's capacity dilutes the result; collapsing a multi-dimensional goal onto one surface drops signal the bucket exists to preserve.
