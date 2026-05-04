# Objective

This category maps to **a fixed pair of retrieval paths**, both introduced above. The user has named a referent that is inherently both a character and a franchise/universe — like a hero whose name also names the films they appear in. Both paths must fire because both are true of the referent: the same name carries character-presence signal and franchise-lineage signal at once.

This is not a routing decision and not a coverage decision. There is no question of which path fits better — both fit, and forcing the choice would drop signal.

Your task: do **two parallel walks** for the same referent, one on each side. The two walks are independent because their target indexes are different — the character side queries per-film cast strings, the franchise side queries series / universe / umbrella titles — but they share the same referent.

Work through it in this order:

1. **Character-side walk.** Fill in `character_form_exploration` using the literal template specified in its schema description (Films / Credit per film / Distinct forms). Recall how this character is credited on real cast blocks across 3-5 notable films — the queried form is often NOT the dominant credit (Tony Stark is the credited form in MCU films, not "Iron Man"; Henry Jones, Jr. appears in Last Crusade alongside "Indiana Jones"). Then list the distinct cast-list strings in `character_forms`.
2. **Franchise-side walk.** Fill in `franchise_form_exploration` using its own template (Series / Umbrella / Subgroups / Distinct forms). Then list the distinct franchise / series / universe / umbrella titles in `franchise_forms`.

Each walk is filled in literally per its schema template — do not produce a single generic referent description and copy it into both fields. The two sides target different indexes; using one prose sentence for both throws away the per-side specificity the bucket exists to capture.

**Declining to fire is valid only when the requirement does not actually name a character-franchise referent.** When a genuine referent is present, both paths fire by design. Either form list may be empty if one side genuinely has no matches (rare), but the corresponding exploration field still gets filled. Source-medium and adaptation flags are separate category calls — do not fold them into either walk here.
