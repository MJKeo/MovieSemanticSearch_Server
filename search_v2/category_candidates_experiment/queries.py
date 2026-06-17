# Query suites for the Step 3 trait-decomposition experiments.
#
# `QUERIES` is the ACTIVE suite the batch runners import. It currently
# holds the consolidation-redesign test set (see CONSOLIDATION_EXPERIMENT.md):
# the original diagnostic queries, a few edge cases that stress the new
# consolidation / inclusion-only logic, and a few regression guards whose
# handling should NOT change.
#
# `RESCORE_EVAL_QUERIES` is the original 25-query verification suite from
# `search_improvement_planning/rescore_overhal_queries.md`, preserved here
# for the earlier min-count experiment. IMPORTANT: those queries MUST NEVER
# be added to any system prompt, schema description, few-shot, or example set
# for any stage of the search pipeline — they are an evaluation harness and
# would lose their diagnostic value if the LLM had been pre-conditioned on
# them. The same discipline applies to every query in this file.

QUERIES: list[str] = [
    # --- Original diagnostic queries (the failure modes under study) ---
    "anime movie where loser guy gets hot girl",
    "family friendly serbian films",
    "serbian movies",
    "wholesome movies for kids",
    "movie where a loser guy pursues a popular girl",
    # --- Edge cases stressing consolidation + inclusion-only ---
    # negative-polarity content axis + genre (SENSITIVE_CONTENT must stay
    # usable as inclusion under negative polarity; genre stays clean)
    "scary horror movie that isn't too gory",
    # nationality + genre (audio-language trap avoidance; legit two-trait)
    "japanese horror films",
    # single coherent story shape (should stay whole, not fragment)
    "movie about a washed-up boxer making a comeback",
    # audience/content inclusion framing ("clean"/"no swearing")
    "clean comedy with no swearing",
    # --- Regression guards (handling should NOT change) ---
    # clean single entity lookup
    "movies starring Tom Hanks",
    # multi-trait, all untouched categories (awards + genre + era)
    "Oscar winning war films from the 1990s",
    # multi-facet figurative trait — guards against OVER-consolidation
    "hidden gem thrillers",
]


# Original 25-query verification suite — preserved, not active. See header.
RESCORE_EVAL_QUERIES: list[str] = [
    "heartwarming holiday films",
    "films with a bittersweet melancholic tone",
    "forgotten gems with brilliant performances",
    "wholesome family movie night picks",
    "intense action thrillers but not too bloody",
    "movies featuring elephants",
    "movies about marathons",
    "films with sentient AI",
    "revenge stories with anti-heroes",
    "cyberpunk dystopias",
    "historical war epics",
    "mind-bending puzzle films about consciousness",
    "comedy musicals about teenage romance",
    "obscure indie passion projects",
    "Studio Ghibli style hand-drawn fantasies",
    "brutal MMA fight movies",
    "gritty crime sagas",
    "like Donnie Darko but funnier",
    "Wes Anderson aesthetic coming-of-age",
    "dark gritty antihero comic-book films",
    "atmospheric folk horror",
    "films about grief and reconciliation",
    "slow-burn psychological mysteries",
    "coming-of-age road trips not too sappy",
    "unreliable narrator with a twist ending",
]


def slugify_first_four(query: str) -> str:
    """Return a filesystem-safe slug built from the first four words
    of the query. Lowercased; non-alphanumerics dropped.

    Examples:
        "heartwarming holiday films" -> "heartwarming_holiday_films"
        "films with a bittersweet melancholic tone" -> "films_with_a_bittersweet"
    """
    words = query.split()[:4]
    pieces: list[str] = []
    for word in words:
        cleaned = "".join(ch for ch in word.lower() if ch.isalnum())
        if cleaned:
            pieces.append(cleaned)
    return "_".join(pieces) if pieces else "query"
