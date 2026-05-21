# Verification query suite used for the category_candidates
# minimum-count experiment.
#
# IMPORTANT: these queries come from
# `search_improvement_planning/rescore_overhal_queries.md`. They MUST
# NEVER be added to any system prompt, schema description, few-shot,
# or example set for any stage of the search pipeline — they are the
# evaluation harness and would lose their diagnostic value if the
# LLM had been pre-conditioned on them.

QUERIES: list[str] = [
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
