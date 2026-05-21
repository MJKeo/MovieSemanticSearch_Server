"""
Concept tags evaluation movie set.

23 movies covering positive and hard-negative cases for every tag in the
concept_tags metadata generator. See concept_tags_table.md (same folder)
for the per-movie expected-tag breakdown.

tmdb_id is the join key (== movie_card.movie_id in Postgres). title and
year are for human reference — the canonical title in the database may
differ slightly (e.g. TMDB stores "Star Wars" rather than
"Star Wars: A New Hope" for the 1977 release).
"""

CONCEPT_TAGS_TEST_MOVIES: list[dict] = [
    {"title": "Kill Bill: Vol. 1",      "year": 2003, "tmdb_id": 24},
    {"title": "Fight Club",             "year": 1999, "tmdb_id": 550},
    {"title": "Frozen",                 "year": 2013, "tmdb_id": 109445},
    {"title": "Groundhog Day",          "year": 1993, "tmdb_id": 137},
    {"title": "Pulp Fiction",           "year": 1994, "tmdb_id": 680},
    {"title": "Deadpool",               "year": 2016, "tmdb_id": 293660},
    {"title": "Taken",                  "year": 2008, "tmdb_id": 8681},
    {"title": "The Conjuring",          "year": 2013, "tmdb_id": 138843},
    {"title": "Get Out",                "year": 2017, "tmdb_id": 419430},
    {"title": "12 Angry Men",           "year": 1957, "tmdb_id": 389},
    {"title": "The Mist",               "year": 2007, "tmdb_id": 5876},
    {"title": "Catch Me If You Can",    "year": 2002, "tmdb_id": 640},
    {"title": "Mad Max: Fury Road",     "year": 2015, "tmdb_id": 76341},
    {"title": "Rocky",                  "year": 1976, "tmdb_id": 1366},
    {"title": "Star Wars: A New Hope",  "year": 1977, "tmdb_id": 11},
    {"title": "Inception",              "year": 2010, "tmdb_id": 27205},
    {"title": "La La Land",             "year": 2016, "tmdb_id": 313369},
    {"title": "Marley & Me",            "year": 2008, "tmdb_id": 14306},
    {"title": "John Wick",              "year": 2014, "tmdb_id": 245891},
    {"title": "Paddington 2",           "year": 2017, "tmdb_id": 346648},
    {"title": "Erin Brockovich",        "year": 2000, "tmdb_id": 462},
    {"title": "A Serious Man",          "year": 2009, "tmdb_id": 12573},
    {"title": "The Graduate",           "year": 1967, "tmdb_id": 37247},
]
